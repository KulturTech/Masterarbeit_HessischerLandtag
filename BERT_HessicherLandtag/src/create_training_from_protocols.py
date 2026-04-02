"""
Pipeline: Hessen-Protokolle (protocols.csv) → Trainingsdaten

Schritte:
1. Hessen-Protokolle aus protocols.csv filtern (state == "he")
2. PDFs parallel herunterladen
3. Text extrahieren (pdfplumber)
4. Text in Segmente aufteilen (~400 Wörter, sliding window)
5. Fine-tuned Modell für Silver Labels anwenden
6. Balanciertes Trainings-CSV speichern
"""

import pandas as pd
import requests
import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time

# ── Konfiguration ──────────────────────────────────────────────────────────────
BASE_DIR = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
PROTOCOLS_CSV   = BASE_DIR / "Data" / "protocols.csv"
PDF_CACHE_DIR   = BASE_DIR / "Data" / "protocols_pdf_cache"
OUTPUT_DIR      = BASE_DIR / "Data" / "training"
OUTPUT_FILE     = OUTPUT_DIR / "labeled_data_hessen_protocols.csv"

MODEL_DIR       = BASE_DIR / "fine_tuned_model_cv" / "best_model"

CONFIDENCE_THRESHOLD    = 0.80   # Nur Vorhersagen über diesem Wert
SEGMENT_WORD_LIMIT      = 350    # Wörter pro Segment
SEGMENT_OVERLAP         = 50     # Überlappung zwischen Segmenten
MIN_SEGMENT_WORDS       = 30     # Minimale Wörter damit Segment behalten wird
SAMPLE_SIZE_PER_LABEL   = 1000   # Max. Samples pro Label im finalen Dataset
REQUEST_TIMEOUT         = 30     # Sekunden pro PDF-Download
DOWNLOAD_WORKERS        = 8      # Parallele Downloads
# ──────────────────────────────────────────────────────────────────────────────


def load_hessen_protocols(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    he = df[df["state"] == "he"].copy()
    print(f"Hessen-Protokolle gefunden: {len(he)}")
    return he


def download_pdf(args) -> tuple[str, Path | None]:
    """PDF herunterladen und cachen. Gibt (protocol_id, Pfad) zurück."""
    protocol_id, url, cache_dir = args
    cache_path = cache_dir / f"{protocol_id}.pdf"
    if cache_path.exists():
        return protocol_id, cache_path
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers={
            "User-Agent": "Mozilla/5.0 (research project)"
        })
        if response.status_code == 200 and b"%PDF" in response.content[:10]:
            cache_path.write_bytes(response.content)
            return protocol_id, cache_path
        else:
            return protocol_id, None
    except Exception:
        return protocol_id, None


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Text aus allen Seiten extrahieren und zusammenführen."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)
    except Exception as e:
        print(f"  [FEHLER] PDF-Extraktion {pdf_path.name}: {e}")
        return ""


def clean_text(text: str) -> str:
    """Grundlegende Textbereinigung für parlamentarische Protokolle."""
    # Seitenzahlen (alleinstehende Zahlen)
    text = re.sub(r"(?m)^\s*\d{1,5}\s*$", " ", text)
    # Mehrfache Leerzeilen reduzieren
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Mehrfache Leerzeichen
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_into_segments(text: str,
                        max_words: int = SEGMENT_WORD_LIMIT,
                        overlap: int = SEGMENT_OVERLAP,
                        min_words: int = MIN_SEGMENT_WORDS) -> list[str]:
    """
    Sliding-window Segmentierung auf Wortebene.
    Funktioniert unabhängig von Paragraph-Struktur.
    """
    words = text.split()
    if len(words) < min_words:
        return []

    segments = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        segment = " ".join(words[start:end])
        if len(segment.split()) >= min_words:
            segments.append(segment)
        if end == len(words):
            break
        start += max_words - overlap

    return segments


def classify_segments(segments: list[str], model, tokenizer, device, batch_size=16) -> list[dict]:
    """Segmente mit dem fine-tuned Modell klassifizieren."""
    results = []
    model.eval()

    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted = torch.argmax(probs, dim=-1)
            scores = probs[range(len(batch)), predicted]

        for text, label_id, score in zip(batch, predicted.cpu(), scores.cpu()):
            label = "HATE" if label_id.item() == 1 else "NON_HATE"
            results.append({
                "text": text,
                "label": label,
                "score": round(score.item(), 4)
            })

    return results


def main():
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Protokolle laden
    print("=" * 70)
    print("SCHRITT 1: Hessen-Protokolle laden")
    print("=" * 70)
    protocols = load_hessen_protocols(PROTOCOLS_CSV)

    # 2. Modell laden
    print("\n" + "=" * 70)
    print("SCHRITT 2: Modell laden")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
    print("Modell geladen.")

    # 3. PDFs parallel herunterladen
    print("\n" + "=" * 70)
    print("SCHRITT 3a: PDFs herunterladen (parallel)")
    print("=" * 70)

    download_args = [
        (row["id"], row["url"], PDF_CACHE_DIR)
        for _, row in protocols.iterrows()
    ]

    pdf_paths: dict[str, Path] = {}
    n_total = len(download_args)
    done = 0

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(download_pdf, args): args[0] for args in download_args}
        for future in as_completed(futures):
            protocol_id, path = future.result()
            done += 1
            status = "OK" if path else "FEHLER"
            if path:
                pdf_paths[protocol_id] = path
            if done % 20 == 0 or done == n_total:
                print(f"  {done}/{n_total} heruntergeladen ({len(pdf_paths)} erfolgreich)")

    print(f"\nPDFs gecacht: {len(pdf_paths)}/{n_total}")

    # 3b. Text extrahieren & segmentieren
    print("\n" + "=" * 70)
    print("SCHRITT 3b: Text extrahieren & segmentieren")
    print("=" * 70)

    all_segments = []
    for i, (protocol_id, pdf_path) in enumerate(pdf_paths.items(), 1):
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            continue
        text = clean_text(raw_text)
        segments = split_into_segments(text)
        all_segments.extend(segments)
        if i % 50 == 0 or i == len(pdf_paths):
            print(f"  {i}/{len(pdf_paths)} extrahiert | Segmente bisher: {len(all_segments)}")

    print(f"\nSegmente gesamt: {len(all_segments)}")

    if not all_segments:
        print("Keine Segmente extrahiert. Abbruch.")
        return

    # 4. Klassifizieren
    print("\n" + "=" * 70)
    print("SCHRITT 4: Klassifizierung mit BERT-Modell")
    print("=" * 70)
    predictions = classify_segments(all_segments, model, tokenizer, device)
    pred_df = pd.DataFrame(predictions)
    print(f"Klassifiziert: {len(pred_df)}")
    print(pred_df["label"].value_counts())

    # 5. High-confidence filtern & balancieren
    print("\n" + "=" * 70)
    print("SCHRITT 5: Filtern & Balancieren")
    print("=" * 70)
    high_conf = pred_df[pred_df["score"] >= CONFIDENCE_THRESHOLD].copy()
    print(f"High-confidence (>= {CONFIDENCE_THRESHOLD}): {len(high_conf)}")
    print(high_conf["label"].value_counts())

    balanced = []
    for label in high_conf["label"].unique():
        subset = high_conf[high_conf["label"] == label]
        n = min(SAMPLE_SIZE_PER_LABEL, len(subset))
        balanced.append(subset.sample(n=n, random_state=42))
        print(f"  {label}: {n} Samples")

    result_df = pd.concat(balanced, ignore_index=True)
    result_df = result_df[["text", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)

    # 6. Speichern
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Trainingsdaten gespeichert: {OUTPUT_FILE}")
    print(f"     Gesamt: {len(result_df)} Samples")
    print(result_df["label"].value_counts())


if __name__ == "__main__":
    main()
