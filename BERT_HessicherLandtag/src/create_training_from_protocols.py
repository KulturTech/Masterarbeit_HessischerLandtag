"""
Pipeline: Hessen-Protokolle (protocols.csv) → Trainingsdaten

Schritte:
1. Hessen-Protokolle aus protocols.csv filtern (state == "he")
2. PDFs herunterladen
3. Text extrahieren (pdfplumber)
4. Text in Segmente aufteilen (~400 Wörter)
5. Fine-tuned Modell für Silver Labels anwenden
6. Balanciertes Trainings-CSV speichern
"""

import pandas as pd
import requests
import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import re
import time
import io

# ── Konfiguration ──────────────────────────────────────────────────────────────
BASE_DIR = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
PROTOCOLS_CSV   = BASE_DIR / "Data" / "protocols.csv"
PDF_CACHE_DIR   = BASE_DIR / "Data" / "protocols_pdf_cache"
OUTPUT_DIR      = BASE_DIR / "Data" / "training"
OUTPUT_FILE     = OUTPUT_DIR / "labeled_data_hessen_protocols.csv"

MODEL_DIR       = BASE_DIR / "fine_tuned_model_cv" / "best_model"

CONFIDENCE_THRESHOLD    = 0.80   # Nur Vorhersagen über diesem Wert
SEGMENT_WORD_LIMIT      = 400    # Maximale Wörter pro Segment
SAMPLE_SIZE_PER_LABEL   = 1000   # Max. Samples pro Label im finalen Dataset
REQUEST_TIMEOUT         = 30     # Sekunden pro PDF-Download
REQUEST_DELAY           = 0.5    # Pause zwischen Downloads (Rate-Limiting)
# ──────────────────────────────────────────────────────────────────────────────


def load_hessen_protocols(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    he = df[df["state"] == "he"].copy()
    print(f"Hessen-Protokolle gefunden: {len(he)}")
    return he


def download_pdf(url: str, cache_dir: Path, protocol_id: str) -> Path | None:
    """PDF herunterladen und cachen. Gibt Pfad zurück oder None bei Fehler."""
    cache_path = cache_dir / f"{protocol_id}.pdf"
    if cache_path.exists():
        return cache_path
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers={
            "User-Agent": "Mozilla/5.0 (research project)"
        })
        if response.status_code == 200 and b"%PDF" in response.content[:10]:
            cache_path.write_bytes(response.content)
            return cache_path
        else:
            print(f"  [SKIP] {protocol_id}: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"  [FEHLER] {protocol_id}: {e}")
        return None


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Text aus PDF extrahieren mit pdfplumber."""
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
    """Grundlegende Textbereinigung."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # Seitenzahlen
    return text.strip()


def split_into_segments(text: str, max_words: int = SEGMENT_WORD_LIMIT) -> list[str]:
    """Text in Paragraphen aufteilen, max. max_words Wörter pro Segment."""
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    segments = []
    current = []
    current_len = 0

    for para in paragraphs:
        words = para.split()
        if current_len + len(words) > max_words and current:
            segments.append(" ".join(current))
            current = words
            current_len = len(words)
        else:
            current.extend(words)
            current_len += len(words)

    if current:
        segments.append(" ".join(current))

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

    # 3. PDFs herunterladen, Text extrahieren, segmentieren
    print("\n" + "=" * 70)
    print("SCHRITT 3: PDFs herunterladen & Text extrahieren")
    print("=" * 70)

    all_segments = []
    n_total = len(protocols)

    for idx, row in protocols.iterrows():
        protocol_id = row["id"]
        url = row["url"]
        progress = f"[{protocols.index.get_loc(idx) + 1}/{n_total}]"

        print(f"{progress} {protocol_id} ...", end=" ", flush=True)

        pdf_path = download_pdf(url, PDF_CACHE_DIR, protocol_id)
        if pdf_path is None:
            continue

        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            print("kein Text")
            continue

        text = clean_text(raw_text)
        segments = split_into_segments(text)
        all_segments.extend(segments)
        print(f"OK ({len(segments)} Segmente)")

        time.sleep(REQUEST_DELAY)

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
