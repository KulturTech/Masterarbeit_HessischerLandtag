"""
LRP-Erklärungen für das BERT Hate-Speech-Modell
================================================
Layer-wise Relevance Propagation (Captum) auf Token-Ebene.

Regeln:
  - Linear-Schichten  → EpsilonRule (ε=1e-6)
  - LayerNorm, GELU, Dropout, sonstige → IdentityRule (Pass-through)

Standard-Eingabe: Data/evaluation/model_confirmed_hate_migration.csv
  (49 vom Modell bestätigte HATE-Redebeiträge, sortiert nach p_hate)

Ausgabe:
  - HTML-Dateien mit Token-Hervorhebung   →  Data/evaluation/lrp/
  - PNG Bar-Plot Top-Token                →  Data/evaluation/lrp/
  - CSV mit Token-Relevanzen              →  Data/evaluation/lrp/lrp_weights.csv

Verwendung:
  python src/lrp_explanations.py
  python src/lrp_explanations.py --n 20
  python src/lrp_explanations.py --csv eigene.csv --n 5
"""

import argparse
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Pfade & Konfiguration
# ---------------------------------------------------------------------------
BASE_DIR    = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
MODEL_DIR   = BASE_DIR / "fine_tuned_model_retrain" / "best_model"
DEFAULT_CSV = BASE_DIR / "Data" / "evaluation" / "model_confirmed_hate_migration.csv"
OUT_DIR     = BASE_DIR / "Data" / "evaluation" / "lrp"

TARGET_CLASS = "HATE"


# ---------------------------------------------------------------------------
# Embedding-Wrapper (LRP braucht Float-Inputs, keine Integer)
# ---------------------------------------------------------------------------
class BertEmbeddingWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, attention_mask):
        return self.model(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask).logits


# ---------------------------------------------------------------------------
# LRP-Regeln setzen
# ---------------------------------------------------------------------------
def set_lrp_rules(model):
    from captum.attr._utils.lrp_rules import EpsilonRule, IdentityRule

    epsilon_rule   = EpsilonRule(epsilon=1e-6)
    identity_rule  = IdentityRule()

    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.rule = epsilon_rule
        else:
            # LayerNorm, GELU, Dropout, Embedding, Attention-Gewichte usw.
            module.rule = identity_rule


# ---------------------------------------------------------------------------
# Modell laden
# ---------------------------------------------------------------------------
def load_model(model_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Lade Modell von: {model_dir}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    id2label    = model.config.id2label
    label_to_idx = {v: k for k, v in id2label.items()}
    hate_idx    = label_to_idx.get(TARGET_CLASS, 1)

    print(f"Label-Mapping: {id2label}  →  HATE-Index: {hate_idx}")
    return tokenizer, model, device, hate_idx


# ---------------------------------------------------------------------------
# LRP für einen Text
# ---------------------------------------------------------------------------
def compute_lrp(text: str, tokenizer, model, wrapper, lrp_attr, device, hate_idx):
    from captum.attr import LRP

    inputs = tokenizer(
        text, return_tensors="pt", padding=True,
        truncation=True, max_length=512
    ).to(device)

    # Embeddings berechnen (Float-Tensor für LRP)
    with torch.no_grad():
        embeds = model.bert.embeddings(input_ids=inputs.input_ids)
    embeds = embeds.detach().requires_grad_(True)

    attributions = lrp_attr.attribute(
        inputs=embeds,
        additional_forward_args=(inputs.attention_mask,),
        target=hate_idx,
    )

    # Relevanz pro Token: Summe über Embedding-Dimension
    token_relevance = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0].cpu())

    # Wahrscheinlichkeit (für Ausgabe)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    return tokens, token_relevance, probs


# ---------------------------------------------------------------------------
# HTML-Visualisierung
# ---------------------------------------------------------------------------
def _relevance_to_color(value: float, vmax: float) -> str:
    norm = max(-1.0, min(1.0, value / (vmax + 1e-9)))
    if norm > 0:
        r, g, b = 215, int(215 * (1 - norm)), int(215 * (1 - norm))
    else:
        r, g, b = int(215 * (1 + norm)), int(215 * (1 + norm)), 215
    return f"rgb({r},{g},{b})"


def save_html(tokens: list, relevance: np.ndarray, text_idx: int,
              pred_label: str, p_hate: float, out_dir: Path,
              speaker: str = "", party: str = ""):
    vmax = np.abs(relevance).max()
    spans = []
    for tok, rel in zip(tokens, relevance):
        display = tok.replace("##", "").replace("▁", " ")
        color   = _relevance_to_color(rel, vmax)
        title   = f"{rel:+.4f}"
        spans.append(
            f'<span style="background:{color};padding:2px 4px;margin:1px;'
            f'border-radius:3px;font-size:14px" title="{title}">{display}</span>'
        )

    meta = f"{speaker} ({party}) — " if speaker else ""
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>LRP {text_idx:03d}</title></head>
<body style="font-family:sans-serif;max-width:900px;margin:40px auto">
<h2>LRP Relevance #{text_idx:03d}</h2>
<p><b>{meta}Vorhersage: {pred_label}</b> &nbsp; P(HATE)={p_hate:.4f}</p>
<p style="line-height:2.2">{''.join(spans)}</p>
<p style="color:#888;font-size:11px">
  Rot = HATE-fördernd &nbsp;|&nbsp; Blau = HATE-dämpfend
</p>
</body></html>"""

    path = out_dir / f"lrp_text_{text_idx:03d}.html"
    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Hauptschleife
# ---------------------------------------------------------------------------
def explain_texts(df: pd.DataFrame, text_col: str,
                  tokenizer, model, device, hate_idx, out_dir: Path):
    from captum.attr import LRP

    set_lrp_rules(model)
    wrapper  = BertEmbeddingWrapper(model).to(device)
    lrp_attr = LRP(wrapper)

    out_dir.mkdir(parents=True, exist_ok=True)
    texts = df[text_col].tolist()
    print(f"\nBerechne LRP-Relevanzen für {len(texts)} Redebeiträge ...")

    all_rows = []

    for i, text in enumerate(texts):
        meta       = df.iloc[i]
        speaker    = str(meta.get("speaker", "")) if hasattr(meta, "get") else ""
        party      = str(meta.get("party",   "")) if hasattr(meta, "get") else ""
        p_hate_src = float(meta.get("p_hate", 0.0)) if hasattr(meta, "get") else 0.0

        print(f"\n[{i+1}/{len(texts)}] {speaker} ({party})" if speaker
              else f"\n[{i+1}/{len(texts)}]")
        print(f"  '{text[:80]}{'...' if len(text) > 80 else ''}'")

        try:
            tokens, relevance, probs = compute_lrp(
                text, tokenizer, model, wrapper, lrp_attr, device, hate_idx
            )
        except Exception as exc:
            print(f"  [FEHLER] {exc}")
            continue

        p_hate     = float(probs[hate_idx])
        pred_label = TARGET_CLASS if p_hate >= 0.5 else "NON_HATE"

        print(f"  Vorhersage: {pred_label}  (P(HATE)={p_hate:.4f})")

        # Top-Token
        top_idx = np.argsort(np.abs(relevance))[::-1][:10]
        print("  Top-Token (|LRP|):")
        for idx in top_idx:
            rel = relevance[idx]
            print(f"    {'+'if rel>0 else '-'}{abs(rel):.4f}  '{tokens[idx]}'")

        # HTML
        html_path = save_html(tokens, relevance, i + 1, pred_label, p_hate,
                               out_dir, speaker, party)
        print(f"  HTML gespeichert: {html_path.name}")

        # CSV-Zeilen
        for tok, rel in zip(tokens, relevance):
            all_rows.append({
                "text_idx":    i + 1,
                "speaker":     speaker,
                "party":       party,
                "p_hate_orig": round(p_hate_src, 4),
                "pred_label":  pred_label,
                "p_hate":      round(p_hate, 4),
                "text_preview": text[:120],
                "token":       tok,
                "lrp_value":   round(float(rel), 6),
            })

    if all_rows:
        df_out = pd.DataFrame(all_rows)
        csv_path = out_dir / "lrp_weights.csv"
        df_out.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"\nAlle LRP-Relevanzen gespeichert: {csv_path}")

        top_hate = (
            df_out[df_out["lrp_value"] > 0]
            .groupby("token")["lrp_value"]
            .mean()
            .sort_values(ascending=False)
            .head(20)
        )
        print("\nTop-Token mit stärkstem HATE-Einfluss (Ø LRP über alle Reden):")
        print(top_hate.to_string())

        _save_bar_plot(top_hate, out_dir)


def _save_bar_plot(top_series: pd.Series, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_series.index[::-1], top_series.values[::-1], color="#d73027")
    ax.set_xlabel("Ø LRP-Relevanz (HATE-Klasse)")
    ax.set_title("Top-20 Token nach LRP-Einfluss\n(Ø über alle erklärten Redebeiträge)",
                 fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    out = out_dir / "lrp_top_tokens_bar.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Bar-Plot gespeichert: {out}")


# ---------------------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------------------
_TOC_RE = re.compile(r'[A-ZÄÖÜ][a-zäöüß]+(?:\s+\S+){0,4}\.{5,}\d', re.MULTILINE)

def is_toc_page(text: str) -> bool:
    return len(_TOC_RE.findall(str(text))) >= 3


def load_texts(csv_path: Path, n: int, label_filter: str | None,
               no_toc_filter: bool) -> pd.DataFrame:
    df = pd.read_parquet(str(csv_path)) if str(csv_path).endswith(".parquet") \
         else pd.read_csv(csv_path)

    text_col  = next((c for c in ["text", "context"] if c in df.columns), None)
    label_col = next((c for c in ["label", "hate_label"] if c in df.columns), None)

    if not text_col:
        raise ValueError(f"Keine 'text'-Spalte. Gefunden: {df.columns.tolist()}")

    df = df[df[text_col].notna()]
    df = df[df[text_col].str.split().str.len() >= 20]

    if label_filter and label_col:
        df = df[df[label_col] == label_filter]
        print(f"Gefiltert auf '{label_filter}': {len(df)} Texte")

    if not no_toc_filter:
        before = len(df)
        df = df[~df[text_col].apply(is_toc_page)]
        if before - len(df):
            print(f"TOC-Filter: {before - len(df)} Seiten entfernt")

    if "p_hate" in df.columns:
        df = df.sort_values("p_hate", ascending=False)

    return df.head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="LRP-Erklärungen für BERT Hate-Speech-Modell")
    parser.add_argument("--text",      type=str, default=None,
                        help="Einzelner Text direkt als Argument (überschreibt --csv)")
    parser.add_argument("--csv",       type=str, default=str(DEFAULT_CSV),
                        help=f"Pfad zur CSV oder Parquet (Standard: {DEFAULT_CSV.name})")
    parser.add_argument("--n",         type=int, default=10,
                        help="Anzahl zu erklärender Texte aus CSV (Standard: 10)")
    parser.add_argument("--model",     type=str, default=str(MODEL_DIR),
                        help="Pfad zum Modellverzeichnis")
    parser.add_argument("--filter",    type=str, default=None,
                        choices=["HATE", "NON_HATE"],
                        help="Nur Texte mit diesem Label laden (wenn label-Spalte vorhanden)")
    parser.add_argument("--no-filter", action="store_true",
                        help="TOC-Filter deaktivieren")
    return parser.parse_known_args()[0]


def main():
    args = parse_args()

    tokenizer, model, device, hate_idx = load_model(Path(args.model))

    if args.text:
        # Einzelner Text direkt per CLI
        print(f"\nEinzeltext: '{args.text[:80]}...'")
        df = pd.DataFrame([{"text": args.text}])
        text_col = "text"
    else:
        csv_path = Path(args.csv)
        print(f"\nLade Texte aus: {csv_path.name}")
        df = load_texts(csv_path, args.n, args.filter, args.no_filter)
        text_col = next((c for c in ["text", "context"] if c in df.columns))

        if "speaker" in df.columns:
            print("\nRedner in diesem Lauf:")
            for _, row in df.iterrows():
                print(f"  {row['speaker']} ({row.get('party','?')})  p_hate={row.get('p_hate','?')}")

    print(f"\nAusgabe: {OUT_DIR}")
    print("=" * 70)

    explain_texts(df, text_col, tokenizer, model, device, hate_idx, OUT_DIR)

    print("\n" + "=" * 70)
    print("Fertig! HTML-Dateien im Browser öffnen für interaktive Token-Ansicht.")
    print(f"  {OUT_DIR}")


if __name__ == "__main__":
    main()
