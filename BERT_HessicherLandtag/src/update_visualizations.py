"""
1. Alle Dokumente mit neuem Modell (fine_tuned_model_combined) neu klassifizieren
2. Alle Visualisierungen regenerieren
"""
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

BASE_DIR   = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
MODEL_DIR  = BASE_DIR / "fine_tuned_model_combined"
INPUT_PATH = BASE_DIR / "Data" / "prep_v1" / "all_docs_classified.parquet"
OUTPUT_PARQUET = BASE_DIR / "Data" / "prep_v1" / "all_docs_classified.parquet"
VIZ_DIR    = BASE_DIR / "Data" / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
DPI = 300

# ── 1. Neu klassifizieren ──────────────────────────────────────────────────────
print("=" * 65)
print("SCHRITT 1: Neu-Klassifizierung mit fine_tuned_model_combined")
print("=" * 65)

df = pd.read_parquet(INPUT_PATH)
print(f"Dokumente: {len(df)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
model.eval()
print("Modell geladen.")

texts = df['text'].tolist()
labels, scores = [], []

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i + BATCH_SIZE]
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        sc = probs[range(len(batch)), preds]
    labels.extend(["HATE" if p.item() == 1 else "NON_HATE" for p in preds])
    scores.extend([round(s.item(), 4) for s in sc])
    if (i // BATCH_SIZE + 1) % 50 == 0:
        print(f"  {i + len(batch)}/{len(texts)} klassifiziert")

df['label'] = labels
df['score'] = scores
df.to_parquet(OUTPUT_PARQUET)
print(f"\nNeu-Klassifizierung abgeschlossen:")
print(df['label'].value_counts())

# ── 2. Visualisierungen ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SCHRITT 2: Visualisierungen generieren")
print("=" * 65)

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# 1. Label-Verteilung
print("[1/7] Label-Verteilung...")
fig, ax = plt.subplots(figsize=(10, 6))
label_counts = df['label'].value_counts()
bars = ax.bar(label_counts.index, label_counts.values, color=['#388e3c', '#d32f2f'], alpha=0.85)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}\n({h/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontsize=10)
ax.set_ylabel('Anzahl Dokumente')
ax.set_title('Verteilung der Klassifikations-Labels (neues Modell)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / '1_label_distribution.png', dpi=DPI, bbox_inches='tight')
plt.close()

# 2. Confidence-Score-Verteilung
print("[2/7] Confidence-Scores...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].hist(df['score'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(df['score'].mean(), color='red', linestyle='--', label=f'Mittelwert: {df["score"].mean():.3f}')
axes[0].axvline(df['score'].median(), color='green', linestyle='--', label=f'Median: {df["score"].median():.3f}')
axes[0].set_xlabel('Confidence Score'); axes[0].set_ylabel('Häufigkeit')
axes[0].set_title('Confidence Score Verteilung', fontweight='bold'); axes[0].legend(); axes[0].grid(alpha=0.3)
for label, color in [('NON_HATE', 'steelblue'), ('HATE', 'crimson')]:
    axes[1].hist(df[df['label']==label]['score'], bins=30, alpha=0.6, label=label, color=color, edgecolor='black')
axes[1].set_xlabel('Confidence Score'); axes[1].set_ylabel('Häufigkeit')
axes[1].set_title('Confidence Score nach Label', fontweight='bold'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / '2_confidence_scores.png', dpi=DPI, bbox_inches='tight')
plt.close()

# 3. Boxplot
print("[3/7] Boxplot...")
fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column='score', by='label', ax=ax, patch_artist=True)
ax.set_xlabel('Label'); ax.set_ylabel('Confidence Score')
ax.set_title('Confidence Score nach Label', fontweight='bold'); plt.suptitle(''); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / '3_score_boxplot.png', dpi=DPI, bbox_inches='tight')
plt.close()

# 4. Confidence-Level-Verteilung
print("[4/7] Confidence-Level...")
df['confidence_level'] = pd.cut(df['score'], bins=[0, 0.6, 0.8, 1.0],
                                labels=['Niedrig (<0.6)', 'Mittel (0.6-0.8)', 'Hoch (>0.8)'])
fig, ax = plt.subplots(figsize=(10, 6))
conf_counts = df['confidence_level'].value_counts().sort_index()
bars = ax.bar(conf_counts.index, conf_counts.values, color=['#d32f2f', '#ff9800', '#388e3c'], alpha=0.85)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}\n({h/len(df)*100:.1f}%)', ha='center', va='bottom')
ax.set_ylabel('Anzahl Dokumente'); ax.set_title('Dokumente nach Confidence-Level', fontweight='bold')
ax.grid(axis='y', alpha=0.3); plt.tight_layout()
plt.savefig(VIZ_DIR / '4_confidence_levels.png', dpi=DPI, bbox_inches='tight')
plt.close()

# 5. Heatmap
print("[5/7] Heatmap...")
fig, ax = plt.subplots(figsize=(10, 5))
pivot = df.groupby(['label', 'confidence_level']).size().unstack(fill_value=0)
sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
ax.set_title('Vorhersagen nach Label und Confidence-Level', fontweight='bold')
plt.tight_layout()
plt.savefig(VIZ_DIR / '5_confidence_heatmap.png', dpi=DPI, bbox_inches='tight')
plt.close()

# 6. Textlänge
print("[6/7] Textlängen-Analyse...")
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax_row, col, label_str in [(axes[0], 'text_length', 'Zeichen'), (axes[1], 'word_count', 'Wörter')]:
    ax_row[0].hist(df[col], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax_row[0].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mittelwert: {df[col].mean():.0f}')
    ax_row[0].set_xlabel(f'Textlänge ({label_str})'); ax_row[0].set_ylabel('Häufigkeit')
    ax_row[0].set_title(f'Textlänge ({label_str})', fontweight='bold'); ax_row[0].legend(); ax_row[0].grid(alpha=0.3)
    df.boxplot(column=col, by='label', ax=ax_row[1], patch_artist=True)
    ax_row[1].set_xlabel('Label'); ax_row[1].set_ylabel(f'Textlänge ({label_str})')
    ax_row[1].set_title(f'Textlänge ({label_str}) nach Label', fontweight='bold'); ax_row[1].get_figure().suptitle('')
plt.tight_layout()
plt.savefig(VIZ_DIR / '6_text_length_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()

# 7. Zusammenfassung Tabelle
print("[7/7] Summary-Tabelle...")
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')
rows = []
for lbl in sorted(df['label'].unique()):
    sub = df[df['label'] == lbl]
    rows.append([lbl, len(sub), f"{len(sub)/len(df)*100:.1f}%",
                 f"{sub['score'].mean():.3f}", f"{sub['score'].median():.3f}"])
rows.append(['GESAMT', len(df), '100.0%', f"{df['score'].mean():.3f}", f"{df['score'].median():.3f}"])
cols = ['Label', 'Anzahl', 'Anteil', 'Ø Score', 'Median Score']
tbl = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.5)
for i in range(len(cols)):
    tbl[(0, i)].set_facecolor('#4472C4'); tbl[(0, i)].set_text_props(weight='bold', color='white')
    tbl[(len(rows), i)].set_facecolor('#FFC000'); tbl[(len(rows), i)].set_text_props(weight='bold')
ax.set_title('Klassifikations-Zusammenfassung (neues Modell)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(VIZ_DIR / '8_summary_statistics.png', dpi=DPI, bbox_inches='tight')
plt.close()

# CSVs aktualisieren
df[df['score'] < 0.6][['doc_id','text','label','score']].sort_values('score').to_csv(
    VIZ_DIR / 'low_confidence_predictions.csv', index=False)
pd.DataFrame(rows, columns=cols).to_csv(VIZ_DIR / 'summary_statistics.csv', index=False)

print("\n[OK] Alle Visualisierungen gespeichert in:", VIZ_DIR)
print(f"  HATE: {(df['label']=='HATE').sum()} | NON_HATE: {(df['label']=='NON_HATE').sum()}")
