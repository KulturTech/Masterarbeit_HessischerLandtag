"""
Hate-Dokumente bzgl. Migranten: Redner und Parteien identifizieren
Verfeinerte Version mit Name→Partei-Lookup aus allen Protokollen
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from collections import Counter

BASE_DIR = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
VIZ_DIR  = BASE_DIR / "Data" / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)
DPI = 300

PARTY_COLORS = {
    'CDU':   '#333333',
    'SPD':   '#E3000F',
    'GRÜNE': '#64A12D',
    'FDP':   '#FFED00',
    'LINKE': '#BE3075',
    'AfD':   '#009EE0',
}

PARTY_PATTERNS = {
    'CDU':   r'\bCDU\b',
    'SPD':   r'\bSPD\b',
    'GRÜNE': r'\bGR[ÜU]NE[N]?\b|\bB[üu]ndnis\s*90\b',
    'FDP':   r'\bFDP\b|\bFreie\s+Demokraten\b',
    'LINKE': r'\bLINKE[N]?\b|\bDie\s+Linke\b',
    'AfD':   r'\bAfD\b',
}

MIGRATION_KEYWORDS = [
    r'\bFlüchtling', r'\bAsylbewerber', r'\bMigrant', r'\bAusländer',
    r'\bZuwanderer', r'\bIntegration\b', r'\bAbschiebung', r'\bGeflüchtete',
    r'\bEinwanderer', r'\bGestattete', r'\bDuldung', r'\billegal\s+Eingereiste',
]
MIGRATION_RE = re.compile('|'.join(MIGRATION_KEYWORDS), re.IGNORECASE)

SKIP_WORDS = {
    'Präsident', 'Vizepräsident', 'Ministerpräsident', 'Staatssekretär',
    'Minister', 'Im', 'Auf', 'Abwesende', 'Ausgegeben', 'Hessischer',
    'Plenarprotokoll', 'Wahlperiode', 'Sitzung', 'Wiesbaden'
}

def extract_party_from_text(ctx: str) -> str | None:
    for party, pat in PARTY_PATTERNS.items():
        if re.search(pat, ctx, re.IGNORECASE):
            return party
    return None


def build_name_party_lookup(all_texts: list[str]) -> dict[str, str]:
    """
    Baut ein Name→Partei-Lexikon aus allen Protokolltexten.
    Nutzt das Format: "Abg. Name (Partei)" und "Name (Partei):"
    """
    # Pattern 1: "Abg. Vorname Nachname (Partei)"
    abg_re = re.compile(
        r'Abg\.\s+([A-ZÄÖÜ][a-zäöüß\-\.]+(?:\s+(?:Dr\.|h\.c\.|Prof\.)?\s*[A-ZÄÖÜ][a-zäöüß\-]+){1,4})'
        r'\s*\(([^)]{2,60})\)',
        re.MULTILINE
    )
    # Pattern 2: "Name (Partei):" am Redebeginn
    speech_re = re.compile(
        r'\n([A-ZÄÖÜ][a-zäöüß\-\.]+(?:\s+(?:Dr\.|h\.c\.|Prof\.)?\s*[A-ZÄÖÜ][a-zäöüß\-]+){1,4})'
        r'\s*\(([^)]{2,60})\)\s*:',
        re.MULTILINE
    )

    lookup: dict[str, Counter] = {}

    for text in all_texts:
        for pattern in [abg_re, speech_re]:
            for m in pattern.finditer(text):
                name = m.group(1).strip()
                ctx  = m.group(2).strip()
                # Präsidium / Minister herausfiltern
                if any(w in name for w in SKIP_WORDS):
                    continue
                party = extract_party_from_text(ctx)
                if party:
                    lookup.setdefault(name, Counter())[party] += 1

    # Beste Partei pro Name
    return {name: cnt.most_common(1)[0][0] for name, cnt in lookup.items()}


def extract_speakers_from_text(text: str) -> list[str]:
    """Redner aus Text extrahieren (Name am Zeilenanfang gefolgt von ':')."""
    speaker_re = re.compile(
        r'\n([A-ZÄÖÜ][a-zäöüß\-\.]+(?:\s+(?:Dr\.|h\.c\.|Prof\.)?\s*[A-ZÄÖÜ][a-zäöüß\-]+){1,4})'
        r'(?:\s*\([^)]{2,60}\))?\s*:',
        re.MULTILINE
    )
    speakers = []
    for m in speaker_re.finditer(text):
        name = m.group(1).strip()
        if any(w in name for w in SKIP_WORDS):
            continue
        if len(name) < 5:
            continue
        speakers.append(name)
    return speakers


# ── 1. Daten laden ─────────────────────────────────────────────────────────────
print("=" * 65)
print("Lade Daten...")
print("=" * 65)

df = pd.read_parquet(BASE_DIR / "Data" / "prep_v1" / "all_docs_classified.parquet")
print(f"Alle Dokumente: {len(df)}")

# Name→Partei-Lookup aus ALLEN Texten aufbauen
print("\nBaue Name→Partei-Lookup aus allen Protokollen...")
name_party = build_name_party_lookup(df['text'].tolist())
print(f"Bekannte Abgeordnete: {len(name_party)}")

# HATE + Migration filtern
hate_df = df[df['label'] == 'HATE'].copy()
hate_df['mentions_migration'] = hate_df['text'].str.contains(MIGRATION_RE)
migrant_hate = hate_df[hate_df['mentions_migration']].copy()
print(f"\nHATE-Dokumente gesamt: {len(hate_df)}")
print(f"HATE + Migrations-Bezug: {len(migrant_hate)}")

# ── 2. Redner & Parteien aus HATE+Migrations-Docs ─────────────────────────────
print("\nExtrahiere Redner...")

speaker_counter: Counter = Counter()
party_doc_counter: Counter = Counter()  # pro Dokument zählen (nicht pro Erwähnung)
speaker_party_map: dict[str, str] = {}

for _, row in migrant_hate.iterrows():
    speakers_in_doc = extract_speakers_from_text(row['text'])
    parties_in_doc = set()

    for name in speakers_in_doc:
        speaker_counter[name] += 1
        party = name_party.get(name)
        if party:
            speaker_party_map[name] = party
            parties_in_doc.add(party)

    for party in parties_in_doc:
        party_doc_counter[party] += 1

print(f"Unique Redner: {len(speaker_counter)}")
print(f"Davon Partei bekannt: {sum(1 for n in speaker_counter if n in speaker_party_map)}")
print(f"\nPartei-Verteilung (Dokumente mit mind. 1 Redner dieser Partei):")
for p, c in party_doc_counter.most_common():
    print(f"  {p}: {c}")

# ── 3. Visualisierungen ───────────────────────────────────────────────────────
sns.set_style("whitegrid")

# Plot 1: Parteien nach Dokument-Anzahl
print("\n[1/3] Parteien-Verteilung...")
fig, ax = plt.subplots(figsize=(10, 6))
parties_sorted = [(p, party_doc_counter[p]) for p in PARTY_COLORS if party_doc_counter[p] > 0]
parties_sorted.sort(key=lambda x: x[1], reverse=True)
if parties_sorted:
    p_names, p_counts = zip(*parties_sorted)
    bars = ax.bar(p_names, p_counts, color=[PARTY_COLORS[p] for p in p_names], alpha=0.85, edgecolor='black')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.2, str(int(h)),
                ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_xlabel('Partei')
ax.set_ylabel('Anzahl Dokumente')
ax.set_title('Redner welcher Partei sprechen in\nHATE-Dokumenten mit Migrations-Bezug?', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'hate_migration_by_party.png', dpi=DPI, bbox_inches='tight')
plt.close()

# Plot 2: Top-20 Redner (nur mit bekannter Partei)
print("[2/3] Top-Redner...")
top_speakers = [(n, c) for n, c in speaker_counter.most_common(30) if n in speaker_party_map][:20]
if top_speakers:
    fig, ax = plt.subplots(figsize=(12, 8))
    spk_names, spk_counts = zip(*top_speakers)
    spk_colors = [PARTY_COLORS.get(speaker_party_map[n], '#888888') for n in spk_names]
    bars = ax.barh(spk_names[::-1], spk_counts[::-1], color=spk_colors[::-1], alpha=0.85, edgecolor='black')
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.1, bar.get_y() + bar.get_height()/2., str(int(w)),
                va='center', fontsize=9)
    ax.set_xlabel('Anzahl Vorkommen in HATE+Migrations-Dokumenten')
    ax.set_title('Top-20 Redner in HATE-Dokumenten mit Migrations-Bezug\n(Farbe = Partei)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=p) for p, c in PARTY_COLORS.items() if party_doc_counter.get(p, 0) > 0]
    ax.legend(handles=legend, loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'hate_migration_top_speakers.png', dpi=DPI, bbox_inches='tight')
    plt.close()

# Plot 3: Partei-Anteile Pie Chart
print("[3/3] Pie Chart...")
fig, ax = plt.subplots(figsize=(8, 8))
pie_data = [(p, party_doc_counter[p]) for p in PARTY_COLORS if party_doc_counter[p] > 0]
pie_data.sort(key=lambda x: x[1], reverse=True)
p_labels, p_vals = zip(*pie_data)
p_colors = [PARTY_COLORS[p] for p in p_labels]
wedges, texts, autotexts = ax.pie(
    p_vals, labels=p_labels, colors=p_colors, autopct='%1.1f%%',
    startangle=90, pctdistance=0.8
)
for t in autotexts:
    t.set_fontsize(11); t.set_fontweight('bold')
    t.set_color('white' if t.get_text() else 'black')
ax.set_title('Parteien-Anteil in HATE+Migrations-Dokumenten\n(nach identifizierten Rednern)', fontweight='bold')
plt.tight_layout()
plt.savefig(VIZ_DIR / 'hate_migration_party_pie.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ── 4. CSV speichern ──────────────────────────────────────────────────────────
out = pd.DataFrame([
    {'name': n, 'party': speaker_party_map.get(n, 'unbekannt'), 'count': c}
    for n, c in speaker_counter.most_common(100)
])
out.to_csv(BASE_DIR / "Data" / "evaluation" / "hate_migration_speakers.csv", index=False)

pd.DataFrame([
    {'party': p, 'doc_count': party_doc_counter.get(p, 0)} for p in PARTY_COLORS
]).sort_values('doc_count', ascending=False).to_csv(
    BASE_DIR / "Data" / "evaluation" / "hate_migration_parties.csv", index=False)

print("\n[OK] Fertig. Visualisierungen:")
print("  - hate_migration_by_party.png")
print("  - hate_migration_top_speakers.png")
print("  - hate_migration_party_pie.png")
print(f"\nTop-15 Redner mit Partei:")
print(out[out['party'] != 'unbekannt'].head(15).to_string(index=False))
