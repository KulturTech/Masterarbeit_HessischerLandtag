"""
Lexikon-basierte Suche nach abwertender Migrationssprache
in echten Redebeiträgen des Hessischen Landtags
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
VIZ_DIR  = BASE_DIR / "Data" / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)
OUT_DIR  = BASE_DIR / "Data" / "evaluation"
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

SKIP_WORDS = {
    'Präsident', 'Vizepräsident', 'Ministerpräsident', 'Staatssekretär',
    'Minister', 'Im', 'Auf', 'Abwesende', 'Ausgegeben', 'Hessischer',
    'Plenarprotokoll', 'Wahlperiode', 'Sitzung', 'Wiesbaden'
}

# ── Hate-Lexikon ────────────────────────────────────────────────────────────
# Begriffe die auf abwertende / fremdenfeindliche Rhetorik hinweisen
HATE_LEXICON = {
    'Überfremdung':       [r'\bÜberfremdung\b', r'\büberfremd'],
    'Invasion/Flut':      [r'\bMassenmigration\b', r'\bMigrationswelle\b',
                           r'\bFlut\s+(?:von\s+)?(?:Migranten|Flüchtlingen|Ausländern)\b',
                           r'\bAnsturm\s+(?:von\s+)?(?:Migranten|Flüchtlingen|Ausländern|Asylbewerbern)\b',
                           r'\b(?:Migranten|Flüchtlinge|Ausländer)\s+(?:überfluten|überschwemmen|strömen)\b',
                           r'\bInvasion\s+(?:von\s+)?(?:Migranten|Flüchtlingen|Ausländern|Muslimen)\b',
                           r'\b(?:Migranten|Flüchtlings)(?:invasion|welle|flut|strom|ansturm)\b'],
    'Kriminalität':       [r'\bAusländerkriminalität\b', r'\bMigrantenkriminalität\b',
                           r'\bkriminelle\s+(?:Ausländer|Migranten|Asylbewerber|Flüchtlinge)\b',
                           r'\b(?:Ausländer|Migranten|Asylbewerber)\s+(?:sind\s+)?kriminell\b'],
    'Asylmissbrauch':     [r'\bAsylmissbrauch\b', r'\bAsylbetrug\b', r'\bScheinasylant',
                           r'\bWirtschaftsflüchtling', r'\bAsylant\b'],
    'Islamisierung':      [r'\bIslamisierung\b', r'\bIslamisier',
                           r'\bParallelgesellschaft\b', r'\bParallelgesellschaften\b'],
    'Leitkultur':         [r'\bLeitkultur\b', r'\bkulturelle\s+Überfremdung\b'],
    'Illegale Einreise':  [r'\billegal\s+(?:Eingereiste|Migranten|Einwanderer)\b',
                           r'\bunerlaubt\s+eingereist\b', r'\bIllegaleinreise\b'],
    'Bedrohungsrhetorik': [r'\bGefahr\s+(?:durch|von)\s+(?:Migranten|Ausländer|Flüchtlinge|Asylbewerber)\b',
                           r'\b(?:Migranten|Ausländer|Flüchtlinge)\s+bedrohen\b',
                           r'\bÜbernahme\b.*?(?:Islam|Migranten|Ausländer)\b',
                           r'\bAbendland\b', r'\bVolksaustausch\b', r'\bRemigration\b'],
}

# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def extract_party_from_text(ctx: str) -> str | None:
    for party, pat in PARTY_PATTERNS.items():
        if re.search(pat, ctx, re.IGNORECASE):
            return party
    return None


def build_name_party_lookup(all_texts: list[str]) -> dict[str, str]:
    abg_re = re.compile(
        r'Abg\.\s+([A-ZÄÖÜ][a-zäöüß\-\.]+(?:\s+(?:Dr\.|h\.c\.|Prof\.)?\s*[A-ZÄÖÜ][a-zäöüß\-]+){1,4})'
        r'\s*\(([^)]{2,60})\)',
        re.MULTILINE
    )
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
                if any(w in name for w in SKIP_WORDS):
                    continue
                party = extract_party_from_text(ctx)
                if party:
                    lookup.setdefault(name, Counter())[party] += 1
    return {name: cnt.most_common(1)[0][0] for name, cnt in lookup.items()}


TOC_RE    = re.compile(r'[A-ZÄÖÜ][a-zäöüß]+(?:\s+\S+){0,4}\.{5,}\d', re.MULTILINE)
YEAR_RE   = re.compile(r'\b(20\d{2})\b')
SPEAKER_RE = re.compile(
    r'\n([A-ZÄÖÜ][a-zäöüß\-\.]+(?:\s+(?:Dr\.|h\.c\.|Prof\.)?\s*[A-ZÄÖÜ][a-zäöüß\-]+){1,4})'
    r'(?:\s*\([^)]{2,60}\))?\s*:',
    re.MULTILINE
)

NAME_CLEAN_RE = re.compile(r'^[\s\W]+')  # führende Nicht-Buchstaben entfernen

def clean_speaker_name(name: str) -> str:
    """Artefakte wie 'Wort.\n\n' oder 'Bitte.\n\n' vor dem echten Namen entfernen."""
    # Letztes Wort nach Zeilenumbruch nehmen, falls mehrere Zeilen
    parts = [p.strip() for p in name.split('\n') if p.strip()]
    if parts:
        name = parts[-1]
    # Führende Satzzeichen / Kleinbuchstaben entfernen
    name = NAME_CLEAN_RE.sub('', name).strip()
    return name

def get_current_speaker(text: str, match_start: int) -> str | None:
    """Letzter Redner vor der gefundenen Position."""
    best_name, best_pos = None, -1
    for m in SPEAKER_RE.finditer(text):
        if m.start() <= match_start:
            name = clean_speaker_name(m.group(1))
            if any(w in name for w in SKIP_WORDS) or len(name) < 5:
                continue
            if m.start() > best_pos:
                best_pos = m.start()
                best_name = name
    return best_name


# Muster für kritisch-zitierende Verwendung (120 Zeichen vor dem Treffer)
CRITIQUE_RE = re.compile(
    r'(?:'
    r'[Ss]ie\s+(?:reden|sprechen|sagen|schreiben|behaupten|nennen\s+das|bezeichnen)\s+(?:\w+\s+){0,4}(?:von|als)\b'
    r'|[Ss]ogenannte[nrms]?\b'
    r'|[Ss]o\s+genannte[nrms]?\b'
    r'|[Ii]n\s+(?:Anführungszeichen|Anführungsstrichen)\b'
    r'|[Ff]abuliere[nt]\b'
    r'|[Dd]iffamier\w+'
    r'|[Hh]etze[nt]\b'
    r'|[Kk]ritisier\w+'
    r'|[Aa]ls\s+(?:\w+\s+){0,3}bezeichnet\b'
    r'|[Uu]nterstellung\b'
    r'|[Vv]orwurf\b'
    r'|[Zz]itat\b'
    r'|„[^"]{0,80}$'          # öffnendes deutsches Anführungszeichen kurz davor
    r'|"[^"]{0,80}$'
    r')'
)

def is_critical_use(context_before: str) -> bool:
    """Prüft ob der Begriff kritisch/zitierend verwendet wird."""
    return bool(CRITIQUE_RE.search(context_before))


# ── 1. Daten laden ───────────────────────────────────────────────────────────
print("=" * 65)
print("Lade Daten...")
print("=" * 65)
df = pd.read_parquet(BASE_DIR / "Data" / "prep_v1" / "all_docs_classified.parquet")
print(f"Dokumente gesamt: {len(df)}")

print("\nBaue Name→Partei-Lookup...")
name_party = build_name_party_lookup(df['text'].tolist())
print(f"Bekannte Abgeordnete: {len(name_party)}")

# Echte Rededokumente (kein TOC)
df['is_toc'] = df['text'].apply(lambda t: len(TOC_RE.findall(t)) >= 3)
speech_df = df[~df['is_toc']].copy()
print(f"\nEchte Rededokumente: {len(speech_df)} (TOC gefiltert: {df['is_toc'].sum()})")

# ── 2. Lexikon-Suche ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Lexikon-Suche in Rededokumenten...")
print("=" * 65)

# Kompiliere alle Pattern
compiled_lexicon = {
    cat: re.compile('|'.join(pats), re.IGNORECASE)
    for cat, pats in HATE_LEXICON.items()
}

hits = []  # Liste: {doc_idx, category, speaker, party, context}

for idx, row in speech_df.iterrows():
    text = row['text']
    for cat, pattern in compiled_lexicon.items():
        for m in pattern.finditer(text):
            speaker = get_current_speaker(text, m.start())
            party   = name_party.get(speaker) if speaker else None
            ctx_before = text[max(0, m.start() - 120): m.start()]
            critical   = is_critical_use(ctx_before)
            start = max(0, m.start() - 100)
            end   = min(len(text), m.end() + 150)
            # Jahr aus erstem Vorkommen im Text extrahieren
            year_m = YEAR_RE.search(text[:500])
            year   = int(year_m.group(1)) if year_m else None
            hits.append({
                'category': cat,
                'term':     m.group(0),
                'speaker':  speaker or 'unbekannt',
                'party':    party or 'unbekannt',
                'critical': critical,
                'year':     year,
                'context':  text[start:end].replace('\n', ' ').strip(),
                'score':    row['score'],
                'label':    row['label'],
            })

hits_df = pd.DataFrame(hits)
print(f"\nTreffer gesamt: {len(hits_df)}")
if len(hits_df) == 0:
    print("Keine Treffer gefunden.")
    exit()

critical_n = hits_df['critical'].sum()
print(f"  davon kritisch/zitierend: {critical_n}")
print(f"  davon Eigenverwending:    {len(hits_df) - critical_n}")

# Für Analyse: nur Eigenverwendung
own_df = hits_df[~hits_df['critical']].copy()
print(f"\nNach Filterung (nur Eigenverwendung): {len(own_df)} Treffer")

print("\nTreffer pro Kategorie (Eigenverwendung):")
for cat, n in own_df['category'].value_counts().items():
    print(f"  {cat}: {n}")

print("\nTreffer pro Partei (bekannt, Eigenverwendung):")
known = own_df[own_df['party'] != 'unbekannt']
for party, n in known['party'].value_counts().items():
    print(f"  {party}: {n}")

# ── 3. Visualisierungen ──────────────────────────────────────────────────────
sns.set_style("whitegrid")

# Plot 1: Treffer pro Kategorie nach Partei (Heatmap)
print("\n[1/3] Heatmap Kategorie × Partei...")
pivot = known.groupby(['party', 'category']).size().unstack(fill_value=0)
pivot = pivot.reindex([p for p in PARTY_COLORS if p in pivot.index])

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax, linewidths=0.5)
ax.set_title('Lexikon-Treffer nach Kategorie und Partei\n(Eigenverwendung, kein Zitat/Kritik, kein TOC)', fontweight='bold')
ax.set_xlabel('Kategorie')
ax.set_ylabel('Partei')
plt.tight_layout()
plt.savefig(VIZ_DIR / 'lexikon_heatmap_partei_kategorie.png', dpi=DPI, bbox_inches='tight')
plt.close()

# Plot 2: Balkendiagramm Treffer pro Partei (mit Stacked: Eigen vs. Kritik)
print("[2/3] Treffer pro Partei (Eigen vs. Kritik)...")
all_known = hits_df[hits_df['party'] != 'unbekannt']
stacked = all_known.groupby(['party', 'critical']).size().unstack(fill_value=0)
stacked.columns = ['Eigenverwendung', 'Kritisch/Zitat']
stacked = stacked.reindex([p for p in PARTY_COLORS if p in stacked.index])

fig, ax = plt.subplots(figsize=(10, 6))
bottom = [0] * len(stacked)
for col, color in [('Eigenverwendung', '#d32f2f'), ('Kritisch/Zitat', '#90caf9')]:
    if col in stacked.columns:
        vals = stacked[col].values
        bars = ax.bar(stacked.index, vals, bottom=bottom,
                      color=color, alpha=0.85, edgecolor='black', label=col)
        bottom = [b + v for b, v in zip(bottom, vals)]
ax.set_xlabel('Partei')
ax.set_ylabel('Anzahl Lexikon-Treffer')
ax.set_title('Abwertende Migrationssprache nach Partei\n(rot = Eigenverwendung, blau = Zitat/Kritik)', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'lexikon_treffer_partei.png', dpi=DPI, bbox_inches='tight')
plt.close()

# Plot 3: Top-20 Redner (nur Eigenverwendung)
print("[3/3] Top-Redner (Eigenverwendung)...")
speaker_counts = known.groupby(['speaker', 'party']).size().reset_index(name='count')
speaker_counts = speaker_counts.sort_values('count', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(12, 8))
colors = [PARTY_COLORS.get(p, '#888888') for p in speaker_counts['party']]
bars = ax.barh(speaker_counts['speaker'][::-1], speaker_counts['count'][::-1],
               color=colors[::-1], alpha=0.85, edgecolor='black')
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.1, bar.get_y() + bar.get_height()/2., str(int(w)), va='center', fontsize=9)
ax.set_xlabel('Anzahl Lexikon-Treffer')
ax.set_title('Top-20 Redner: Abwertende Migrationssprache\n(Farbe = Partei)', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
from matplotlib.patches import Patch
legend = [Patch(color=c, label=p) for p, c in PARTY_COLORS.items() if p in known['party'].values]
ax.legend(handles=legend, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'lexikon_top_redner.png', dpi=DPI, bbox_inches='tight')
plt.close()

# Plot 4: Jährliche Entwicklung nach Partei
print("[4/4] Jährliche Entwicklung...")
yearly = known.dropna(subset=['year']).copy()
yearly['year'] = yearly['year'].astype(int)

years = sorted(yearly['year'].unique())
parties_present = [p for p in PARTY_COLORS if p in yearly['party'].unique()]

# Absolute Treffer pro Jahr & Partei
pivot_year = yearly.groupby(['year', 'party']).size().unstack(fill_value=0)
pivot_year = pivot_year.reindex(columns=parties_present, fill_value=0)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Links: gestapeltes Balkendiagramm
bottom = [0] * len(pivot_year)
for party in parties_present:
    vals = pivot_year[party].values
    axes[0].bar(pivot_year.index, vals, bottom=bottom,
                color=PARTY_COLORS[party], alpha=0.85, edgecolor='black', label=party)
    bottom = [b + v for b, v in zip(bottom, vals)]
axes[0].set_xlabel('Jahr')
axes[0].set_ylabel('Anzahl Lexikon-Treffer')
axes[0].set_title('Abwertende Migrationssprache pro Jahr\n(gestapelt nach Partei)', fontweight='bold')
axes[0].legend(loc='upper right', fontsize=9)
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_xticks(years)

# Rechts: Liniendiagramm pro Partei
for party in parties_present:
    vals = pivot_year[party]
    axes[1].plot(vals.index, vals.values, marker='o', color=PARTY_COLORS[party],
                 label=party, linewidth=2, markersize=6)
axes[1].set_xlabel('Jahr')
axes[1].set_ylabel('Anzahl Lexikon-Treffer')
axes[1].set_title('Entwicklung je Partei über die Zeit', fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)
axes[1].set_xticks(years)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'lexikon_jahresverlauf.png', dpi=DPI, bbox_inches='tight')
plt.close()

print(f"\nJährliche Treffer (Eigenverwendung, bekannte Partei):")
print(pivot_year.to_string())

# ── 4. CSV + Beispieltexte ───────────────────────────────────────────────────
hits_df.to_csv(OUT_DIR / 'lexikon_treffer_alle.csv', index=False)
own_df.to_csv(OUT_DIR / 'lexikon_treffer_eigenverwendung.csv', index=False)
speaker_counts.to_csv(OUT_DIR / 'lexikon_top_redner.csv', index=False)

print("\n[OK] Fertig.")
print(f"  Visualisierungen in: {VIZ_DIR}")
print(f"  CSVs in: {OUT_DIR}")
print(f"\nTop-15 Redner (Eigenverwendung, bekannte Partei):")
print(speaker_counts.head(15).to_string(index=False))

print("\n\nBeispiel-Treffer pro Kategorie (Eigenverwendung):")
for cat in own_df['category'].unique():
    sub = own_df[own_df['category'] == cat]
    print(f"\n--- {cat} ({len(sub)} Treffer) ---")
    example = sub[sub['party'] != 'unbekannt'].head(1)
    if not example.empty:
        r = example.iloc[0]
        print(f"  Redner: {r['speaker']} ({r['party']})")
        print(f"  Begriff: '{r['term']}'")
        print(f"  Kontext: ...{r['context']}...")
