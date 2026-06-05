"""
Regelbasierter Redebeitrag-Segmentierer für Hessische Landtagsprotokolle
=========================================================================
Erkennt Sprechermarker (z.B. "Robert Lambrou (AfD):") in den TXT-Seiten
des HPG-Exports und teilt die Protokolle in einzelne Redebeiträge auf.

Ausgabe: Data/prep_v1/speech_segments.parquet
  Spalten: speaker, party, text, wahlperiode, sitzung, datum, source

Verwendung:
  python src/segment_speeches.py
  python src/segment_speeches.py --out Data/prep_v1/speech_segments.parquet
  python src/segment_speeches.py --preview 5   # zeigt erste N Segmente
"""
import argparse
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Stdout auf UTF-8 stellen (verhindert UnicodeEncodeError auf Windows)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
DATA_DIR = BASE_DIR / "Data" / "HPG-Export_HES-20_2019-2024"
OUT_PATH = BASE_DIR / "Data" / "prep_v1" / "speech_segments.parquet"

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
SESSION_HDR = re.compile(
    r'Hessischer Landtag[^\n]*?(\d+)\.\s*Wahlperiode[^\n]*?(\d+)\.\s*Sitzung[^\n]*?'
    r'(\d{1,2})\.\s*(Januar|Februar|März|April|Mai|Juni|Juli|August|'
    r'September|Oktober|November|Dezember)\s*(\d{4})',
    re.DOTALL,
)

PARTY_SUFFIX = re.compile(r'\s*\(([^)]+)\)\s*$')

MONTHS = {
    'Januar': 1, 'Februar': 2, 'März': 3, 'April': 4,
    'Mai': 5, 'Juni': 6, 'Juli': 7, 'August': 8,
    'September': 9, 'Oktober': 10, 'November': 11, 'Dezember': 12,
}

MIN_WORDS = 10  # kürzere Beiträge ("Jawohl.") werden verworfen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_header(text: str):
    m = SESSION_HDR.search(text)  # gesamten Text durchsuchen
    if not m:
        return None, None, None
    wp, sitzung, day, month_str, year = m.groups()
    try:
        dt = date(int(year), MONTHS[month_str], int(day))
    except ValueError:
        dt = None
    return int(wp), int(sitzung), dt


# Bekannte Phrasen, die fälschlich als Sprechermarker erkannt werden
_FALSE_SPEAKER_PREFIXES = re.compile(
    r'^(Wir |Ich |Sie |Der |Die |Das |Es |Im |Zum |Zur |Bei |Nach |Dann |'
    r'Damit |Dazu |Hier |Nun |Also |Bitte |Danke |Leider |Deshalb )',
    re.IGNORECASE,
)

# Rollenbezeichnungen, die allein als Sprechermarker gelten
_ROLE_PREFIXES = re.compile(
    r'^(Präsident(?:in)?|Vizepräsident(?:in)?|Alterspräsident(?:in)?|'
    r'Minister(?:in)?|Staatssekretär(?:in)?|Staatssekretär)\b'
)


def is_speaker_line(line: str, next_line: str) -> bool:
    """Erkennt Zeilen wie 'Robert Lambrou (AfD):' oder 'Präsident Boris Rhein:'."""
    s = line.strip()
    if not s.endswith(':'):
        return False
    if s.startswith('('):
        return False
    name = s[:-1].strip()
    if not (3 <= len(name) <= 80):
        return False
        # Keine Satzanfänge / Strukturformeln
    if _FALSE_SPEAKER_PREFIXES.match(name):
        return False
    # Kein Verb am Anfang (Zitateinleitungen wie "Burke betonte bereits Folgendes")
    words = name.split()
    if len(words) >= 3 and re.search(r'[a-zäöüß]{3,}te$|[a-zäöüß]{3,}te\b', words[-1]):
        return False
    # Muss Rollenbezeichnung ODER mindestens zwei Großbuchstaben-Wörter enthalten
    has_role = bool(_ROLE_PREFIXES.match(name))
    cap_words = re.findall(r'[A-ZÄÖÜ][a-zäöüß]{1,}', name)
    has_name = len(cap_words) >= 2
    if not (has_role or has_name):
        return False
    # Sprechermarker sind immer von einer Leerzeile gefolgt
    if next_line.strip() != '':
        return False
    return True


def parse_speaker(line: str):
    s = line.strip().rstrip(':').strip()
    m = PARTY_SUFFIX.search(s)
    if m:
        return s[:m.start()].strip(), m.group(1).strip()
    return s, None


def clean_speech(text: str) -> str:
    # Seitenköpfe aus Folgeseiten entfernen
    text = SESSION_HDR.sub('', text)
    # Mehrfache Leerzeilen reduzieren
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Segmentierung eines Protokolls (alle Seiten zusammen)
# ---------------------------------------------------------------------------
def segment_protocol(txt_files: list[Path]) -> list[dict]:
    full = '\n'.join(
        f.read_text(encoding='utf-8', errors='replace')
        for f in sorted(txt_files)
    )

    wahlperiode, sitzung, datum = parse_header(full)
    source = txt_files[0].parent.parent.name

    lines = full.split('\n')
    segments = []
    current_speaker = current_party = None
    current_lines: list[str] = []

    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else ''

        if is_speaker_line(line, next_line):
            if current_speaker and current_lines:
                speech = clean_speech('\n'.join(current_lines))
                if len(speech.split()) >= MIN_WORDS:
                    segments.append({
                        'speaker':     current_speaker,
                        'party':       current_party,
                        'text':        speech,
                        'wahlperiode': wahlperiode,
                        'sitzung':     sitzung,
                        'datum':       datum,
                        'source':      source,
                    })
            current_speaker, current_party = parse_speaker(line)
            current_lines = []
        elif current_speaker is not None:
            current_lines.append(line)

    # letztes Segment
    if current_speaker and current_lines:
        speech = clean_speech('\n'.join(current_lines))
        if len(speech.split()) >= MIN_WORDS:
            segments.append({
                'speaker':     current_speaker,
                'party':       current_party,
                'text':        speech,
                'wahlperiode': wahlperiode,
                'sitzung':     sitzung,
                'datum':       datum,
                'source':      source,
            })

    return segments


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(OUT_PATH))
    parser.add_argument('--preview', type=int, default=0,
                        help='Erste N Segmente ausgeben')
    args, _ = parser.parse_known_args()

    print('=' * 65)
    print('Regelbasierter Protokoll-Segmentierer')
    print('=' * 65)

    protocol_dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())
    print(f'Protokollverzeichnisse: {len(protocol_dirs)}')

    all_segments: list[dict] = []
    for pdir in protocol_dirs:
        txt_dir = pdir / 'txt'
        if not txt_dir.exists():
            continue
        files = list(txt_dir.glob('*.txt'))
        if not files:
            continue
        segs = segment_protocol(files)
        all_segments.extend(segs)
        print(f'  {pdir.name}: {len(segs)} Segmente')

    df = pd.DataFrame(all_segments)

    # Parteien normalisieren (OCR-/Tippfehler aus Protokollen)
    party_map = {
        'DIELINKE': 'DIE LINKE',
        'Freie Demokaten': 'Freie Demokraten',
        'BÜNDNIS90/DIE GRÜNEN': 'BÜNDNIS 90/DIE GRÜNEN',
    }
    df['party'] = df['party'].replace(party_map)

    print(f'\nGesamt: {len(df)} Redebeiträge')
    print(f'Sprecher: {df["speaker"].nunique()}')
    parties = df['party'].dropna().unique()
    print(f'Parteien: {sorted(parties)}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f'\nGespeichert: {out}')

    print('\nTop-15 Redner nach Anzahl Beiträge:')
    print(df['speaker'].value_counts().head(15).to_string())

    print('\nBeiträge pro Partei:')
    print(df['party'].value_counts().head(10).to_string())

    if args.preview:
        print(f'\n--- Vorschau: erste {args.preview} Segmente ---')
        for _, row in df.head(args.preview).iterrows():
            print(f"\n[{row['speaker']} ({row['party']})] "
                  f"Sitzung {row['sitzung']}, {row['datum']}")
            print(row['text'][:300])
            print('...')


if __name__ == '__main__':
    main()
