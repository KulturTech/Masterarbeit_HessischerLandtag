"""
Kritische Diskursanalyse (KDA) der Hassrede-Segmente aus dem Hessischen Landtag
mittels LLama 3.1 via Ollama oder Groq API.

Verwendung:
  python kda_llm_analysis.py                        # bestätigte Hassrede (49 Segmente)
  python kda_llm_analysis.py --input fp             # False Positives (6 Segmente)
  python kda_llm_analysis.py --input segments       # alle HATE-Segmente aus baseline
  python kda_llm_analysis.py --input segments --party AfD --limit 30
  python kda_llm_analysis.py --input segments --limit 50 --min-score 0.9
  python kda_llm_analysis.py --test                 # nur 1 Segment (Test)
  python kda_llm_analysis.py --model llama3.1:70b
  python kda_llm_analysis.py --backend groq
  python kda_llm_analysis.py --meta                 # Metaanalyse aus gespeicherter CSV
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SEGMENTS = ROOT / "Data/prep_v1/speech_segments.parquet"

DATA_FILES = {
    "hate":     ROOT / "Data/evaluation/model_confirmed_hate_migration.csv",
    "fp":       ROOT / "Data/evaluation/false_positives_combined_model.csv",
    "segments": ROOT / "Data/evaluation/baseline_comparison.csv",
}
OUT_FILES = {
    "hate":     ROOT / "Data/evaluation/kda_analysis.csv",
    "fp":       ROOT / "Data/evaluation/kda_fp_analysis.csv",
    "segments": ROOT / "Data/evaluation/kda_segments_analysis.csv",
}
META_FILES = {
    "hate":     ROOT / "Data/evaluation/kda_meta_analysis.json",
    "fp":       ROOT / "Data/evaluation/kda_fp_meta_analysis.json",
    "segments": ROOT / "Data/evaluation/kda_segments_meta_analysis.json",
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
KDA_PROMPT = """\
Du bist ein Experte für Kritische Diskursanalyse (KDA) nach Ruth Wodak.
Analysiere den folgenden Ausschnitt aus einer Hessischen Landtagsdebatte (20. Wahlperiode, 2019–2024).

Strukturiere deine Analyse EXAKT nach diesen sechs Abschnitten (Nummerierung beibehalten):

1. DISKURSPOSITION: Welche politische/ideologische Haltung wird eingenommen?
2. ARGUMENTATIONSSTRATEGIE: Welche rhetorischen Strategien werden verwendet? (z.B. Verallgemeinerung, Naturalisierung, Kriminalisierung, Bedrohungsszenario, Kulturalisierung, Ökonomisierung)
3. SUBJEKTKONSTRUKTION: Wie werden Migranten/Einwanderer als Subjekte konstruiert und dargestellt?
4. SPRACHLICHE_MITTEL: Auffällige Lexik, Metaphern, Kollokationen, Fachbegriffe
5. MACHTVERHÄLTNISSE: Welche Machtrelationen werden (re-)produziert oder legitimiert?
6. DISKURSVERSCHRAENKUNG: Verbindungen zu anderen Diskurssträngen (Sicherheit, Kriminalität, Wirtschaft, Identität)

TEXT: {text}
SPRECHER: {speaker}
PARTEI: {party}
DATUM: {datum}

Antworte auf Deutsch in präziser akademischer Sprache. Halte jeden Abschnitt auf 2–4 Sätze.\
"""

KDA_FP_PROMPT = """\
Du bist ein Experte für Kritische Diskursanalyse (KDA) und automatische \
Hassrede-Erkennung.

Der folgende Text wurde von einem BERT-Modell fälschlicherweise als Hassrede \
gegen Migranten klassifiziert (False Positive). Das Modell-Konfidenz: {score:.2%}.
Das tatsächliche manuell annotierte Label ist NON_HATE.

Analysiere den Text nach diesen sechs Abschnitten (Nummerierung beibehalten):

1. DISKURSPOSITION: Welcher thematische/politische Kontext liegt vor?
2. MIGRATIONSBEZUG: Welche migrationsbezogenen Begriffe oder Themen kommen vor, \
die das Modell möglicherweise aktiviert haben?
3. SUBJEKTKONSTRUKTION: Werden Migranten/Einwanderer erwähnt und wenn ja, wie?
4. SPRACHLICHE_MITTEL: Auffällige Lexik, Fachbegriffe, Kollokationen
5. FALSCHKLASSIFIKATIONSGRUND: Warum könnte das Modell diesen Text als Hassrede \
eingestuft haben? Welche oberflächlichen Merkmale täuschen?
6. DISKURSIVER_BEFUND: Was zeigt dieser False Positive über Migrationsdiskurse \
im Landtag? Gibt es implizite Rahmungen, auch ohne explizite Hassrede?

TEXT: {text}
MODELL-KONFIDENZ (HATE): {score:.2%}

Antworte auf Deutsch in präziser akademischer Sprache. Halte jeden Abschnitt auf 2–4 Sätze.\
"""

META_PROMPT = """\
Du bist ein Experte für Kritische Diskursanalyse. Ich gebe dir {n} Einzelanalysen \
von Hassrede-Segmenten aus dem Hessischen Landtag (2019–2024).

Erstelle eine übergreifende Metaanalyse mit folgenden Abschnitten:

1. DOMINANTE_STRATEGIEN: Welche Argumentationsstrategien treten am häufigsten auf?
2. PARTEISPEZIFISCHE_MUSTER: Gibt es parteispezifische Diskursmuster?
3. SUBJEKTKONSTRUKTIONEN: Wiederkehrende Konstruktionen von Migranten/Einwanderern
4. DISKURSIVE_VERSCHIEBUNG: Erkennbare Veränderungen im Zeitverlauf (2019–2024)?
5. ZENTRALE_BEFUNDE: 3–5 zentrale Befunde für die Masterarbeit

EINZELANALYSEN (JSON):
{analyses}

Antworte auf Deutsch in akademischer Sprache.\
"""

META_FP_PROMPT = """\
Du bist ein Experte für Kritische Diskursanalyse und NLP-Fehleranalyse.
Ich gebe dir {n} Analysen von False Positives eines Hassrede-Detektors \
(Hessischer Landtag, 2019–2024).

Erstelle eine übergreifende Analyse mit folgenden Abschnitten:

1. HAUPTURSACHEN: Welche Muster erklären die Fehlklassifikationen am häufigsten?
2. DISKURSIVE_AMBIGUITAET: Welche Diskursmerkmale liegen an der Grenze \
zwischen Hassrede und legitimer politischer Sprache?
3. IMPLIZITE_RAHMUNGEN: Gibt es auch in diesen NON_HATE-Texten problematische \
Migrationskonstruktionen unterhalb der Hassrede-Schwelle?
4. MODELLGRENZEN: Was zeigen die False Positives über die Grenzen des BERT-Modells?
5. BEFUNDE_MASTERARBEIT: 3–5 relevante Befunde für die Masterarbeit

EINZELANALYSEN (JSON):
{analyses}

Antworte auf Deutsch in akademischer Sprache.\
"""


# ---------------------------------------------------------------------------
# Backend-Klassen
# ---------------------------------------------------------------------------
class OllamaBackend:
    def __init__(self, model: str):
        try:
            import ollama
            self._ollama = ollama
        except ImportError:
            sys.exit("Fehler: 'ollama' nicht installiert. Bitte: pip install ollama")
        self.model = model

    def chat(self, prompt: str) -> str:
        response = self._ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

# ---------------------------------------------------------------------------
# Parser für LLM-Antworten
# ---------------------------------------------------------------------------
SECTION_KEYS_HATE = [
    "diskursposition",
    "argumentationsstrategie",
    "subjektkonstruktion",
    "sprachliche_mittel",
    "machtverhältnisse",
    "diskursverschraenkung",
]

SECTION_KEYS_FP = [
    "diskursposition",
    "migrationsbezug",
    "subjektkonstruktion",
    "sprachliche_mittel",
    "falschklassifikationsgrund",
    "diskursiver_befund",
]

_NUM = r"(?:^|\n)\s*\*{0,2}\s*\d+\.\s*\*{0,2}\s*"
_SEP = r"\*{0,2}\s*[:\-]?\s*\*{0,2}\s*"
_END = r"(?=\n\s*\*{0,2}\s*\d+\.|$)"

SECTION_PATTERN_HATE = re.compile(
    _NUM
    + r"(DISKURSPOSITION|ARGUMENTATIONSSTRATEGIE|SUBJEKTKONSTRUKTION"
    + r"|SPRACHLICHE_MITTEL|MACHTVERHÄLTNISSE|DISKURSVERSCHRAENKUNG)"
    + _SEP + r"(.*?)" + _END,
    re.IGNORECASE | re.DOTALL,
)

SECTION_PATTERN_FP = re.compile(
    _NUM
    + r"(DISKURSPOSITION|MIGRATIONSBEZUG|SUBJEKTKONSTRUKTION"
    + r"|SPRACHLICHE_MITTEL|FALSCHKLASSIFIKATIONSGRUND|DISKURSIVER_BEFUND)"
    + _SEP + r"(.*?)" + _END,
    re.IGNORECASE | re.DOTALL,
)


def parse_response(raw: str, section_keys: list, pattern: re.Pattern) -> dict:
    result = {k: "" for k in section_keys}
    for match in pattern.finditer(raw):
        key = match.group(1).lower()
        for sk in section_keys:
            if sk in key or key in sk:
                result[sk] = match.group(2).strip()
                break
    parts = [v.split(".")[0] if v else "" for v in result.values()]
    result["kda_zusammenfassung"] = ". ".join(p for p in parts[:2] if p) + "."
    return result


# ---------------------------------------------------------------------------
# Einzelanalyse
# ---------------------------------------------------------------------------
def analyze_segment(row: pd.Series, backend, mode: str) -> dict:
    if mode == "fp":
        score = float(row.get("new_score", 0.0))
        prompt = KDA_FP_PROMPT.format(text=row["text"], score=score)
        raw = backend.chat(prompt)
        parsed = parse_response(raw, SECTION_KEYS_FP, SECTION_PATTERN_FP)
    else:
        def val(key):
            v = row.get(key)
            return v if pd.notna(v) else "unbekannt"

        prompt = KDA_PROMPT.format(
            text=row["text"],
            speaker=val("speaker"),
            party=val("party"),
            datum=val("datum"),
        )
        raw = backend.chat(prompt)
        parsed = parse_response(raw, SECTION_KEYS_HATE, SECTION_PATTERN_HATE)
    parsed["llm_raw"] = raw
    return parsed


# ---------------------------------------------------------------------------
# Metaanalyse
# ---------------------------------------------------------------------------
def run_meta_analysis(df: pd.DataFrame, backend, mode: str) -> dict:
    if mode == "fp":
        keys = SECTION_KEYS_FP
        meta_prompt_tpl = META_FP_PROMPT
    else:
        keys = SECTION_KEYS_HATE
        meta_prompt_tpl = META_PROMPT

    cols = [k for k in keys if k in df.columns]
    extra = [c for c in ["speaker", "party", "datum"] if c in df.columns]
    subset = df[cols + extra].fillna("").to_dict("records")
    analyses_json = json.dumps(subset[:30], ensure_ascii=False, indent=2)
    prompt = meta_prompt_tpl.format(n=len(subset), analyses=analyses_json)
    raw = backend.chat(prompt)
    return {"meta_raw": raw, "n_analyzed": len(df)}


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def load_data(mode: str, party: str | None = None,
              limit: int | None = None, min_score: float = 0.0) -> pd.DataFrame:
    path = DATA_FILES[mode]
    df = pd.read_csv(path)

    if mode == "segments":
        # baseline_comparison enthält bereits alle Metadaten aus speech_segments
        df = df.rename(columns={"ft_label": "label", "ft_score": "score"})
        df = df[df["label"] == "HATE"].copy()
        if party:
            df = df[df["party"].str.contains(party, case=False, na=False)]
        if min_score > 0.0:
            df = df[df["score"] >= min_score]
        df = df.sort_values("score", ascending=False)
        if limit:
            df = df.head(limit)
        print(f"  → {len(df)} Segmente nach Filterung (Partei={party or 'alle'}, "
              f"min_score={min_score}, limit={limit or 'kein'})")

    elif mode == "hate":
        if SEGMENTS.exists():
            df_seg = pd.read_parquet(SEGMENTS)[["text", "speaker", "party", "datum"]]
            df = df.merge(df_seg, on="text", how="left")
        else:
            print("Warnung: speech_segments.parquet nicht gefunden.")
            df["speaker"] = "unbekannt"
            df["party"] = "unbekannt"
            df["datum"] = "unbekannt"

    return df


def run(args):
    mode = args.input
    data_out = OUT_FILES[mode]
    meta_out = META_FILES[mode]
    section_keys = SECTION_KEYS_FP if mode == "fp" else SECTION_KEYS_HATE

    print(f"Modus: {'False Positives' if mode == 'fp' else 'Bestätigte Hassrede'} | "
          f"Backend: {args.backend.upper()} | Modell: {args.model}")

    if args.backend == "groq":
        backend = GroqBackend(model="llama-3.1-70b-versatile")
    else:
        backend = OllamaBackend(model=args.model)

    if args.meta:
        if not data_out.exists():
            sys.exit(f"Fehler: {data_out} nicht gefunden. Erst Einzelanalyse durchführen.")
        df = pd.read_csv(data_out)
        print("Starte Metaanalyse...")
        meta = run_meta_analysis(df, backend, mode)
        meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Metaanalyse gespeichert: {meta_out}")
        print("\n--- METAANALYSE ---\n")
        print(meta["meta_raw"])
        return

    df = load_data(
        mode,
        party=getattr(args, "party", None),
        limit=getattr(args, "limit", None),
        min_score=getattr(args, "min_score", 0.0),
    )

    if args.test:
        df = df.head(1)
        print("TESTMODUS: analysiere 1 Segment\n")

    print(f"Analysiere {len(df)} Segmente...")
    results = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        speaker = row.get("speaker")
        label = speaker if pd.notna(speaker) else row.get("label", "?")
        print(f"  [{i}/{len(df)}] {label}...", end=" ", flush=True)
        try:
            result = analyze_segment(row, backend, mode)
            results.append(result)
            print("OK")
        except Exception as e:
            print(f"FEHLER: {e}")
            results.append({k: "" for k in section_keys} | {"llm_raw": f"FEHLER: {e}", "kda_zusammenfassung": ""})

    df_results = pd.DataFrame(results)
    df_out = pd.concat([df.reset_index(drop=True), df_results], axis=1)
    df_out.to_csv(data_out, index=False, encoding="utf-8-sig")
    print(f"\nErgebnisse gespeichert: {data_out}")

    if args.test:
        print("\n--- TESTAUSGABE ---")
        for col in section_keys + ["kda_zusammenfassung"]:
            val = df_results.iloc[0].get(col, "")
            print(f"\n[{col.upper()}]\n{val}")


def main():
    parser = argparse.ArgumentParser(description="KDA-Analyse mit LLama 3.1")
    parser.add_argument("--input", default="hate", choices=["hate", "fp", "segments"],
                        help="Datenquelle: 'hate' (Standard), 'fp', 'segments'")
    parser.add_argument("--party", default=None,
                        help="Partei-Filter für --input segments (z.B. 'AfD', 'CDU')")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max. Anzahl Segmente (empfohlen: 30–100)")
    parser.add_argument("--min-score", type=float, default=0.0, dest="min_score",
                        help="Mindestkonfidenz für HATE (0.0–1.0), z.B. 0.9")
    parser.add_argument("--test", action="store_true", help="Nur 1 Segment analysieren")
    parser.add_argument("--meta", action="store_true", help="Metaanalyse aus gespeicherter CSV")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama-Modellname")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "groq"])
    args, _ = parser.parse_known_args()
    run(args)


if __name__ == "__main__":
    main()
