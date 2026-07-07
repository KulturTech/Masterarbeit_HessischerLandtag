#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mine_hard_negatives.py
======================
Schritt-2-Werkzeug: zieht aus baseline_comparison.csv eine validierungs-
fertige Kandidatenliste fuer das Weitertrainieren des feingetunten Modells.

Erzeugt drei Ausgaben:

  1. hard_negatives_candidates.csv
     Mutmassliche False Positives (ft_label==HATE), heuristisch vorsortiert
     (Praesidium/Verfahrensregister zuerst), stratifiziert gesampelt und mit
     leerer Spalte `human_label` fuer die manuelle Validierung.

  2. positive_candidates.csv
     Kandidaten fuer In-Domain-Positivbeispiele: migrationsbezogene
     Redebeitraege von Abgeordneten (kein Praesidium), fuer Pre-Screening
     durch die CoT-Pipeline bzw. manuelle Annotation.

  3. corpus_cleaning_report.txt
     Segmentierungs-Probleme (Verzeichnis-/Strukturtext) zum Bereinigen.

Prinzipien (Zirkularitaets-Schutz):
  * NICHTS aus diesen Listen geht ungeprueft ins Training - `human_label`
    muss von Hand gefuellt werden (Spalte bleibt hier bewusst leer).
  * Die Validierungs-/Test-Stichprobe der Arbeit muss VOR dem Training
    abgetrennt werden; dieses Skript zieht dafuer optional ein Holdout.

Aufruf:
  python mine_hard_negatives.py --input baseline_comparison.csv --outdir out/
  python mine_hard_negatives.py --input ... --outdir out/ --n-negatives 400 \\
      --n-positives 300 --holdout 100 --seed 42
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PRESIDIUM_RE = re.compile(r"Präsident|Vizepräsident", re.I)
PROCEDURAL_RE = re.compile(
    r"das Wort|ans Rednerpult|Nächste[r]? Redner|erteile ich|Zwischenfrage|"
    r"um (etwas mehr )?Ruhe|Sitzung ist (eröffnet|geschlossen)|Tagesordnung",
    re.I,
)
MIGRATION_RE = re.compile(
    r"Migra|Flücht|Asyl|Ausländer|Abschieb|Zuwander|Einwander|Geflüchtet|"
    r"Islam|Muslim|Clan",
    re.I,
)
# Reiner Verzeichnis-/Strukturtext: Punktreihen-Leader (Inhaltsverzeichnis)
# oder kurze, satzlose Drucksachen-Fragmente. WICHTIG: eingebettete
# Seitenzahlen und Drucksachen-Verweise kommen auch in echten Reden vor
# (Vorverarbeitungsartefakt bzw. normale Zitierweise) und duerfen NICHT
# zum Ausschluss fuehren - sie werden separat als Bereinigungshinweis gezaehlt.
DOTLEADER_RE = re.compile(r"\.{6,}")
DRUCKS_RE = re.compile(r"Drucks\.\s*\d+/\d+")
EMBEDDED_PAGE_RE = re.compile(r"\n\s*\d{4}\s*\n")


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    df["len"] = df["text"].str.len()
    df["is_presidium"] = df["speaker"].astype(str).str.contains(PRESIDIUM_RE)
    df["is_procedural"] = df["text"].str.contains(PROCEDURAL_RE)
    has_sentence = df["text"].str.contains(r"[a-zäöüß]{3,}\s+[a-zäöüß]{3,}.*[.!?]")
    df["is_structure"] = df["text"].str.contains(DOTLEADER_RE) | (
        df["text"].str.contains(DRUCKS_RE) & (df["len"] < 600) & (~has_sentence)
    )
    df["has_page_artifact"] = df["text"].str.contains(EMBEDDED_PAGE_RE)
    df["has_migration"] = df["text"].str.contains(MIGRATION_RE)
    return df


def mine_negatives(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """FP-Kandidaten: ft==HATE, stratifiziert nach Register-Heuristik.

    Strata (in dieser Prioritaet):
      praesidium      - Sprecherrolle Praesidium
      prozedural      - Verfahrensformeln von Nicht-Praesidium
      abgeordnete     - uebrige Abgeordneten-/Ministertexte (interessanteste
                        Gruppe: hier koennten echte Positives versteckt sein!)
    """
    fp = df[(df["ft_label"] == "HATE") & (~df["is_structure"])].copy()

    def stratum(row) -> str:
        if row["is_presidium"]:
            return "praesidium"
        if row["is_procedural"]:
            return "prozedural"
        return "abgeordnete"

    fp["stratum"] = fp.apply(stratum, axis=1)

    # Alle Nicht-Praesidiums-Faelle mitnehmen (kleine, wichtige Gruppe),
    # Praesidium nur als Stichprobe.
    parts = []
    rest = n
    for name in ("abgeordnete", "prozedural"):
        grp = fp[fp["stratum"] == name]
        take = min(len(grp), max(rest // 3, 50))
        parts.append(grp.sample(take, random_state=seed) if len(grp) > take else grp)
        rest -= len(parts[-1])
    praes = fp[fp["stratum"] == "praesidium"]
    take = min(len(praes), max(rest, 0))
    if take:
        parts.append(praes.sample(take, random_state=seed))

    out = pd.concat(parts, ignore_index=True)
    out["human_label"] = ""          # von Hand fuellen: NON_HATE / HATE / UNKLAR
    out["notes"] = ""
    cols = ["stratum", "speaker", "party", "datum", "ft_score", "base_label",
            "text", "human_label", "notes"]
    return out[cols].sort_values(["stratum", "ft_score"], ascending=[True, False])


def mine_positives(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Kandidaten fuer In-Domain-Positivbeispiele.

    Quelle: migrationsbezogene Beitraege von Abgeordneten (kein Praesidium,
    kein Strukturtext). Priorisierung: kuerzere, pointierte Beitraege und
    Zwischenruf-Kontexte zuerst (dort ist manifest problematische Sprache
    am wahrscheinlichsten), dann laengere Redepassagen.
    """
    cand = df[
        df["has_migration"]
        & (~df["is_presidium"])
        & (~df["is_structure"])
        & df["party"].notna()
    ].copy()
    cand["priority"] = (cand["len"] < 400).astype(int) * 2 + \
                       cand["text"].str.contains(r"\(Zuruf|\(Beifall", regex=True).astype(int)
    cand = cand.sort_values(["priority", "len"], ascending=[False, True])
    out = cand.head(n).copy() if len(cand) > n else cand
    out["human_label"] = ""
    out["notes"] = ""
    cols = ["speaker", "party", "datum", "ft_label", "ft_score",
            "text", "human_label", "notes"]
    return out[cols]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-negatives", type=int, default=400)
    ap.add_argument("--n-positives", type=int, default=300)
    ap.add_argument("--holdout", type=int, default=0,
                    help="N Instanzen je Liste als Test-Holdout abtrennen "
                         "(gehen NIE ins Training)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load(Path(args.input))

    neg = mine_negatives(df, args.n_negatives, args.seed)
    pos = mine_positives(df, args.n_positives, args.seed)

    if args.holdout:
        for name, frame in (("negatives", neg), ("positives", pos)):
            hold = frame.sample(min(args.holdout, len(frame)),
                                random_state=args.seed)
            frame.drop(hold.index, inplace=True)
            hold.to_csv(outdir / f"holdout_{name}.csv", index=False)

    neg.to_csv(outdir / "hard_negatives_candidates.csv", index=False)
    pos.to_csv(outdir / "positive_candidates.csv", index=False)

    # Bereinigungsreport
    struct = df[df["is_structure"]]
    pages = df[df["has_page_artifact"] & (~df["is_structure"])]
    with (outdir / "corpus_cleaning_report.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"Segmente gesamt: {len(df)}\n\n")
        fh.write(f"[1] Reiner Struktur-/Verzeichnistext (ausschliessen): {len(struct)}\n")
        fh.write(f"    davon ft_label==HATE: {(struct['ft_label']=='HATE').sum()}\n")
        fh.write("    Beispiele (erste 10, je 120 Zeichen):\n")
        for _, r in struct.head(10).iterrows():
            fh.write(f"    - [{r['speaker']}] {r['text'][:120]!r}\n")
        fh.write(f"\n[2] Echte Reden mit eingebetteten Seitenzahl-Artefakten "
                 f"(Text bereinigen, NICHT ausschliessen): {len(pages)}\n")
        fh.write("    Empfehlung: Muster r'\\n\\s*\\d{4}\\s*\\n' aus den Texten\n"
                 "    entfernen, bevor tokenisiert wird - sonst lernen Modelle\n"
                 "    Seitenzahlen als Feature und Zitate werden verunreinigt.\n")

    print(f"Hard-Negative-Kandidaten : {len(neg):>5}  -> hard_negatives_candidates.csv")
    print(f"Positiv-Kandidaten       : {len(pos):>5}  -> positive_candidates.csv")
    print(f"Strukturtext-Segmente    : {len(struct):>5}  -> corpus_cleaning_report.txt")
    print("\nStrata der Negative:")
    print(neg["stratum"].value_counts().to_string())
    print("\nNaechste Schritte:")
    print(" 1. human_label in beiden CSVs von Hand fuellen (NON_HATE/HATE/UNKLAR)")
    print(" 2. Nur validierte Zeilen ins Weitertrainieren uebernehmen")
    print(" 3. holdout_*.csv NIEMALS ins Training - das ist euer Testset")


if __name__ == "__main__":
    main()