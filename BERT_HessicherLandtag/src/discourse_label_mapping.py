
"""
diagnose_label_mapping.py
=========================
Schritt-1-Diagnose fuer den Klassifikator-Artefakt in baseline_comparison.csv:
Prueft, ob die HATE/NON_HATE-Zuordnung ein Label-Mapping-/Export-Bug ist
oder ob das Modell das Muster tatsaechlich gelernt hat.

Vorgehen:
  1. laedt Modell + Tokenizer, druckt config.id2label / label2id
  2. jagt Kontrollinstanzen (eindeutig HATE vs. eindeutig NON_HATE, inkl.
     Praesidiumsfloskeln aus dem Korpus) durch das Modell
  3. druckt ROHE Logits + Softmax pro Klasse-Index, NICHT nur das Label
  4. Interpretationshilfe am Ende

Aufruf:
  python diagnose_label_mapping.py --model /pfad/zum/finetuned-checkpoint
  python diagnose_label_mapping.py --model /pfad/checkpoint --base deepset/dehatebert-mono-german

Erwartung, wenn KEIN Mapping-Bug vorliegt:
  - eindeutige Hetze -> hoher Logit auf dem Index, den id2label als HATE fuehrt
  - Praesidiumsfloskel -> ebenfalls hoher HATE-Logit (= echt gelerntes Artefakt)
Erwartung bei Mapping-Bug:
  - Hetze bekommt hohen Logit auf Index X, aber id2label[X] == "NON_HATE"
    (oder das Export-Skript hat die Indizes anders interpretiert als config)
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Kontrollinstanzen
# ---------------------------------------------------------------------------
# Gruppe A: eindeutig hetzerisch (synthetisch, GermEval-Stil - NUR Diagnose,
#           nicht ins Korpus/Training uebernehmen)
# Gruppe B: Praesidiumsfloskeln (woertlich aus baseline_comparison.csv)
# Gruppe C: neutrale lange Sachrede (Kontrolle)

CONTROLS: list[tuple[str, str]] = [
    ("A_hetze_1", "Diese Parasiten fluten unser Land und gehören alle sofort abgeschoben, jeder einzelne von denen."),
    ("A_hetze_2", "Ihr seid der letzte Dreck und habt in diesem Land nichts verloren, verschwindet endlich."),
    ("B_floskel_1", "Für die AfD-Fraktion bitte ich nun Herrn Schenk ans Rednerpult."),
    ("B_floskel_2", "Herr Kollege! Ich darf alle im Raum Anwesenden wieder um etwas mehr Ruhe bitten."),
    ("B_floskel_3", "Vielen Dank. Ich gebe Herrn Staatsminister Prof. Lorz zur Beantwortung das Wort."),
    ("C_sachrede", "Die bundesweite Verteilung der Asylsuchenden richtet sich grundsätzlich nach dem "
                   "Königsteiner Schlüssel. Daran wird weiterhin einvernehmlich festgehalten, um eine "
                   "solidarische Verteilung zwischen den Ländern sicherzustellen."),
]


def inspect(model_path: str, device: str) -> None:
    print(f"\n{'=' * 70}\nModell: {model_path}\n{'=' * 70}")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device).eval()

    print(f"num_labels : {model.config.num_labels}")
    print(f"id2label   : {model.config.id2label}")
    print(f"label2id   : {model.config.label2id}")
    if all(v.startswith("LABEL_") for v in model.config.id2label.values()):
        print(">>> WARNUNG: id2label sind generische Platzhalter (LABEL_0/LABEL_1).")
        print(">>> Das Export-Skript hat das Mapping dann selbst festgelegt -")
        print(">>> genau dort entsteht der klassische Vertauschungs-Bug.")

    header = f"\n{'instanz':<14} {'logits':<24} {'softmax':<24} argmax -> label"
    print(header)
    print("-" * len(header))
    with torch.no_grad():
        for name, text in CONTROLS:
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=512).to(device)
            logits = model(**enc).logits[0].float().cpu()
            probs = torch.softmax(logits, dim=-1)
            am = int(torch.argmax(logits))
            lab = model.config.id2label.get(am, f"LABEL_{am}")
            lg = "[" + ", ".join(f"{v:+.3f}" for v in logits.tolist()) + "]"
            pr = "[" + ", ".join(f"{v:.3f}" for v in probs.tolist()) + "]"
            print(f"{name:<14} {lg:<24} {pr:<24} {am} -> {lab}")

    print("""
Interpretation:
  * A-Instanzen landen auf HATE, B-Instanzen auf NON_HATE
      -> Mapping ok UND Modell ok; der Artefakt liegt dann im Export-/
         Inferenz-Skript der CSV (dort dasselbe Mapping pruefen!).
  * A-Instanzen landen auf NON_HATE, B-Instanzen auf HATE
      -> Modell hat das Register-Artefakt ECHT gelernt -> Schritt 2
         (Hard Negative Mining + Weitertrainieren).
  * A-Instanzen bekommen hohe Logits auf Index i, aber id2label[i] ist
    NON_HATE -> LABEL-MAPPING-BUG. Fix: config.json korrigieren bzw.
    Export-Skript auf config.id2label umstellen statt hartcodiertem Mapping.
""")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Pfad/HF-ID des feingetunten Checkpoints")
    ap.add_argument("--base", default=None, help="optional: Basismodell zum Vergleich")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    inspect(args.model, args.device)
    if args.base:
        inspect(args.base, args.device)


if __name__ == "__main__":
    main()