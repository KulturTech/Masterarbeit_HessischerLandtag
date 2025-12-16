# Install dependencies from the terminal (not inside this Python file):
# pip install -r requirements.txt
import pathlib
from tqdm import tqdm
import pandas as pd

base = pathlib.Path("/home/gseraria/Masterarbeit/BERT_HessicherLandtag/Data/HPG-Export_HES-20_2019-2024")
rows = []
for pdf_dir in base.glob("**/047461152_20_2019-24_pdf_*"):
    txt_dir = pdf_dir / "txt"
    if not txt_dir.exists():
        continue
    for txt_file in txt_dir.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8", errors="replace")
        doc_id = f"{pdf_dir.name}_{txt_file.stem}"
        rows.append({"doc_id": doc_id, "text": text, "source_path": str(txt_file)})
df = pd.DataFrame(rows)
print(df.shape)

#Bereinigung
import re
import unicodedata

def normalize_text(text: str) -> str:
    # Unicode Normalisierung
    text = unicodedata.normalize("NFKC", text)
    # Entfernen häufige Seiten-/Header-Fußzeilen (heuristisch)
    text = re.sub(r"^\s*Seite\s*\d+\s*$", " ", text, flags=re.MULTILINE)
    # Entferne multiple Leerzeilen
    text = re.sub(r"\n{2,}", "\n\n", text)
    # Entferne viele Whitespaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Standardisiere Leerzeichen um Satzzeichen
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Optional: alle Zeilen zusammenführen (wenn sinnvoll)
    text = text.replace("\r\n", "\n").strip()
    return text


if __name__ == "__main__":
    # Apply normalization
    print("Starte Bereinigung der Texte...")
    df["text"] = df["text"].fillna("")
    df["clean_text"] = df["text"].apply(normalize_text)

    # Entferne komplett leere Einträge nach Bereinigung
    before = len(df)
    df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)
    after = len(df)
    print(f"Dokumente insgesamt: {before}, nach Bereinigung: {after}")

    # Optional: entferne Duplikate (identische bereinigte Texte)
    dup_before = len(df)
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    dup_after = len(df)
    print(f"Duplikate entfernt: {dup_before - dup_after}")

    # Erstelle Ausgabepfad
    out_dir = pathlib.Path("data/prep_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "all_docs_clean.csv"
    parquet_path = out_dir / "all_docs_clean.parquet"

    # Speichern
    df_to_save = df[["doc_id", "clean_text", "source_path"]].rename(columns={"clean_text": "text"})
    df_to_save.to_csv(csv_path, index=False)
    try:
        df_to_save.to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"Parquet write failed: {e}")

    print(f"Bereinigte Daten gespeichert: {csv_path} ({len(df_to_save)} rows)")