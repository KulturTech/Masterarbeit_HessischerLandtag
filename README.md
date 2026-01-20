# Masterarbeit: Hessischer Landtag

Dieses Repository enthält alle Datensätze und Code-Implementierungen für meine Masterarbeit zur Analyse von Hate Speech bzgl. Menschen mit Migrationshintegrund von Daten des Hessischen Landtags mittels BERT-Modellen.

## 📋 Übersicht

Das Projekt befasst sich mit der Anwendung von Natural Language Processing (NLP) Techniken, insbesondere BERT (Bidirectional Encoder Representations from Transformers), zur Analyse parlamentarischer Daten des Hessischen Landtags.

## 🗂️ Projektstruktur

```
Masterarbeit_HessischerLandtag/
├── BERT_HessischerLandtag/    # BERT-Modell-Implementierung und Trainingsscripts
├── requirements.txt            # Python-Abhängigkeiten
└── README.md                   # Projektdokumentation
```

## 🚀 Installation

### Voraussetzungen

- Python 3.8 oder höher
- pip (Python Package Installer)

### Setup

1. Repository klonen:
```bash
git clone https://github.com/KulturTech/Masterarbeit_HessischerLandtag.git
cd Masterarbeit_HessischerLandtag
```

2. Virtuelle Umgebung erstellen (empfohlen):
```bash
python -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate
```

3. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

## 💻 Verwendung

### BERT-Modell

Die BERT-Implementierung befindet sich im Ordner `BERT_HessischerLandtag/`. Detaillierte Anweisungen zur Verwendung der einzelnen Komponenten folgen.

```bash
cd BERT_HessischerLandtag
# Weitere Anweisungen folgen je nach Implementierung
```

## 📊 Datensätze

Die verwendeten Datensätze stammen aus öffentlich zugänglichen Quellen des Hessischen Landtags und umfassen:

- Parlamentarische Protokolle
- Anträge und Beschlüsse
- Redebeiträge von Abgeordneten

*Hinweis: Bitte beachten Sie eventuelle Lizenzbedingungen der verwendeten Daten.*

## 🔬 Methodik

Das Projekt nutzt BERT für folgende Aufgaben:

- Textklassifikation
- Named Entity Recognition (NER)
- Sentiment-Analyse
- Themenmodellierung

## 📝 Abhängigkeiten

Die vollständige Liste der benötigten Python-Pakete finden Sie in der `requirements.txt`. 

Hauptabhängigkeiten umfassen voraussichtlich:
- transformers (Hugging Face)
- torch / tensorflow
- pandas
- numpy
- scikit-learn

## 🤝 Beiträge

Da dies ein Masterarbeits-Projekt ist, werden derzeit keine externen Beiträge akzeptiert.

## 📄 Lizenz

Dieses Projekt wurde im Rahmen einer Masterarbeit erstellt. Die Nutzungsbedingungen werden zu einem späteren Zeitpunkt festgelegt.

## 👤 Autor

**KulturTech**

- GitHub: [@KulturTech](https://github.com/KulturTech)

## 📧 Kontakt

Bei Fragen zur Masterarbeit oder zum Code können Sie gerne ein Issue erstellen.

## 🙏 Danksagungen

- Hessischer Landtag für die Bereitstellung öffentlich zugänglicher Daten
- Hugging Face für die BERT-Implementierung
- Betreuende Professor:innen und Mentor:innen

## 📚 Weitere Ressourcen

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Hessischer Landtag](https://hessischer-landtag.de/)

---

*Letzte Aktualisierung: Januar 2026*
