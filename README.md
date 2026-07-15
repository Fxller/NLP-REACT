# 🧠 REACT — Review Evaluation and Analysis through Classification Techniques

> Progetto di Natural Language Processing: Sentiment Analysis su recensioni Amazon Beauty con confronto tra approcci classici e modelli transformer.

**Autori:** Luigi Guida, Francesco Perilli

---

## 📋 Indice

1. [Introduzione](#introduzione)
2. [Dominio Applicativo](#dominio-applicativo)
3. [Dataset](#dataset)
4. [Stato dell'Arte](#stato-dellarte)
5. [Metodologia](#metodologia)
6. [Metriche di Valutazione](#metriche-di-valutazione)
7. [Struttura del Progetto](#struttura-del-progetto)
8. [Come Riprodurre](#come-riprodurre)
9. [Risultati](#risultati)
10. [Limiti e Sviluppi Futuri](#limiti-e-sviluppi-futuri)

---

## Introduzione

La **sentiment analysis** è uno dei task più rilevanti nel Natural Language Processing, con applicazioni dirette nel mondo dell'e-commerce, del marketing e della customer experience. Questo progetto affronta il problema della classificazione automatica del sentiment nelle recensioni di prodotti di bellezza su Amazon, confrontando approcci classici basati su feature engineering (TF-IDF + classificatori ML) con un modello transformer pre-trained (DistilBERT) fine-tuned sul dominio specifico.

Come task secondario e complementare, il progetto include anche la **generazione automatica di recensioni** condizionata al rating, usando un modello T5 fine-tuned, per esplorare la dualità tra comprensione e produzione del linguaggio naturale nel dominio beauty.

**Task NLP principale:** Sentiment Analysis (classificazione a 3 classi: Negativo, Neutro, Positivo)  
**Task NLP secondario:** Conditional Text Generation

---

## Dominio Applicativo

Il dominio di applicazione è l'**e-commerce cosmetico e beauty**. Le recensioni dei prodotti di bellezza presentano caratteristiche linguistiche specifiche:

- **Terminologia di dominio** (ingredienti, texture, fragranze)
- **Sentiment spesso misto** (es. "profumo buono ma durata scarsa")
- **Forte sbilanciamento** verso le recensioni positive (rating 4-5)
- **Brevità dei testi** rispetto ad altri domini (elettronica, libri)

Queste specificità rendono la sentiment analysis nel dominio beauty una sfida interessante dal punto di vista NLP.

---

## Dataset

**Amazon Reviews 2023 — All Beauty**  
- **Fonte:** [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (HuggingFace)
- **Dimensione:** ~700.000 recensioni
- **Campi utilizzati:** `text` (corpo della recensione), `rating` (1-5 stelle)
- **Mapping classi:**
  - ⭐ Rating 1-2 → **Negativo** (classe 0)
  - ⭐⭐⭐ Rating 3 → **Neutro** (classe 1)
  - ⭐⭐⭐⭐ Rating 4-5 → **Positivo** (classe 2)

Il dataset viene scaricato automaticamente da HuggingFace alla prima esecuzione e salvato localmente per le successive.

---

## Stato dell'Arte

La sentiment analysis su recensioni e-commerce è stata ampiamente studiata in letteratura. Gli approcci principali includono:

| Approccio | Tecnica | Pro | Contro |
|-----------|---------|-----|--------|
| **Classico** | TF-IDF + SVM / Logistic Regression | Veloce, interpretabile, buone baseline | Non cattura contesto semantico |
| **Ensemble** | TF-IDF + XGBoost / Random Forest | Performance migliori delle baseline lineari | Richiede feature engineering |
| **Neural** | BERT / DistilBERT fine-tuned | Cattura contesto, transfer learning | Richiede GPU, meno interpretabile |
| **Domain-specific** | Modelli pre-trained su recensioni | Ottimizzato per il dominio | Meno generalizzabile |

Il nostro progetto segue un **approccio tradizionale**: avanzamento su più fasi della pipeline NLP (preprocessing dominio-specifico + confronto baseline/transformer + explainability).

---

## Metodologia

### Pipeline

```
Dataset Raw → Preprocessing → Feature Engineering → Modellazione → Valutazione → Error Analysis
```

### 1. Preprocessing
- Rimozione emoji, URL, HTML, numeri
- Lowercase e rimozione stopword
- Espansione contrazioni inglesi
- **Negation marking** (prefisso `NOT_` per bag-of-words)
- Filtro lingua (FastText) per rimuovere recensioni non inglesi
- Rimozione recensioni incoerenti (rating vs. sentiment) tramite analisi della polarità (TextBlob)

### 2. Bilanciamento
Il dataset originale è fortemente sbilanciato (predominanza di rating 4-5). Applichiamo un **undersampling stratificato basato su K-Means clustering** per preservare la diversità dei campioni nella classe maggioritaria.

### 3. Modelli

| Modello | Tipo | Input |
|---------|------|-------|
| **TF-IDF + SVM** | Baseline classica | Testo preprocessato + features (polarità, lunghezza) |
| **TF-IDF + XGBoost** | Baseline ensemble | Testo preprocessato + features (polarità, lunghezza) |
| **DistilBERT** | Transformer pre-trained | Testo originale (tokenizzato dal modello) |

### 4. Task secondario: Text Generation
Un modello **T5-small** fine-tuned genera recensioni condizionate al prodotto e al rating, esplorando la capacità generativa nel dominio beauty.

---

## Metriche di Valutazione

- **Accuracy**: percentuale di predizioni corrette
- **F1-score (macro)**: media armonica di precision e recall, media non pesata tra le classi — robusta per dataset sbilanciati
- **F1-score (weighted)**: media pesata per il numero di campioni per classe
- **Precision (macro)**: proporzione di veri positivi tra le predizioni positive
- **Recall (macro)**: proporzione di veri positivi tra i campioni reali positivi
- **Confusion Matrix**: per analisi per-classe degli errori

L'**explainability** è analizzata con **LIME** (Local Interpretable Model-agnostic Explanations) per comprendere quali feature/parole influenzano le predizioni.

---

## Struttura del Progetto

```
📦 REACT/
├── README.md                              # Questo file
├── requirements.txt                       # Dipendenze Python
├── .gitignore
│
├── data/                                  # Dataset (scaricato a runtime)
│   └── .gitkeep
│
├── notebooks/
│   ├── 01_data_exploration.ipynb          # EDA
│   ├── 02_preprocessing.ipynb             # Pipeline preprocessing
│   ├── 03_baseline_models.ipynb           # TF-IDF + SVM/XGBoost
│   ├── 04_transformer_finetuning.ipynb    # Fine-tuning DistilBERT
│   ├── 05_evaluation.ipynb                # Confronto finale
│   ├── 06_error_analysis.ipynb            # LIME + analisi errori
│   └── 07_text_generation.ipynb           # T5 review generation
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Download dataset da HuggingFace
│   ├── preprocessing.py                   # Funzioni di pulizia testo
│   └── evaluation.py                      # Metriche e visualizzazioni
│
└── results/
    └── figures/                           # Grafici generati
```

---

## Come Riprodurre

### Prerequisiti
- Python 3.10+
- (Consigliato) GPU per il fine-tuning di DistilBERT (es. Google Colab con T4)

### Setup

```bash
# 1. Clona la repository
git clone https://github.com/Fxller/REACT.git
cd REACT

# 2. Crea e attiva un virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure: venv\Scripts\activate  # Windows

# 3. Installa le dipendenze
pip install -r requirements.txt

# 4. Esegui i notebook in ordine
# Il dataset viene scaricato automaticamente da HuggingFace alla prima esecuzione
```

### Ordine di esecuzione dei notebook

1. `01_data_exploration.ipynb` — Analisi esplorativa
2. `02_preprocessing.ipynb` — Preprocessing (genera `data/amazon_beauty_clean.csv`)
3. `03_baseline_models.ipynb` — Training baseline
4. `04_transformer_finetuning.ipynb` — Fine-tuning DistilBERT (**richiede GPU**)
5. `05_evaluation.ipynb` — Confronto finale
6. `06_error_analysis.ipynb` — Analisi errori e LIME
7. `07_text_generation.ipynb` — (Opzionale) Generazione recensioni con T5

---

## Risultati

> **Nota:** I risultati verranno popolati dopo l'esecuzione completa dei notebook.

| Modello | Accuracy | F1 (macro) | F1 (weighted) | Precision (macro) | Recall (macro) |
|---------|----------|------------|----------------|-------------------|----------------|
| TF-IDF + SVM | — | — | — | — | — |
| TF-IDF + XGBoost | — | — | — | — | — |
| DistilBERT | — | — | — | — | — |

---

## Limiti e Sviluppi Futuri

### Limiti
- Dataset specifico per il dominio beauty: i risultati potrebbero non generalizzarsi.
- L'undersampling riduce la dimensione del training set.
- La classificazione a 3 classi (Neg/Neutro/Pos) perde informazione rispetto ai 5 livelli di rating.
- Non è stata esplorata la combinazione ensemble baseline + transformer.

### Sviluppi futuri
- Sperimentare con altri transformer (RoBERTa, ALBERT, modelli domain-specific).
- Data augmentation per mitigare lo sbilanciamento senza undersampling.
- Approccio ensemble: combinare le predizioni dei modelli classici e del transformer.
- Estendere l'analisi ad altri domini di recensioni Amazon.
- Multi-task learning: sentiment analysis + aspect extraction.

---

## Riferimenti

- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
- Sanh et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. NeurIPS Workshop.
- Hou et al. (2024). *Bridging Language and Items for Retrieval and Recommendation*. Amazon Reviews 2023 Dataset.
- Ribeiro et al. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD (LIME).
