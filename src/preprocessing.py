"""
preprocessing.py — Modulo con funzioni di preprocessing testo per sentiment analysis.

Contiene le funzioni di pulizia, normalizzazione, feature engineering
e bilanciamento del dataset. Estratte e rifattorizzate dal codice originale
del progetto REACT.
"""

import re
import numpy as np
import pandas as pd
import emoji
import contractions
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.utils import resample

# Download risorse NLTK (idempotente)
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words("english"))


# =============================================================================
# Funzioni di pulizia testo
# =============================================================================

def preprocess_text(text: str) -> str:
    """
    Pipeline di pulizia testo per sentiment analysis.
    
    Steps:
    1. Lowercase
    2. Rimozione emoji
    3. Rimozione URL e tag HTML
    4. Rimozione punteggiatura (preserva apostrofi)
    5. Rimozione numeri
    6. Rimozione stopword
    7. Rimozione spazi multipli
    
    Args:
        text: Testo da pulire.
    
    Returns:
        Testo pulito e normalizzato.
    """
    # 1. Lowercase
    text = text.lower()
    
    # 2. Rimozione emoji
    text = emoji.replace_emoji(text, replace='')
    
    # 3. Rimozione URL e HTML
    text = re.sub(r"http\S+|www\S+|<.*?>", " ", text)
    
    # 4. Rimozione punteggiatura (preserva apostrofi)
    text = re.sub(r"[^\w\s']", " ", text)
    
    # 5. Rimozione numeri
    text = re.sub(r"\d+", " ", text)
    
    # 6. Rimozione stopword
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    
    # 7. Ricomponi e rimuovi spazi multipli
    text = " ".join(words)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def expand_contractions_text(text: str) -> str:
    """
    Espande le contrazioni inglesi (es. "don't" → "do not").
    
    Args:
        text: Testo con possibili contrazioni.
    
    Returns:
        Testo con contrazioni espanse.
    """
    return contractions.fix(text)


def mark_negations(text: str) -> str:
    """
    Marca le negazioni nel testo aggiungendo il prefisso 'NOT_' alle parole
    che seguono una negazione, fino alla punteggiatura successiva.
    
    Utile per modelli bag-of-words che non catturano il contesto.
    
    Args:
        text: Testo preprocessato.
    
    Returns:
        Testo con negazioni marcate.
    """
    negation_words = {"not", "no", "never", "n't"}
    words = text.split()
    result = []
    negating = False

    for word in words:
        if any(neg in word for neg in negation_words):
            negating = True
            result.append(word)
        elif negating:
            result.append("NOT_" + word)
            if any(punct in word for punct in [".", ",", "!", "?", ";"]):
                negating = False
        else:
            result.append(word)
    return " ".join(result)


def full_text_pipeline(text: str) -> str:
    """
    Applica la pipeline completa di preprocessing testo:
    preprocess → expand contractions → mark negations.
    
    Args:
        text: Testo raw.
    
    Returns:
        Testo completamente preprocessato.
    """
    text = preprocess_text(text)
    text = expand_contractions_text(text)
    text = mark_negations(text)
    return text


# =============================================================================
# Mapping e feature engineering
# =============================================================================

def map_rating_to_sentiment(rating: float) -> int:
    """
    Mappa il rating (1-5) a una classe di sentiment:
    - 0: Negativo (rating 1, 2)
    - 1: Neutro   (rating 3)
    - 2: Positivo  (rating 4, 5)
    
    Args:
        rating: Rating numerico 1-5.
    
    Returns:
        Classe sentiment (0, 1, 2).
    """
    if rating in [1, 2]:
        return 0
    elif rating == 3:
        return 1
    else:  # rating 4, 5
        return 2


SENTIMENT_LABELS = {0: "Negativo", 1: "Neutro", 2: "Positivo"}


def add_features(df: pd.DataFrame, text_col: str = "text_final") -> pd.DataFrame:
    """
    Aggiunge feature di sentiment engineering al DataFrame:
    - polarity: polarità del testo (TextBlob)
    - text_len: lunghezza del testo in parole
    
    Args:
        df: DataFrame con colonna di testo preprocessato.
        text_col: Nome della colonna di testo.
    
    Returns:
        DataFrame con colonne aggiuntive 'polarity' e 'text_len'.
    """
    df = df.copy()
    df["polarity"] = df[text_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df["text_len"] = df[text_col].apply(lambda x: len(str(x).split()))
    return df


# =============================================================================
# Filtri e bilanciamento
# =============================================================================

def filter_incoherent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove recensioni incoerenti (es. rating basso con sentiment positivo)
    e recensioni troppo corte (< 3 parole).
    
    Args:
        df: DataFrame con colonne 'rating', 'polarity', 'text_len'.
    
    Returns:
        DataFrame filtrato.
    """
    df = df.copy()
    
    # Calcolo soglie per rating 3 (IQR)
    q3 = df[df["rating"] == 3]["polarity"].quantile([0.25, 0.75])
    iqr3 = q3[0.75] - q3[0.25]
    lower3 = q3[0.25] - 1.5 * iqr3
    upper3 = q3[0.75] + 1.5 * iqr3

    def is_incoherent(row):
        r, p = row["rating"], row["polarity"]
        if r in [1, 2] and p >= 0:
            return True
        elif r == 3 and not (lower3 <= p <= upper3):
            return True
        elif r in [4, 5] and p <= 0:
            return True
        return False

    mask = df.apply(is_incoherent, axis=1)
    df = df[(~mask) & (df["text_len"] >= 3)]
    
    return df


def balance_dataset(df: pd.DataFrame, target_col: str = "sentiment_class", 
                    n_clusters: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Bilancia il dataset con undersampling basato su K-Means clustering.
    
    Campiona le classi maggioritarie per uguagliare la classe minoritaria,
    usando il clustering per preservare la diversità dei campioni.
    
    Args:
        df: DataFrame con colonna target.
        target_col: Nome della colonna target.
        n_clusters: Numero di cluster per il campionamento stratificato.
        random_state: Seed per la riproducibilità.
    
    Returns:
        DataFrame bilanciato.
    """
    df = df.copy()
    target_count = df[df[target_col] == 0].shape[0]  # classe minoritaria
    balanced_parts = [df[df[target_col] == 0].copy()]

    for cls in [1, 2]:
        subset = df[df[target_col] == cls].copy()
        
        if len(subset) <= target_count:
            balanced_parts.append(subset)
            continue
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        subset["cluster"] = kmeans.fit_predict(subset[["polarity", "text_len"]])
        samples_per_cluster = target_count // n_clusters
        sampled_clusters = []

        for c in range(n_clusters):
            cluster_df = subset[subset["cluster"] == c]
            sample = resample(
                cluster_df,
                replace=False,
                n_samples=min(samples_per_cluster, len(cluster_df)),
                random_state=random_state
            )
            sampled_clusters.append(sample)

        total_current = sum(len(s) for s in sampled_clusters)
        if total_current < target_count:
            remaining = subset[~subset.index.isin(pd.concat(sampled_clusters).index)]
            additional = resample(
                remaining,
                replace=False,
                n_samples=min(target_count - total_current, len(remaining)),
                random_state=random_state
            )
            sampled_clusters.append(additional)
        
        balanced_parts.append(pd.concat(sampled_clusters))

    result = pd.concat(balanced_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Rimuovi colonna cluster se presente
    if "cluster" in result.columns:
        result = result.drop(columns=["cluster"])
    
    return result
