"""
data_loader.py — Modulo per il download e caricamento del dataset Amazon Reviews 2023 (All Beauty).

Utilizza la libreria HuggingFace `datasets` per scaricare il dataset direttamente,
senza necessità di DVC o file CSV pre-scaricati nel repository.
"""

import os
import pandas as pd
from datasets import load_dataset


def load_amazon_beauty_reviews(cache_dir: str = "data", force_download: bool = False) -> pd.DataFrame:
    """
    Scarica il dataset Amazon Reviews 2023 (All Beauty) da HuggingFace
    e lo restituisce come DataFrame pandas.
    
    Se il dataset è già stato scaricato e salvato localmente in `cache_dir`,
    lo ricarica dal file CSV per velocizzare le esecuzioni successive.
    
    Args:
        cache_dir: Directory dove salvare/caricare il CSV locale.
        force_download: Se True, forza il ri-download da HuggingFace.
    
    Returns:
        pd.DataFrame con le colonne del dataset Amazon Reviews.
    """
    csv_path = os.path.join(cache_dir, "amazon_beauty_reviews.csv")
    
    if os.path.exists(csv_path) and not force_download:
        print(f"📂 Caricamento dataset da cache locale: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("⬇️  Download dataset da HuggingFace (può richiedere alcuni minuti)...")
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_review_All_Beauty",
            split="full",
            trust_remote_code=True
        )
        df = dataset.to_pandas()
        
        # Salva localmente per le prossime esecuzioni
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"💾 Dataset salvato in: {csv_path}")
    
    print(f"✅ Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
    return df


def load_amazon_beauty_metadata(cache_dir: str = "data", force_download: bool = False) -> pd.DataFrame:
    """
    Scarica i metadati del dataset Amazon Reviews 2023 (All Beauty) da HuggingFace.
    
    Args:
        cache_dir: Directory dove salvare/caricare il CSV locale.
        force_download: Se True, forza il ri-download da HuggingFace.
    
    Returns:
        pd.DataFrame con i metadati dei prodotti.
    """
    csv_path = os.path.join(cache_dir, "amazon_beauty_metadati.csv")
    
    if os.path.exists(csv_path) and not force_download:
        print(f"📂 Caricamento metadati da cache locale: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("⬇️  Download metadati da HuggingFace (può richiedere alcuni minuti)...")
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_meta_All_Beauty",
            split="full",
            trust_remote_code=True
        )
        df = dataset.to_pandas()
        
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"💾 Metadati salvati in: {csv_path}")
    
    print(f"✅ Metadati caricati: {df.shape[0]} righe, {df.shape[1]} colonne")
    return df
