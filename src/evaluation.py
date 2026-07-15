"""
evaluation.py — Modulo con funzioni di valutazione e visualizzazione per sentiment analysis.

Contiene helper per metriche di classificazione, confusion matrix,
confronto tra modelli e visualizzazioni.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score, 
    precision_score, 
    recall_score,
    ConfusionMatrixDisplay
)


SENTIMENT_LABELS = ["Negativo", "Neutro", "Positivo"]


def print_classification_metrics(y_true, y_pred, model_name: str = "Modello"):
    """
    Stampa il report di classificazione completo e le metriche principali.
    
    Args:
        y_true: Etichette vere.
        y_pred: Etichette predette.
        model_name: Nome del modello per il titolo.
    """
    print(f"\n{'='*60}")
    print(f"📊 Risultati: {model_name}")
    print(f"{'='*60}")
    print(f"\nAccuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Macro:  {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1 Weight: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=SENTIMENT_LABELS)}")


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Modello", 
                          save_path: str = None, normalize: str = None):
    """
    Visualizza la confusion matrix come heatmap.
    
    Args:
        y_true: Etichette vere.
        y_pred: Etichette predette.
        model_name: Nome del modello per il titolo.
        save_path: Se specificato, salva la figura in questo percorso.
        normalize: Se 'true', normalizza la matrice (valori tra 0 e 1).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    
    fmt = ".2f" if normalize else "d"
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=SENTIMENT_LABELS,
                yticklabels=SENTIMENT_LABELS,
                ax=ax)
    
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight='bold')
    ax.set_ylabel("Vero", fontsize=12)
    ax.set_xlabel("Predetto", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Confusion matrix salvata in: {save_path}")
    
    plt.show()


def get_metrics_dict(y_true, y_pred, model_name: str = "Modello") -> dict:
    """
    Calcola le metriche di classificazione e le restituisce come dizionario.
    
    Args:
        y_true: Etichette vere.
        y_pred: Etichette predette.
        model_name: Nome del modello.
    
    Returns:
        Dict con accuracy, f1_macro, f1_weighted, precision_macro, recall_macro.
    """
    return {
        "Modello": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 (macro)": f1_score(y_true, y_pred, average='macro'),
        "F1 (weighted)": f1_score(y_true, y_pred, average='weighted'),
        "Precision (macro)": precision_score(y_true, y_pred, average='macro'),
        "Recall (macro)": recall_score(y_true, y_pred, average='macro'),
    }


def compare_models(results: list[dict], save_path: str = None) -> pd.DataFrame:
    """
    Confronta i risultati di più modelli in una tabella e un grafico a barre.
    
    Args:
        results: Lista di dizionari (output di get_metrics_dict).
        save_path: Se specificato, salva il grafico.
    
    Returns:
        DataFrame con il confronto dei modelli.
    """
    df = pd.DataFrame(results)
    df = df.set_index("Modello")
    
    # Tabella
    print("\n" + "="*80)
    print("📊 CONFRONTO MODELLI")
    print("="*80)
    print(df.round(4).to_string())
    
    # Grafico
    metrics_to_plot = ["Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)"]
    available_metrics = [m for m in metrics_to_plot if m in df.columns]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    df[available_metrics].plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_title("Confronto Modelli — Metriche di Classificazione", fontsize=14, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Aggiungi valori sulle barre
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Grafico salvato in: {save_path}")
    
    plt.show()
    
    return df


def plot_class_distribution(y, title: str = "Distribuzione Classi", save_path: str = None):
    """
    Visualizza la distribuzione delle classi di sentiment.
    
    Args:
        y: Array/Series con le etichette.
        title: Titolo del grafico.
        save_path: Se specificato, salva il grafico.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if isinstance(y, pd.Series):
        counts = y.value_counts().sort_index()
    else:
        unique, cnt = np.unique(y, return_counts=True)
        counts = pd.Series(cnt, index=unique)
    
    colors = ['#e74c3c', '#f39c12', '#27ae60']  # Rosso, Arancione, Verde
    bars = ax.bar([SENTIMENT_LABELS[i] for i in counts.index], counts.values, 
                  color=colors[:len(counts)], edgecolor='black', linewidth=0.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Conteggio", fontsize=12)
    
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
