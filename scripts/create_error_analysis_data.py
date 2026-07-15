import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix

# Load dataset
df = pd.read_csv("data/amazon_beauty_clean.csv")
df = df.dropna(subset=["text_final"]).reset_index(drop=True)

X_text = df["text_final"]
X_numeric = df[["polarity", "text_len"]].values
y = df["sentiment_class"].values

# Split stratificato exactly as in notebook 03
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y
)

# Load saved vectorizer and scaler
vectorizer = joblib.load("models/vectorizer.pkl")
scaler = joblib.load("models/scaler.pkl")

# Transform test data
X_test_text_tfidf = vectorizer.transform(X_text.iloc[test_idx])
X_test_num_scaled = scaler.transform(X_numeric[test_idx])
X_test_combined = hstack([X_test_text_tfidf, csr_matrix(X_test_num_scaled)])

# Save test split files
joblib.dump(X_text.iloc[test_idx], "models/texts_test.pkl")
joblib.dump(X_test_combined, "models/X_test.pkl")
joblib.dump(df["sentiment_class"].iloc[test_idx], "models/y_test.pkl")

print("Successfully created test data files:")
print(f" - models/texts_test.pkl: {len(test_idx)} rows")
print(f" - models/X_test.pkl: {X_test_combined.shape}")
print(f" - models/y_test.pkl: {len(test_idx)} rows")
