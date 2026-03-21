"""
Fake News Detection System - Training Pipeline
Author: Your Name
Description: Trains and evaluates a fake news classifier using TF-IDF + ML models.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from preprocess import preprocess_text


# ─── Configuration ────────────────────────────────────────────────────────────

DATA_PATH = "data/news.csv"          # Path to your dataset
MODEL_DIR = "models/"                # Where to save trained models
TEST_SIZE = 0.2                         # 20% held out for testing
RANDOM_SEED = 42
MAX_FEATURES = 50_000                   # TF-IDF vocabulary size
NGRAM_RANGE = (1, 2)                    # Unigrams + bigrams


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Load and merge fake/real news CSV files.

    Supports two formats:
      1. Single file with 'text' and 'label' columns
      2. Separate Fake.csv / True.csv files (ISOT format)
    """
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)

    # Standardise column names (handle common variations)
    df.columns = [c.lower().strip() for c in df.columns]

    # Combine title + text if both exist (gives model more signal)
    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["content"] = df["text"].fillna("")
    else:
        raise ValueError("Dataset must have a 'text' column.")

    # Normalise labels → 0 = REAL, 1 = FAKE
    if "label" in df.columns:
        label_map = {"real": 0, "true": 0, "fake": 1, "false": 1, 0: 0, 1: 1}
        df["label"] = df["label"].str.lower().map(label_map)
    else:
        raise ValueError("Dataset must have a 'label' column.")

    df = df.dropna(subset=["content", "label"])
    print(f"[INFO] Loaded {len(df):,} articles | "
          f"Real: {(df.label==0).sum():,} | Fake: {(df.label==1).sum():,}")
    return df


# ─── Feature Extraction ───────────────────────────────────────────────────────

def build_tfidf(train_texts: pd.Series):
    """
    Fit a TF-IDF vectoriser on training texts.

    TF-IDF (Term Frequency–Inverse Document Frequency) scores each word by
    how often it appears in THIS document vs. how rare it is across ALL docs.
    This helps distinguish signal words from common filler words.
    """
    vectoriser = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,    # Captures "not true", "breaking news" etc.
        sublinear_tf=True,          # Replace raw TF with log(1+TF) — smooths extremes
        min_df=3,                   # Ignore terms appearing in fewer than 3 docs
        max_df=0.95,                # Ignore terms appearing in >95% of docs (too common)
        strip_accents="unicode",
    )
    return vectoriser.fit(train_texts)


# ─── Model Training ───────────────────────────────────────────────────────────

def train_models(X_train, y_train) -> dict:
    """
    Train multiple classifiers and return them in a dict.
    Logistic Regression is recommended as primary — it is fast, interpretable,
    and performs excellently on TF-IDF features.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,                  # Regularisation strength (lower = stronger)
            solver="lbfgs",
            class_weight="balanced",  # Handles class imbalance automatically
            random_state=RANDOM_SEED,
        ),
        "Naive Bayes": MultinomialNB(
            alpha=0.1,              # Laplace smoothing
        ),
    }

    trained = {}
    for name, model in models.items():
        print(f"[TRAIN] Fitting {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"        ✓ Done")

    return trained


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, model_name: str):
    """
    Print a full classification report and return key metrics.
    """
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  Results for: {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}  (of predicted FAKE, how many truly are?)")
    print(f"  Recall    : {recall:.4f}  (of all FAKE articles, how many caught?)")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Real','Fake'])}")

    return {"accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1}


# ─── Save Artefacts ───────────────────────────────────────────────────────────

def save_model(obj, filename: str):
    """Persist a Python object to disk with pickle."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[SAVE] Saved → {path}")


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Preprocess text (clean, tokenise, lemmatise)
    print("\n[PREPROCESS] Cleaning text — this may take a minute...")
    df["clean_text"] = df["content"].apply(preprocess_text)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label"],       # Keeps class ratio the same in both splits
    )
    print(f"\n[SPLIT] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 4. Feature extraction
    print("\n[FEATURES] Fitting TF-IDF vectoriser...")
    vectoriser = build_tfidf(X_train)
    X_train_vec = vectoriser.transform(X_train)
    X_test_vec  = vectoriser.transform(X_test)
    print(f"           Vocabulary size: {len(vectoriser.vocabulary_):,} terms")

    # 5. Train models
    print("\n[TRAIN] Training classifiers...")
    models = train_models(X_train_vec, y_train)

    # 6. Evaluate all models
    results = {}
    for name, model in models.items():
        results[name] = evaluate(model, X_test_vec, y_test, name)

    # 7. Pick the best model by F1-score
    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = models[best_name]
    print(f"\n[BEST] Best model: {best_name}  (F1 = {results[best_name]['f1']:.4f})")

    # 8. Save artefacts
    save_model(vectoriser, "tfidf_vectoriser.pkl")
    save_model(best_model, "best_model.pkl")
    save_model({"name": best_name, "metrics": results[best_name]}, "metadata.pkl")

    print("\n[DONE] Training complete. Run the Streamlit app: streamlit run app/app.py")


if __name__ == "__main__":
    main()
