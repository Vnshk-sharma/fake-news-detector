"""
Fake News Detection System - Explainability
Uses LIME (Local Interpretable Model-agnostic Explanations) to show users
WHICH words caused the model to label an article as Fake or Real.
"""

import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.dirname(__file__))
from preprocess import preprocess_text

MODEL_DIR = "../models/"


# ─── Load artefacts ───────────────────────────────────────────────────────────

def load_artefacts():
    """Load saved vectoriser + model from disk."""
    def _load(filename):
        with open(os.path.join(MODEL_DIR, filename), "rb") as f:
            return pickle.load(f)

    vectoriser = _load("tfidf_vectoriser.pkl")
    model      = _load("best_model.pkl")
    return vectoriser, model


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict(text: str, vectoriser, model) -> dict:
    """
    Run the full pipeline on raw text and return prediction details.

    Returns:
        {
          "label":      "FAKE" | "REAL",
          "confidence": float (0-1),
          "fake_prob":  float,
          "real_prob":  float,
          "clean_text": str,
        }
    """
    clean = preprocess_text(text)
    if not clean:
        return {"label": "UNKNOWN", "confidence": 0.0,
                "fake_prob": 0.5, "real_prob": 0.5, "clean_text": ""}

    vec  = vectoriser.transform([clean])
    prob = model.predict_proba(vec)[0]  # [P(real), P(fake)]

    label      = "FAKE" if prob[1] >= 0.5 else "REAL"
    confidence = float(max(prob))

    return {
        "label":      label,
        "confidence": confidence,
        "fake_prob":  float(prob[1]),
        "real_prob":  float(prob[0]),
        "clean_text": clean,
    }


# ─── LIME Explanation ─────────────────────────────────────────────────────────

def get_lime_explanation(text: str, vectoriser, model,
                         num_features: int = 10) -> list[tuple[str, float]]:
    """
    Generate a LIME explanation: a list of (word, weight) pairs.
    Positive weight → pushed toward FAKE.
    Negative weight → pushed toward REAL.

    LIME works by:
      1. Perturbing the input (randomly hiding words)
      2. Observing how predictions change
      3. Fitting a simple linear model on these perturbations
      4. Returning the linear weights as "importance"
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        print("[WARN] LIME not installed. Run: pip install lime")
        return []

    # LIME needs a pipeline function: raw text → class probabilities
    def predict_proba_pipeline(texts):
        cleaned = [preprocess_text(t) for t in texts]
        vecs = vectoriser.transform(cleaned)
        return model.predict_proba(vecs)

    explainer = LimeTextExplainer(
        class_names=["Real", "Fake"],
        random_state=42,
    )

    explanation = explainer.explain_instance(
        text,
        predict_proba_pipeline,
        num_features=num_features,
        num_samples=300,            # Fewer samples = faster but less stable
        labels=(1,),                # Explain the "Fake" class
    )

    # Returns [(word, weight)] where weight > 0 → evidence for FAKE
    return explanation.as_list(label=1)


# ─── Simple fallback: top TF-IDF words ───────────────────────────────────────

def get_top_tfidf_words(text: str, vectoriser, model,
                        n: int = 10) -> list[tuple[str, float]]:
    """
    Lightweight alternative to LIME.
    Multiplies the TF-IDF feature values by the model's learned coefficients
    to find which words contributed most to the fake-news score.

    Only works with linear models (Logistic Regression).
    """
    clean = preprocess_text(text)
    vec   = vectoriser.transform([clean])

    if not hasattr(model, "coef_"):
        return []  # Non-linear model — use LIME instead

    # coef_[0] is the coefficient for the positive class (FAKE = 1)
    coefs     = model.coef_[0]
    feature_names = vectoriser.get_feature_names_out()

    # Multiply TF-IDF weight × learned coefficient for each token
    nonzero_idx = vec.nonzero()[1]
    word_scores = [
        (feature_names[i], float(vec[0, i] * coefs[i]))
        for i in nonzero_idx
    ]

    # Sort by absolute contribution; return top n
    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return word_scores[:n]


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    vectoriser, model = load_artefacts()

    sample = (
        "Scientists confirm 5G towers are spreading coronavirus. "
        "The government is hiding the truth from citizens."
    )

    result = predict(sample, vectoriser, model)
    print(f"\nPrediction : {result['label']} ({result['confidence']:.1%} confident)")

    print("\nTop contributing words (TF-IDF method):")
    for word, score in get_top_tfidf_words(sample, vectoriser, model):
        direction = "→ FAKE" if score > 0 else "→ REAL"
        print(f"  {word:20s}  {score:+.4f}  {direction}")
