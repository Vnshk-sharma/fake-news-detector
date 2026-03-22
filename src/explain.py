"""
Fake News Detection System - Explainability (Improved)
- UNCERTAIN middle zone (not just FAKE/REAL)
- Writing style red flag detection
- Headline vs body disagreement check
- TF-IDF word importance
- LIME explainability (optional)
"""

import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.dirname(__file__))
from preprocess import preprocess_text

MODEL_DIR = "models/"


# ─── Load artefacts ───────────────────────────────────────────────────────────

def load_artefacts():
    def _load(filename):
        with open(os.path.join(MODEL_DIR, filename), "rb") as f:
            return pickle.load(f)
    vectoriser = _load("tfidf_vectoriser.pkl")
    model      = _load("best_model.pkl")
    return vectoriser, model


# ─── Prediction with 3-zone verdict ──────────────────────────────────────────

def predict(text: str, vectoriser, model) -> dict:
    """
    Run the full pipeline on raw text.

    Label logic (more honest than a hard 0.5 cutoff):
      fake_prob >= 0.75  →  FAKE      (strong signal)
      fake_prob <= 0.35  →  REAL      (strong signal)
      between            →  UNCERTAIN (model is not confident)
    """
    clean = preprocess_text(text)
    if not clean:
        return {
            "label": "UNKNOWN", "confidence": 0.0,
            "fake_prob": 0.5, "real_prob": 0.5, "clean_text": ""
        }

    vec       = vectoriser.transform([clean])
    prob      = model.predict_proba(vec)[0]
    fake_prob = float(prob[1])
    real_prob = float(prob[0])

    if fake_prob >= 0.75:
        label      = "FAKE"
        confidence = fake_prob
    elif fake_prob <= 0.35:
        label      = "REAL"
        confidence = real_prob
    else:
        label      = "UNCERTAIN"
        confidence = float(max(prob))

    return {
        "label":      label,
        "confidence": confidence,
        "fake_prob":  fake_prob,
        "real_prob":  real_prob,
        "clean_text": clean,
    }


# ─── Writing style analysis ───────────────────────────────────────────────────

def analyse_writing_style(text: str) -> dict:
    """
    Detect writing patterns common in fake news.
    Returns red flags and a suspicion score (0-100).
    """
    text_lower = text.lower()
    words      = text.split()
    sentences  = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]

    flags = {
        "exclamation_marks":    text.count("!"),
        "caps_words":           sum(1 for w in words if w.isupper() and len(w) > 2),
        "caps_ratio":           round(sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1), 3),
        "has_shocking":         "shocking"         in text_lower,
        "has_breaking":         "breaking"         in text_lower,
        "has_exposed":          "exposed"          in text_lower,
        "has_share_this":       "share this"       in text_lower,
        "has_before_deleted":   "before it"        in text_lower and "delet" in text_lower,
        "has_mainstream_media": "mainstream media" in text_lower,
        "has_whistleblower":    "whistleblower"    in text_lower,
        "has_big_pharma":       "big pharma"       in text_lower,
        "has_wake_up":          "wake up"          in text_lower,
        "has_truth":            "the truth"        in text_lower,
        "has_anonymous":        "anonymous"        in text_lower,
        "question_marks":       text.count("?"),
        "avg_sentence_length":  round(np.mean([len(s.split()) for s in sentences]) if sentences else 0, 1),
    }

    # Suspicion score out of 100
    score  = 0
    score += min(flags["exclamation_marks"] * 10, 25)
    score += min(flags["caps_words"] * 8,          20)
    score += 10 if flags["has_shocking"]          else 0
    score += 10 if flags["has_breaking"]          else 0
    score += 10 if flags["has_share_this"]        else 0
    score += 10 if flags["has_before_deleted"]    else 0
    score +=  8 if flags["has_mainstream_media"]  else 0
    score +=  8 if flags["has_whistleblower"]     else 0
    score +=  8 if flags["has_big_pharma"]        else 0
    score +=  5 if flags["has_wake_up"]           else 0
    score +=  5 if flags["has_anonymous"]         else 0
    score +=  5 if flags["has_exposed"]           else 0
    score  = min(score, 100)

    red_flags = []
    if flags["exclamation_marks"] > 0:
        red_flags.append(f"{flags['exclamation_marks']} exclamation mark(s)")
    if flags["caps_words"] > 0:
        red_flags.append(f"{flags['caps_words']} ALL CAPS word(s)")
    if flags["has_shocking"]:         red_flags.append("uses word 'SHOCKING'")
    if flags["has_breaking"]:         red_flags.append("uses word 'BREAKING'")
    if flags["has_share_this"]:       red_flags.append("urges readers to share")
    if flags["has_before_deleted"]:   red_flags.append("'share before deleted' — classic fake pattern")
    if flags["has_mainstream_media"]: red_flags.append("references 'mainstream media'")
    if flags["has_whistleblower"]:    red_flags.append("mentions unnamed whistleblower")
    if flags["has_big_pharma"]:       red_flags.append("mentions 'Big Pharma'")
    if flags["has_wake_up"]:          red_flags.append("uses 'wake up' phrase")
    if flags["has_anonymous"]:        red_flags.append("cites anonymous sources")

    if score >= 40:   style_verdict = "Highly Suspicious"
    elif score >= 20: style_verdict = "Somewhat Suspicious"
    else:             style_verdict = "Looks Normal"

    return {
        "score":     score,
        "verdict":   style_verdict,
        "red_flags": red_flags,
        "flags":     flags,
    }


# ─── Headline vs body analysis ────────────────────────────────────────────────

def analyse_headline_vs_body(headline: str, body: str,
                              vectoriser, model) -> dict:
    """
    Run prediction on headline and body separately.
    A large gap is a classic clickbait/fake signal.
    """
    h = predict(headline, vectoriser, model)
    b = predict(body,     vectoriser, model)
    gap = abs(h["fake_prob"] - b["fake_prob"])

    if gap > 0.35:
        disagreement = "High"
        message = ("⚠️ Headline and body give very different signals. "
                   "Sensational headline + mundane body is a classic fake pattern.")
    elif gap > 0.2:
        disagreement = "Moderate"
        message = "Headline and body are somewhat inconsistent — possible exaggerated framing."
    else:
        disagreement = "Low"
        message = "Headline and body are consistent with each other."

    return {
        "headline_label":     h["label"],
        "headline_fake_prob": h["fake_prob"],
        "body_label":         b["label"],
        "body_fake_prob":     b["fake_prob"],
        "gap":                round(gap, 3),
        "disagreement":       disagreement,
        "message":            message,
    }


# ─── TF-IDF word importance ───────────────────────────────────────────────────

def get_top_tfidf_words(text: str, vectoriser, model, n: int = 12) -> list:
    """
    Multiply TF-IDF weight × model coefficient for each word.
    Positive → pushes toward FAKE. Negative → pushes toward REAL.
    Only works with linear models (Logistic Regression).
    """
    clean = preprocess_text(text)
    vec   = vectoriser.transform([clean])

    if not hasattr(model, "coef_"):
        return []

    coefs         = model.coef_[0]
    feature_names = vectoriser.get_feature_names_out()
    nonzero_idx   = vec.nonzero()[1]

    word_scores = [
        (feature_names[i], float(vec[0, i] * coefs[i]))
        for i in nonzero_idx
    ]
    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return word_scores[:n]


# ─── LIME (optional, slower but more accurate) ────────────────────────────────

def get_lime_explanation(text: str, vectoriser, model,
                         num_features: int = 10) -> list:
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        print("[WARN] LIME not installed. Run: pip install lime")
        return []

    def predict_proba_pipeline(texts):
        cleaned = [preprocess_text(t) for t in texts]
        vecs    = vectoriser.transform(cleaned)
        return model.predict_proba(vecs)

    explainer   = LimeTextExplainer(class_names=["Real", "Fake"], random_state=42)
    explanation = explainer.explain_instance(
        text, predict_proba_pipeline,
        num_features=num_features, num_samples=300, labels=(1,)
    )
    return explanation.as_list(label=1)


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    vectoriser, model = load_artefacts()
    sample = ("SHOCKING: Scientists confirm 5G towers are spreading coronavirus. "
              "The government is hiding the truth from citizens. Share before deleted!")

    result = predict(sample, vectoriser, model)
    print(f"\nPrediction : {result['label']} ({result['confidence']:.1%})")

    style = analyse_writing_style(sample)
    print(f"Style score: {style['score']}/100 — {style['verdict']}")
    for flag in style["red_flags"]:
        print(f"  • {flag}")

    print("\nTop words:")
    for word, score in get_top_tfidf_words(sample, vectoriser, model):
        print(f"  {word:20s}  {score:+.4f}  {'→ FAKE' if score > 0 else '→ REAL'}")
