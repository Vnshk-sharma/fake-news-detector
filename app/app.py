"""
Fake News Detection System - Streamlit Web App (Upgraded)
Includes: Dark professional UI, CSV bulk upload, Live NewsAPI integration
Run with: streamlit run app/app.py
"""

import sys
import os
import io
import pickle
import time
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from preprocess import preprocess_text
from explain import predict, get_top_tfidf_words

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
DATA_DIR  = os.path.join(os.path.dirname(__file__), "../data")


# ─── Option B: Auto-download dataset + train on first launch ──────────────────

def models_exist():
    """Check if trained model files are already present."""
    return all(
        os.path.exists(os.path.join(MODEL_DIR, f))
        for f in ["best_model.pkl", "tfidf_vectoriser.pkl", "metadata.pkl"]
    )


def download_and_train():
    """
    Downloads the Kaggle ISOT fake news dataset from a public mirror
    and trains the model. Only runs once — results are cached to disk.
    """
    import urllib.request
    import zipfile

    os.makedirs(DATA_DIR,  exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    data_path = os.path.join(DATA_DIR, "news_fixed.csv")

    # ── Step 1: Download dataset ──────────────────────────────────────────────
    if not os.path.exists(data_path):
        st.info("📥 First launch — downloading dataset (~60MB). This takes 1–2 minutes...", icon="⏳")

        # Public mirror of the ISOT / WELFake dataset (permissive licence)
        # WELFake: 72K articles, pre-labelled, single CSV
        DATA_URL = "https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/WELFake_Dataset.csv"

        progress = st.progress(0, text="Downloading dataset...")
        try:
            urllib.request.urlretrieve(DATA_URL, data_path)
            progress.progress(40, text="Download complete — preparing data...")
        except Exception as e:
            progress.empty()
            st.error(f"Download failed: {e}. Please check your internet connection.")
            st.stop()

        # Normalise WELFake columns → text, label (0=real, 1=fake)
        df = pd.read_csv(data_path)
        df.columns = [c.lower().strip() for c in df.columns]

        # WELFake uses: title, text, label (1=fake, 0=real) — already correct
        if "title" in df.columns and "text" in df.columns:
            df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        df = df[["text", "label"]].dropna()
        df = df[df["text"].str.strip() != ""]
        df = df.drop_duplicates(subset=["text"])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.to_csv(data_path, index=False)
        progress.progress(60, text="Dataset ready — starting training...")

    # ── Step 2: Train model ───────────────────────────────────────────────────
    st.info("🤖 Training model on your data (~2–3 minutes on first launch)...", icon="⚙️")
    train_progress = st.progress(60, text="Loading data...")

    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    df = pd.read_csv(data_path)
    df["label"] = df["label"].astype(int)
    train_progress.progress(65, text="Preprocessing text...")

    # Preprocess (use a fast subset if dataset is very large)
    sample = df.sample(min(len(df), 30_000), random_state=42)
    sample["clean"] = sample["text"].apply(preprocess_text)
    train_progress.progress(75, text="Vectorising features...")

    X_train, X_test, y_train, y_test = train_test_split(
        sample["clean"], sample["label"],
        test_size=0.2, random_state=42, stratify=sample["label"]
    )

    vectoriser = TfidfVectorizer(
        max_features=100_000, ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, max_df=0.85
    )
    X_tr = vectoriser.fit_transform(X_train)
    X_te = vectoriser.transform(X_test)
    train_progress.progress(85, text="Fitting classifier...")

    model = LogisticRegression(
        max_iter=1000, C=5.0,
        class_weight="balanced", random_state=42
    )
    model.fit(X_tr, y_train)
    train_progress.progress(95, text="Evaluating model...")

    y_pred = model.predict(X_te)
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
    }
    metadata = {"name": "Logistic Regression", "metrics": metrics}

    # Save artefacts
    for obj, fname in [
        (vectoriser, "tfidf_vectoriser.pkl"),
        (model,      "best_model.pkl"),
        (metadata,   "metadata.pkl"),
    ]:
        with open(os.path.join(MODEL_DIR, fname), "wb") as f:
            pickle.dump(obj, f)

    train_progress.progress(100, text="Done!")
    train_progress.empty()
    st.success(
        f"✅ Model trained! Accuracy: {metrics['accuracy']:.1%} | "
        f"F1: {metrics['f1']:.1%}",
        icon="🎉"
    )
    st.rerun()


# ─── Run auto-train if models don't exist ─────────────────────────────────────

if not models_exist():
    # Show minimal page config before the training UI
    st.markdown("""
        <div style='text-align:center;padding:2rem 1rem'>
            <div style='font-size:3rem'>🔍</div>
            <h2 style='color:#a78bfa'>Fake News Detector</h2>
            <p style='color:#64748b'>Setting up for the first time...</p>
        </div>
    """, unsafe_allow_html=True)
    download_and_train()

st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="wide")

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.stApp { background-color: #0f1117; }
html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, sans-serif; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #1a1d27; border-right: 1px solid #2d3148; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stMetricValue"] { color: #a78bfa !important; font-size: 1.4rem !important; }
.stTabs [data-baseweb="tab-list"] { background: #1a1d27; border-radius: 12px; padding: 4px; gap: 4px; border: 1px solid #2d3148; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #94a3b8; font-weight: 500; padding: 8px 20px; }
.stTabs [aria-selected="true"] { background: #6d28d9 !important; color: white !important; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #6d28d9 0%, #4f46e5 100%); color: white; border: none; border-radius: 10px; font-weight: 600; font-size: 1rem; padding: 0.6rem 2rem; transition: transform 0.15s, box-shadow 0.15s; box-shadow: 0 4px 15px rgba(109, 40, 217, 0.4); }
.stButton > button[kind="primary"]:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(109, 40, 217, 0.6); }
.stButton > button[kind="secondary"] { background: #1e2130; color: #a78bfa; border: 1px solid #3730a3; border-radius: 8px; font-weight: 500; }
.stTextArea textarea { background: #1a1d27 !important; border: 1px solid #2d3148 !important; border-radius: 10px !important; color: #e2e8f0 !important; font-size: 0.95rem !important; line-height: 1.6 !important; }
.stTextArea textarea:focus { border-color: #6d28d9 !important; box-shadow: 0 0 0 2px rgba(109,40,217,0.2) !important; }
[data-testid="stFileUploader"] { background: #1a1d27; border: 2px dashed #3730a3; border-radius: 12px; padding: 1rem; }
.stProgress > div > div { background: #6d28d9; border-radius: 4px; }
.stTextInput input { background: #1a1d27 !important; border: 1px solid #2d3148 !important; border-radius: 8px !important; color: #e2e8f0 !important; }
.stDownloadButton > button { background: #1a1d27; color: #34d399; border: 1px solid #065f46; border-radius: 8px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    def _load(name):
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path): return None
        with open(path, "rb") as f: return pickle.load(f)
    return _load("tfidf_vectoriser.pkl"), _load("best_model.pkl"), _load("metadata.pkl")

def draw_confidence_gauge(confidence, label):
    fig, ax = plt.subplots(figsize=(4, 2.4), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117"); ax.set_axis_off()
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), lw=20, color="#1e2130", solid_capstyle="round")
    fill_theta = np.linspace(np.pi, np.pi - confidence * np.pi, 200)
    color = "#ef4444" if label == "FAKE" else "#22c55e"
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), lw=20, color=color, solid_capstyle="round")
    ax.text(0, -0.1, f"{confidence:.0%}", ha="center", va="center", fontsize=28, fontweight="bold", color=color)
    ax.text(0, -0.42, "Confidence", ha="center", va="center", fontsize=10, color="#64748b")
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-0.65, 1.35)
    st.pyplot(fig, use_container_width=True); plt.close(fig)

def draw_word_importance(word_scores):
    if not word_scores: return
    words = [w for w, _ in word_scores]; scores = [s for _, s in word_scores]
    colors = ["#ef4444" if s > 0 else "#22c55e" for s in scores]
    fig, ax = plt.subplots(figsize=(6, max(3, len(words) * 0.45)))
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a1d27")
    ax.barh(words, scores, color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color="#374151", lw=0.8, linestyle="--")
    ax.set_xlabel("Contribution  (+ → FAKE,  − → REAL)", fontsize=9, color="#64748b")
    ax.tick_params(axis="both", labelsize=9, colors="#94a3b8")
    for spine in ax.spines.values(): spine.set_edgecolor("#2d3148")
    ax.legend(handles=[mpatches.Patch(color="#ef4444", label="→ FAKE"),
                       mpatches.Patch(color="#22c55e", label="→ REAL")],
              fontsize=8, frameon=False, labelcolor="#94a3b8", loc="lower right")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

def render_sidebar(metadata, history):
    with st.sidebar:
        st.markdown("<div style='text-align:center;padding:1rem 0 0.5rem'><div style='font-size:2rem'>🔍</div><div style='font-size:1.1rem;font-weight:600;color:#a78bfa'>Fake News Detector</div><div style='font-size:0.75rem;color:#64748b;margin-top:4px'>AI-powered · NLP · ML</div></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### ⚙️ Model Performance")
        if metadata:
            m = metadata.get("metrics", {})
            c1, c2 = st.columns(2)
            c1.metric("Accuracy",  f"{m.get('accuracy',0):.1%}")
            c2.metric("F1-Score",  f"{m.get('f1',0):.1%}")
            c1.metric("Precision", f"{m.get('precision',0):.1%}")
            c2.metric("Recall",    f"{m.get('recall',0):.1%}")
            st.caption(f"Model: {metadata.get('name','N/A')}")
        st.markdown("---")
        if history:
            st.markdown("#### 📜 Recent Predictions")
            for h in reversed(history[-6:]):
                icon = "❌" if h["label"]=="FAKE" else "✅"
                color = "#ef4444" if h["label"]=="FAKE" else "#22c55e"
                st.markdown(f"<div style='font-size:12px;color:{color};padding:3px 0'>{icon} {h['snippet']} <span style='color:#64748b'>({h['confidence']:.0%})</span></div>", unsafe_allow_html=True)
            st.markdown("---")
        st.caption("Dataset: ISOT Fake News (44K articles)")
        st.caption("Stack: scikit-learn · NLTK · Streamlit")

def tab_single(vectoriser, model):
    st.markdown("<h2 style='font-size:1.5rem;font-weight:600;color:#e2e8f0;margin-bottom:0.2rem'>Analyse a Single Article</h2><p style='color:#64748b;margin-bottom:1.5rem;font-size:0.9rem'>Paste any news headline or full article text below</p>", unsafe_allow_html=True)
    for key, val in [("loaded_text",""),("history",[])]:
        if key not in st.session_state: st.session_state[key] = val
    with st.expander("📋 Load an example article"):
        c1, c2 = st.columns(2)
        if c1.button("🔴 Fake example", use_container_width=True):
            st.session_state.loaded_text = "SHOCKING: Government secretly adding fluoride to water supply to control the population. Whistleblowers reveal the truth mainstream media refuses to report. Big Pharma is hiding the cancer cure. Share before deleted!"
            st.rerun()
        if c2.button("🟢 Real example", use_container_width=True):
            st.session_state.loaded_text = "The Reserve Bank of India raised interest rates by 25 basis points on Wednesday, marking the sixth consecutive hike as policymakers continued efforts to bring inflation back to the 4 percent target. The decision was unanimous among all committee members."
            st.rerun()
    news_input = st.text_area("📰 Article text:", value=st.session_state.get("loaded_text",""), height=180, placeholder="Paste a news article or headline here...", label_visibility="collapsed")
    _, col_btn, _ = st.columns([1,2,1])
    with col_btn:
        analyse_btn = st.button("🔍  Analyse Article", use_container_width=True, type="primary", key="single_btn")
    if analyse_btn and news_input.strip():
        bar = st.progress(0, text="Preprocessing text...")
        for p in range(0,60,15): time.sleep(0.05); bar.progress(p, text="Preprocessing text...")
        result = predict(news_input, vectoriser, model)
        for p in range(60,101,10): time.sleep(0.04); bar.progress(p, text="Running classifier...")
        bar.empty()
        label = result["label"]; confidence = result["confidence"]
        st.session_state.history.append({"label":label,"confidence":confidence,"snippet":news_input[:45]+"..."})
        if label == "FAKE":
            st.markdown("<div style='background:#450a0a;border:1px solid #dc2626;border-radius:12px;padding:16px 20px;margin:12px 0;display:flex;align-items:center;gap:12px'><span style='font-size:1.8rem'>❌</span><div><div style='font-size:1.2rem;font-weight:700;color:#fca5a5'>FAKE NEWS DETECTED</div><div style='font-size:0.85rem;color:#f87171'>This article shows strong indicators of misinformation</div></div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#052e16;border:1px solid #16a34a;border-radius:12px;padding:16px 20px;margin:12px 0;display:flex;align-items:center;gap:12px'><span style='font-size:1.8rem'>✅</span><div><div style='font-size:1.2rem;font-weight:700;color:#86efac'>APPEARS CREDIBLE</div><div style='font-size:0.85rem;color:#4ade80'>This article shows characteristics of legitimate news</div></div></div>", unsafe_allow_html=True)
            st.balloons()
        col_a, col_b = st.columns(2)
        with col_a: draw_confidence_gauge(confidence, label)
        with col_b:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown("**Probability breakdown**")
            st.progress(result["real_prob"], text=f"✅ Real — {result['real_prob']:.1%}")
            st.progress(result["fake_prob"], text=f"❌ Fake — {result['fake_prob']:.1%}")
        st.markdown("---")
        st.markdown("### 🔬 Why this verdict?")
        st.caption("Words that most influenced the prediction")
        word_scores = get_top_tfidf_words(news_input, vectoriser, model, n=12)
        if word_scores: draw_word_importance(word_scores)
        with st.expander("🔧 View preprocessed text"):
            st.code(result.get("clean_text",""), language=None)
    elif analyse_btn:
        st.warning("Please enter some text first.", icon="⚠️")

def tab_bulk(vectoriser, model):
    st.markdown("<h2 style='font-size:1.5rem;font-weight:600;color:#e2e8f0;margin-bottom:0.2rem'>Bulk CSV Analysis</h2><p style='color:#64748b;margin-bottom:1.5rem;font-size:0.9rem'>Upload a CSV file — get every row classified and download the results</p>", unsafe_allow_html=True)
    with st.expander("📌 CSV format required"):
        st.markdown("Your CSV must have a column named `text`, `title`, `headline`, or `content`.\n\nAll other columns are kept in the output unchanged.")
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded is not None:
        try: df = pd.read_csv(uploaded)
        except Exception as e: st.error(f"Could not read CSV: {e}"); return
        text_col = next((c for c in ["text","title","headline","content","article"] if c in df.columns), None)
        if text_col is None: st.error(f"No usable text column found. Columns: {list(df.columns)}"); return
        st.success(f"✅ Loaded {len(df):,} rows — using column **'{text_col}'**")
        st.dataframe(df.head(3), use_container_width=True)
        if st.button("🚀  Run Bulk Analysis", type="primary", key="bulk_btn"):
            bar = st.progress(0, text="Starting...")
            results = []
            for i, row_text in enumerate(df[text_col].fillna("")):
                bar.progress(int((i+1)/len(df)*100), text=f"Analysing row {i+1} of {len(df)}...")
                results.append(predict(str(row_text), vectoriser, model))
            bar.empty()
            df["prediction"] = [r["label"] for r in results]
            df["confidence"] = [f"{r['confidence']:.1%}" for r in results]
            df["fake_prob"]  = [f"{r['fake_prob']:.3f}" for r in results]
            df["real_prob"]  = [f"{r['real_prob']:.3f}" for r in results]
            n_fake = (df["prediction"]=="FAKE").sum()
            c1,c2,c3 = st.columns(3)
            c1.metric("Total Articles", len(df))
            c2.metric("Flagged FAKE", n_fake, delta=f"{n_fake/len(df):.0%} of total", delta_color="inverse")
            c3.metric("Classified REAL", len(df)-n_fake)
            st.dataframe(df[[text_col,"prediction","confidence"]], use_container_width=True, height=300)
            st.download_button("⬇️  Download Full Results CSV", df.to_csv(index=False).encode("utf-8"), "fake_news_results.csv", "text/csv", use_container_width=True)
            st.toast(f"Done! Analysed {len(df):,} articles.", icon="✅")

def tab_live_news(vectoriser, model):
    st.markdown("<h2 style='font-size:1.5rem;font-weight:600;color:#e2e8f0;margin-bottom:0.2rem'>Live News Headlines</h2><p style='color:#64748b;margin-bottom:1.5rem;font-size:0.9rem'>Fetch today's top headlines and run them through the detector in real time</p>", unsafe_allow_html=True)
    st.markdown("**Step 1 — Get a free API key** at [newsapi.org](https://newsapi.org) (takes 2 minutes, free)")
    api_key = st.text_input("Paste your NewsAPI key here:", type="password", placeholder="e.g. a1b2c3d4e5f6...")
    c1, c2 = st.columns(2)
    category = c1.selectbox("Category", ["general","technology","science","health","business","entertainment"])
    country  = c2.selectbox("Country",  ["us","gb","in","au","ca"])
    fetch_btn = st.button("📡  Fetch & Analyse Headlines", type="primary", key="live_btn", disabled=not api_key)
    if not api_key: st.info("Enter your NewsAPI key above to enable live fetching.", icon="🔑")
    if fetch_btn and api_key:
        try: from newsapi import NewsApiClient
        except ImportError: st.error("Run: `pip install newsapi-python` then restart the app"); return
        with st.spinner("Fetching headlines..."):
            try:
                client = NewsApiClient(api_key=api_key)
                response = client.get_top_headlines(language="en", country=country, category=category, page_size=15)
            except Exception as e: st.error(f"NewsAPI error: {e}"); return
        articles = response.get("articles", [])
        if not articles: st.warning("No articles returned. Try a different category."); return
        st.success(f"Fetched {len(articles)} headlines — analysing...")
        st.markdown("---")
        fake_count = 0
        for article in articles:
            title  = article.get("title") or ""
            source = article.get("source",{}).get("name","Unknown")
            url    = article.get("url","#")
            if not title or title == "[Removed]": continue
            result = predict(title, vectoriser, model)
            label  = result["label"]; conf = result["confidence"]
            if label == "FAKE": fake_count += 1
            bc = "#dc2626" if label=="FAKE" else "#16a34a"
            bg = "#2d0a0a" if label=="FAKE" else "#0a2d1a"
            ic = "❌" if label=="FAKE" else "✅"
            lc = "#fca5a5" if label=="FAKE" else "#86efac"
            st.markdown(f"<div style='background:{bg};border:1px solid {bc};border-radius:10px;padding:12px 16px;margin-bottom:10px'><div style='display:flex;justify-content:space-between;align-items:flex-start'><div style='flex:1;margin-right:12px'><div style='font-size:0.95rem;color:#e2e8f0;font-weight:500;line-height:1.4;margin-bottom:4px'>{title}</div><div style='font-size:0.78rem;color:#64748b'>{source} &nbsp;·&nbsp; <a href='{url}' target='_blank' style='color:#6d28d9;text-decoration:none'>Read original ↗</a></div></div><div style='text-align:center;flex-shrink:0;min-width:70px'><div style='font-size:1.3rem'>{ic}</div><div style='font-size:0.75rem;font-weight:600;color:{lc}'>{label}</div><div style='font-size:0.7rem;color:#64748b'>{conf:.0%}</div></div></div></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(f"<div style='text-align:center;color:#94a3b8;font-size:0.9rem;padding:8px 0'>Analysed <b>{len(articles)}</b> headlines — <span style='color:#ef4444'><b>{fake_count} flagged FAKE</b></span> | <span style='color:#22c55e'><b>{len(articles)-fake_count} appear credible</b></span></div>", unsafe_allow_html=True)

def main():
    vectoriser, model, metadata = load_models()
    if vectoriser is None or model is None:
        st.error("Models not found — run `python src/train.py` first.", icon="🚨"); st.stop()
    if "history" not in st.session_state: st.session_state.history = []
    render_sidebar(metadata, st.session_state.history)
    st.markdown("<div style='text-align:center;padding:0.5rem 0 1.5rem'><h1 style='font-size:2.2rem;font-weight:700;margin:0;background:linear-gradient(135deg,#a78bfa,#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent'>🔍 Fake News Detector</h1><p style='color:#64748b;margin-top:6px;font-size:0.95rem'>AI-powered misinformation detection · 98% accuracy · NLP + ML</p></div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📰  Single Article","📂  Bulk CSV Upload","🌐  Live News API"])
    with tab1: tab_single(vectoriser, model)
    with tab2: tab_bulk(vectoriser, model)
    with tab3: tab_live_news(vectoriser, model)
    st.markdown("<hr style='border-color:#1e2130;margin-top:2rem'/><p style='text-align:center;color:#374151;font-size:0.78rem;padding-bottom:1rem'>Built with Python · scikit-learn · NLTK · Streamlit | For educational purposes only</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
