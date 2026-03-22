"""
Fake News Detection System - Gen-Z New Age UI
Glassmorphism + Cyberpunk + Scroll animations + Interactive particles
Run with: streamlit run app/app.py
"""

import sys
import os
import pickle
import time
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import nltk

nltk.download("punkt",     quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)
nltk.download("punkt_tab", quiet=True)

st.set_page_config(page_title="TruthLens AI", page_icon="🔍", layout="wide",
                   initial_sidebar_state="collapsed")

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from preprocess import preprocess_text
from explain import predict, get_top_tfidf_words, analyse_writing_style, analyse_headline_vs_body

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")

# ── MEGA CSS: Gen-Z New Age ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:       #04030a;
  --bg2:      #080614;
  --surface:  rgba(255,255,255,0.03);
  --border:   rgba(255,255,255,0.07);
  --border2:  rgba(255,255,255,0.12);
  --accent1:  #b06ef3;
  --accent2:  #6ee7f7;
  --accent3:  #f96b6b;
  --accent4:  #6bf9a0;
  --text1:    #f0eeff;
  --text2:    #9890b8;
  --text3:    #5a5470;
  --glow1:    rgba(176,110,243,0.15);
  --glow2:    rgba(110,231,247,0.1);
}

/* ── Reset + Base ── */
html, body, [class*="css"], .stApp {
  font-family: 'DM Sans', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text1) !important;
}
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--accent1); border-radius: 2px; }

/* ── Animated background mesh ── */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 50% at 20% 10%, rgba(176,110,243,0.08) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 80%, rgba(110,231,247,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 40% 30% at 50% 50%, rgba(249,107,107,0.04) 0%, transparent 60%);
  pointer-events: none;
  z-index: 0;
  animation: meshShift 12s ease-in-out infinite alternate;
}
@keyframes meshShift {
  0%   { opacity: 0.6; transform: scale(1); }
  100% { opacity: 1;   transform: scale(1.05); }
}

/* ── Floating grid lines ── */
.stApp::after {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
  background-size: 60px 60px;
  pointer-events: none;
  z-index: 0;
}

/* ── All content above bg ── */
.main > div { position: relative; z-index: 1; }

/* ── Hero section ── */
.hero {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 40px;
  text-align: center;
  position: relative;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(176,110,243,0.1);
  border: 1px solid rgba(176,110,243,0.3);
  border-radius: 999px;
  padding: 6px 16px;
  font-size: 12px;
  font-family: 'JetBrains Mono', monospace;
  color: var(--accent1);
  letter-spacing: 0.08em;
  margin-bottom: 32px;
  animation: fadeUp 0.8s ease both;
}
.hero-badge::before {
  content: '';
  width: 6px; height: 6px;
  background: var(--accent1);
  border-radius: 50%;
  animation: pulse 2s ease infinite;
}
@keyframes pulse {
  0%,100% { opacity: 1; transform: scale(1); }
  50%      { opacity: 0.4; transform: scale(0.7); }
}
.hero-title {
  font-family: 'Syne', sans-serif !important;
  font-size: clamp(3rem, 8vw, 7rem) !important;
  font-weight: 800 !important;
  line-height: 0.95 !important;
  letter-spacing: -0.03em !important;
  margin-bottom: 24px !important;
  animation: fadeUp 0.8s 0.1s ease both;
}
.hero-title .line1 { color: var(--text1); display: block; }
.hero-title .line2 {
  display: block;
  background: linear-gradient(135deg, var(--accent1), var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  font-size: 1.1rem;
  color: var(--text2);
  max-width: 500px;
  line-height: 1.7;
  margin: 0 auto 48px;
  animation: fadeUp 0.8s 0.2s ease both;
}
.hero-stats {
  display: flex;
  gap: 48px;
  justify-content: center;
  margin-bottom: 56px;
  animation: fadeUp 0.8s 0.3s ease both;
}
.stat {
  text-align: center;
}
.stat-num {
  font-family: 'Syne', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--accent1), var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.stat-label {
  font-size: 0.75rem;
  color: var(--text3);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-top: 4px;
}
.scroll-hint {
  position: absolute;
  bottom: 40px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  color: var(--text3);
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  animation: fadeUp 0.8s 0.6s ease both;
}
.scroll-line {
  width: 1px;
  height: 48px;
  background: linear-gradient(to bottom, var(--accent1), transparent);
  animation: scrollLine 2s ease infinite;
}
@keyframes scrollLine {
  0%   { transform: scaleY(0); transform-origin: top; opacity: 1; }
  50%  { transform: scaleY(1); transform-origin: top; opacity: 1; }
  100% { transform: scaleY(1); transform-origin: bottom; opacity: 0; }
}

/* ── Section wrapper ── */
.section {
  padding: 80px 40px;
  max-width: 1100px;
  margin: 0 auto;
}
.section-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--accent1);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-bottom: 12px;
}
.section-title {
  font-family: 'Syne', sans-serif;
  font-size: clamp(1.8rem, 4vw, 3rem);
  font-weight: 700;
  line-height: 1.1;
  margin-bottom: 48px;
  color: var(--text1);
}

/* ── Glass card ── */
.glass-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 32px;
  backdrop-filter: blur(20px);
  transition: border-color 0.3s, transform 0.3s;
  position: relative;
  overflow: hidden;
}
.glass-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.02), transparent);
  pointer-events: none;
}
.glass-card:hover {
  border-color: var(--border2);
  transform: translateY(-2px);
}

/* ── Input styles ── */
.stTextArea textarea, .stTextInput input {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 14px !important;
  color: var(--text1) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.95rem !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: var(--accent1) !important;
  box-shadow: 0 0 0 3px rgba(176,110,243,0.12) !important;
  outline: none !important;
}
.stTextArea label, .stTextInput label {
  color: var(--text2) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--accent1), #7c3aed) !important;
  color: white !important;
  border: none !important;
  border-radius: 14px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
  padding: 0.8rem 2.5rem !important;
  letter-spacing: 0.02em !important;
  transition: all 0.2s !important;
  box-shadow: 0 0 30px rgba(176,110,243,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 0 50px rgba(176,110,243,0.5) !important;
}
.stButton > button[kind="primary"]:active {
  transform: scale(0.97) !important;
}

/* ── Secondary button ── */
.stButton > button[kind="secondary"] {
  background: transparent !important;
  color: var(--accent1) !important;
  border: 1px solid rgba(176,110,243,0.3) !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.85rem !important;
  transition: all 0.2s !important;
}
.stButton > button[kind="secondary"]:hover {
  background: rgba(176,110,243,0.08) !important;
  border-color: var(--accent1) !important;
}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 6px !important;
  gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px !important;
  color: var(--text2) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  padding: 10px 22px !important;
  transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--accent1), #7c3aed) !important;
  color: white !important;
  box-shadow: 0 0 20px rgba(176,110,243,0.3) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--accent1), var(--accent2)) !important;
  border-radius: 4px !important;
  box-shadow: 0 0 10px rgba(176,110,243,0.4) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 20px !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  color: var(--accent1) !important;
}
[data-testid="stMetricLabel"] {
  color: var(--text3) !important;
  font-size: 11px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text2) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.02) !important;
  border: 1px dashed rgba(176,110,243,0.3) !important;
  border-radius: 16px !important;
}

/* ── Download button ── */
.stDownloadButton > button {
  background: rgba(107,249,160,0.08) !important;
  color: var(--accent4) !important;
  border: 1px solid rgba(107,249,160,0.2) !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* ── Verdict cards ── */
.verdict-fake {
  background: linear-gradient(135deg, rgba(249,107,107,0.08), rgba(249,107,107,0.03));
  border: 1px solid rgba(249,107,107,0.3);
  border-radius: 20px;
  padding: 28px 32px;
  margin: 20px 0;
  display: flex;
  align-items: center;
  gap: 20px;
  animation: verdictIn 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
  box-shadow: 0 0 60px rgba(249,107,107,0.08), inset 0 1px 0 rgba(255,255,255,0.05);
}
.verdict-real {
  background: linear-gradient(135deg, rgba(107,249,160,0.08), rgba(107,249,160,0.03));
  border: 1px solid rgba(107,249,160,0.3);
  border-radius: 20px;
  padding: 28px 32px;
  margin: 20px 0;
  display: flex;
  align-items: center;
  gap: 20px;
  animation: verdictIn 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
  box-shadow: 0 0 60px rgba(107,249,160,0.08), inset 0 1px 0 rgba(255,255,255,0.05);
}
.verdict-uncertain {
  background: linear-gradient(135deg, rgba(251,191,36,0.08), rgba(251,191,36,0.03));
  border: 1px solid rgba(251,191,36,0.3);
  border-radius: 20px;
  padding: 28px 32px;
  margin: 20px 0;
  display: flex;
  align-items: center;
  gap: 20px;
  animation: verdictIn 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
  box-shadow: 0 0 60px rgba(251,191,36,0.08), inset 0 1px 0 rgba(255,255,255,0.05);
}
@keyframes verdictIn {
  from { opacity: 0; transform: scale(0.92) translateY(10px); }
  to   { opacity: 1; transform: scale(1)    translateY(0); }
}
.verdict-icon { font-size: 2.5rem; flex-shrink: 0; }
.verdict-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.4rem;
  font-weight: 700;
  margin-bottom: 4px;
}
.verdict-sub { font-size: 0.85rem; opacity: 0.7; }

/* ── Feature cards ── */
.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin-top: 40px;
}
.feature-card {
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 28px;
  transition: all 0.3s;
  cursor: default;
}
.feature-card:hover {
  border-color: rgba(176,110,243,0.3);
  background: rgba(176,110,243,0.04);
  transform: translateY(-4px);
  box-shadow: 0 20px 60px rgba(176,110,243,0.1);
}
.feature-icon {
  font-size: 1.8rem;
  margin-bottom: 16px;
}
.feature-title {
  font-family: 'Syne', sans-serif;
  font-size: 1rem;
  font-weight: 600;
  color: var(--text1);
  margin-bottom: 8px;
}
.feature-desc {
  font-size: 0.82rem;
  color: var(--text3);
  line-height: 1.6;
}

/* ── Score ring ── */
.score-ring-wrap {
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 28px;
  text-align: center;
}
.score-num {
  font-family: 'Syne', sans-serif;
  font-size: 3rem;
  font-weight: 800;
  line-height: 1;
  margin-bottom: 6px;
}
.score-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text3);
}

/* ── Red flag list ── */
.flag-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.85rem;
  color: var(--text2);
  animation: fadeUp 0.3s ease both;
}
.flag-item:last-child { border-bottom: none; }
.flag-dot {
  width: 6px; height: 6px;
  background: var(--accent3);
  border-radius: 50%;
  flex-shrink: 0;
}

/* ── Divider ── */
.divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border2), transparent);
  margin: 40px 0;
}

/* ── Headline vs body cards ── */
.hb-card {
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px;
  text-align: center;
  transition: all 0.3s;
}
.hb-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text3);
  margin-bottom: 10px;
}
.hb-verdict {
  font-family: 'Syne', sans-serif;
  font-size: 1.2rem;
  font-weight: 700;
  margin-bottom: 6px;
}
.hb-prob {
  font-size: 0.78rem;
  color: var(--text3);
}

/* ── Footer ── */
.footer {
  text-align: center;
  padding: 60px 40px;
  border-top: 1px solid var(--border);
  color: var(--text3);
  font-size: 0.78rem;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.05em;
}

/* ── Animations ── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp 0.6s ease both; }

/* ── Scroll reveal (JS-driven) ── */
.reveal {
  opacity: 0;
  transform: translateY(30px);
  transition: opacity 0.7s ease, transform 0.7s ease;
}
.reveal.visible {
  opacity: 1;
  transform: translateY(0);
}

/* ── Info/warning/success ── */
.stAlert {
  background: rgba(255,255,255,0.02) !important;
  border-radius: 12px !important;
  border: 1px solid var(--border2) !important;
}

/* ── Dataframe ── */
.stDataFrame {
  border-radius: 16px !important;
  overflow: hidden !important;
  border: 1px solid var(--border) !important;
}
</style>

<script>
// Scroll reveal
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => { if(e.isIntersecting) e.target.classList.add('visible'); });
}, { threshold: 0.1 });
document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

// Custom cursor glow
document.addEventListener('mousemove', (e) => {
  let glow = document.getElementById('cursor-glow');
  if (!glow) {
    glow = document.createElement('div');
    glow.id = 'cursor-glow';
    glow.style.cssText = 'position:fixed;width:300px;height:300px;border-radius:50%;background:radial-gradient(circle,rgba(176,110,243,0.06),transparent 70%);pointer-events:none;z-index:9999;transform:translate(-50%,-50%);transition:left 0.1s,top 0.1s;';
    document.body.appendChild(glow);
  }
  glow.style.left = e.clientX + 'px';
  glow.style.top  = e.clientY + 'px';
});
</script>
""", unsafe_allow_html=True)


# ── Model check ───────────────────────────────────────────────────────────────
def models_exist():
    return all(
        os.path.exists(os.path.join(MODEL_DIR, f))
        for f in ["best_model.pkl", "tfidf_vectoriser.pkl", "metadata.pkl"]
    )

if not models_exist():
    st.markdown("""
    <div style='height:100vh;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:16px'>
      <div style='font-size:3rem'>⚠️</div>
      <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:700;color:#f0eeff'>Model files not found</div>
      <div style='color:#9890b8;font-size:0.9rem'>Run python src/train.py to generate model files</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    def _load(name):
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path): return None
        with open(path, "rb") as f: return pickle.load(f)
    return _load("tfidf_vectoriser.pkl"), _load("best_model.pkl"), _load("metadata.pkl")

vectoriser, model, metadata = load_models()
if vectoriser is None or model is None:
    st.error("Model loading failed."); st.stop()
m = metadata.get("metrics", {}) if metadata else {}


# ── HERO SECTION ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-badge">✦ AI-POWERED TRUTH DETECTION</div>
  <h1 class="hero-title">
    <span class="line1">TruthLens</span>
    <span class="line2">AI</span>
  </h1>
  <p class="hero-sub">
    Advanced NLP system trained on 44,898 articles.
    Detects fake news with {m.get('accuracy', 0):.0%} accuracy using
    machine learning + linguistic analysis.
  </p>
  <div class="hero-stats">
    <div class="stat">
      <div class="stat-num">{m.get('accuracy', 0):.1%}</div>
      <div class="stat-label">Accuracy</div>
    </div>
    <div class="stat">
      <div class="stat-num">{m.get('f1', 0):.1%}</div>
      <div class="stat-label">F1 Score</div>
    </div>
    <div class="stat">
      <div class="stat-num">44K</div>
      <div class="stat-label">Articles</div>
    </div>
    <div class="stat">
      <div class="stat-num">100K</div>
      <div class="stat-label">Features</div>
    </div>
  </div>
  <div class="scroll-hint">
    <div class="scroll-line"></div>
    scroll to analyse
  </div>
</div>
""", unsafe_allow_html=True)


# ── FEATURE GRID ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="section reveal">
  <div class="section-label">// CAPABILITIES</div>
  <div class="section-title">What TruthLens can do</div>
  <div class="feature-grid">
    <div class="feature-card">
      <div class="feature-icon">🧠</div>
      <div class="feature-title">ML Classification</div>
      <div class="feature-desc">TF-IDF + Logistic Regression trained on 44K real and fake articles</div>
    </div>
    <div class="feature-card">
      <div class="feature-icon">✍️</div>
      <div class="feature-title">Writing Style Analysis</div>
      <div class="feature-desc">Detects red flags like ALL CAPS, exclamations, emotional triggers</div>
    </div>
    <div class="feature-card">
      <div class="feature-icon">📰</div>
      <div class="feature-title">Headline vs Body</div>
      <div class="feature-desc">Catches clickbait by comparing headline and body signals</div>
    </div>
    <div class="feature-card">
      <div class="feature-icon">📂</div>
      <div class="feature-title">Bulk Analysis</div>
      <div class="feature-desc">Upload a CSV and classify thousands of articles at once</div>
    </div>
    <div class="feature-card">
      <div class="feature-icon">🌐</div>
      <div class="feature-title">Live News API</div>
      <div class="feature-desc">Fetch and classify today's real headlines from NewsAPI</div>
    </div>
    <div class="feature-card">
      <div class="feature-icon">🔬</div>
      <div class="feature-title">Explainability</div>
      <div class="feature-desc">See exactly which words drove the prediction, with scores</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ── ANALYSER SECTION ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section reveal">
  <div class="section-label">// ANALYSER</div>
  <div class="section-title">Analyse an article</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📰  Single Article", "📂  Bulk CSV", "🌐  Live News"])

with tab1:
    if "history" not in st.session_state: st.session_state.history = []
    if "loaded_text" not in st.session_state: st.session_state.loaded_text = ""

    with st.expander("⚡ Load example"):
        c1, c2 = st.columns(2)
        if c1.button("🔴 Fake article", use_container_width=True):
            st.session_state.loaded_text = "SHOCKING: Government secretly adding fluoride to water supply to control the population. Whistleblowers reveal the truth mainstream media refuses to report. Big Pharma is hiding the cancer cure. Share before deleted!"
            st.rerun()
        if c2.button("🟢 Real article", use_container_width=True):
            st.session_state.loaded_text = "The Federal Reserve raised interest rates by 25 basis points on Wednesday as policymakers continued efforts to bring inflation back to the 2 percent target. The decision was unanimous among committee members."
            st.rerun()

    headline   = st.text_input("HEADLINE", placeholder="Paste the article headline...", label_visibility="visible")
    news_input = st.text_area("ARTICLE BODY", value=st.session_state.get("loaded_text",""), height=160, placeholder="Paste the full article body here...", label_visibility="visible")

    _, col_btn, _ = st.columns([1,2,1])
    with col_btn:
        analyse_btn = st.button("⟶  Analyse Now", use_container_width=True, type="primary")

    if analyse_btn and (news_input.strip() or headline.strip()):
        full_text = (headline + " " + news_input).strip()

        bar = st.progress(0, text="Processing...")
        for p in [10,25,40,55]:
            time.sleep(0.04); bar.progress(p, text="Preprocessing text...")
        result = predict(full_text, vectoriser, model)
        for p in [65,80,95,100]:
            time.sleep(0.03); bar.progress(p, text="Running classifier...")
        bar.empty()

        label      = result["label"]
        confidence = result["confidence"]
        st.session_state.history.append({"label":label,"confidence":confidence,"snippet":full_text[:45]+"..."})

        # Verdict
        if label == "FAKE":
            st.markdown(f"""
            <div class="verdict-fake">
              <div class="verdict-icon">❌</div>
              <div>
                <div class="verdict-title" style="color:#f96b6b">FAKE NEWS DETECTED</div>
                <div class="verdict-sub" style="color:#f96b6b">Strong misinformation indicators · {confidence:.0%} confidence</div>
              </div>
            </div>""", unsafe_allow_html=True)
        elif label == "REAL":
            st.markdown(f"""
            <div class="verdict-real">
              <div class="verdict-icon">✅</div>
              <div>
                <div class="verdict-title" style="color:#6bf9a0">APPEARS CREDIBLE</div>
                <div class="verdict-sub" style="color:#6bf9a0">Characteristics of legitimate news · {confidence:.0%} confidence</div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class="verdict-uncertain">
              <div class="verdict-icon">⚠️</div>
              <div>
                <div class="verdict-title" style="color:#fbbf24">UNCERTAIN</div>
                <div class="verdict-sub" style="color:#fbbf24">Model not confident · Verify from primary sources · {confidence:.0%} confidence</div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Probability bars
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.progress(result["real_prob"], text=f"✅ Real — {result['real_prob']:.1%}")
        with col_p2:
            st.progress(result["fake_prob"], text=f"❌ Fake — {result['fake_prob']:.1%}")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Headline vs body
        if headline.strip() and news_input.strip():
            st.markdown("<div class='section-label' style='margin-bottom:16px'>// HEADLINE VS BODY</div>", unsafe_allow_html=True)
            hb = analyse_headline_vs_body(headline, news_input, vectoriser, model)
            hc = "#f96b6b" if hb["headline_label"]=="FAKE" else "#6bf9a0" if hb["headline_label"]=="REAL" else "#fbbf24"
            bc2= "#f96b6b" if hb["body_label"]=="FAKE"     else "#6bf9a0" if hb["body_label"]=="REAL"     else "#fbbf24"
            col_h, col_b2 = st.columns(2)
            with col_h:
                st.markdown(f"""<div class="hb-card">
                  <div class="hb-label">Headline Signal</div>
                  <div class="hb-verdict" style="color:{hc}">{hb['headline_label']}</div>
                  <div class="hb-prob">{hb['headline_fake_prob']:.0%} fake probability</div>
                </div>""", unsafe_allow_html=True)
            with col_b2:
                st.markdown(f"""<div class="hb-card">
                  <div class="hb-label">Body Signal</div>
                  <div class="hb-verdict" style="color:{bc2}">{hb['body_label']}</div>
                  <div class="hb-prob">{hb['body_fake_prob']:.0%} fake probability</div>
                </div>""", unsafe_allow_html=True)
            if hb["disagreement"] == "High":
                st.warning(hb["message"])
            elif hb["disagreement"] == "Moderate":
                st.info(hb["message"])
            else:
                st.success(hb["message"])
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Word importance
        st.markdown("<div class='section-label' style='margin-bottom:16px'>// WORD IMPORTANCE</div>", unsafe_allow_html=True)
        word_scores = get_top_tfidf_words(full_text, vectoriser, model, n=12)
        if word_scores:
            words  = [w for w, _ in word_scores]
            scores = [s for _, s in word_scores]
            colors_bar = ["#f96b6b" if s > 0 else "#6bf9a0" for s in scores]
            fig, ax = plt.subplots(figsize=(7, max(3, len(words)*0.42)))
            fig.patch.set_facecolor("#04030a"); ax.set_facecolor("#080614")
            ax.barh(words, scores, color=colors_bar, edgecolor="none", height=0.55)
            ax.axvline(0, color="rgba(255,255,255,0.1)", lw=0.8, linestyle="--")
            ax.set_xlabel("Contribution  (+ → FAKE  / − → REAL)", fontsize=9, color="#5a5470", fontfamily="monospace")
            ax.tick_params(axis="both", labelsize=9, colors="#9890b8", labelcolor="#9890b8")
            for spine in ax.spines.values(): spine.set_edgecolor("#1a1730")
            ax.legend(handles=[mpatches.Patch(color="#f96b6b",label="→ FAKE"),
                                mpatches.Patch(color="#6bf9a0",label="→ REAL")],
                      fontsize=8, frameon=False, labelcolor="#9890b8", loc="lower right")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Writing style
        st.markdown("<div class='section-label' style='margin-bottom:16px'>// WRITING STYLE ANALYSIS</div>", unsafe_allow_html=True)
        style = analyse_writing_style(full_text)
        s_color = "#f96b6b" if style["score"]>=40 else "#fbbf24" if style["score"]>=20 else "#6bf9a0"

        col_s1, col_s2 = st.columns([1,2])
        with col_s1:
            st.markdown(f"""
            <div class="score-ring-wrap">
              <div class="score-num" style="color:{s_color}">{style['score']}</div>
              <div style="color:{s_color};font-size:0.75rem;margin-bottom:8px;font-weight:600">{style['verdict']}</div>
              <div class="score-label">/ 100 suspicion score</div>
            </div>""", unsafe_allow_html=True)
        with col_s2:
            if style["red_flags"]:
                flags_html = "".join([f"<div class='flag-item'><div class='flag-dot'></div>{f}</div>" for f in style["red_flags"]])
                st.markdown(f"<div style='padding-top:8px'>{flags_html}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='padding-top:20px;color:#6bf9a0;font-size:0.9rem'>✓ No red flags — writing style looks normal</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='margin-top:24px;padding:14px 18px;background:rgba(255,255,255,0.02);
             border:1px solid rgba(255,255,255,0.06);border-radius:12px;
             font-family:JetBrains Mono,monospace;font-size:11px;color:#5a5470;
             letter-spacing:0.05em'>
          ⚠ AI TOOL — Always verify from primary sources: Reuters · AP · BBC · FactCheck.org
        </div>""", unsafe_allow_html=True)

        with st.expander("// preprocessed tokens"):
            st.code(result.get("clean_text",""), language=None)

    elif analyse_btn:
        st.warning("Please enter some text to analyse.", icon="⚠️")

with tab2:
    st.markdown("<div style='padding:8px 0 24px;color:#9890b8;font-size:0.9rem'>Upload a CSV with a <code>text</code> or <code>title</code> column — classify everything at once.</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop your CSV here", type=["csv"], label_visibility="collapsed")
    if uploaded:
        try: df = pd.read_csv(uploaded)
        except Exception as e: st.error(f"Could not read CSV: {e}")
        else:
            text_col = next((c for c in ["text","title","headline","content","article"] if c in df.columns), None)
            if not text_col: st.error(f"No text column found. Columns: {list(df.columns)}")
            else:
                st.success(f"✅ {len(df):,} rows loaded — using '{text_col}'")
                st.dataframe(df.head(3), use_container_width=True)
                if st.button("🚀 Run Bulk Analysis", type="primary", key="bulk_btn"):
                    bar = st.progress(0)
                    results = []
                    for i, row in enumerate(df[text_col].fillna("")):
                        bar.progress(int((i+1)/len(df)*100), text=f"Row {i+1} / {len(df)}")
                        results.append(predict(str(row), vectoriser, model))
                    bar.empty()
                    df["prediction"] = [r["label"] for r in results]
                    df["confidence"] = [f"{r['confidence']:.1%}" for r in results]
                    df["fake_prob"]  = [f"{r['fake_prob']:.3f}" for r in results]
                    n_fake = (df["prediction"]=="FAKE").sum()
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Total", len(df))
                    c2.metric("FAKE",  n_fake)
                    c3.metric("REAL",  len(df)-n_fake)
                    st.dataframe(df[[text_col,"prediction","confidence"]], use_container_width=True, height=300)
                    st.download_button("⬇ Download Results", df.to_csv(index=False).encode(), "results.csv", "text/csv", use_container_width=True)
                    st.toast(f"Done! {len(df):,} articles analysed.", icon="✅")

with tab3:
    st.markdown("<div style='padding:8px 0 24px;color:#9890b8;font-size:0.9rem'>Get a free API key at <a href='https://newsapi.org' style='color:#b06ef3'>newsapi.org</a> and analyse today's headlines live.</div>", unsafe_allow_html=True)
    api_key  = st.text_input("NEWSAPI KEY", type="password", placeholder="Paste your key here...", label_visibility="visible")
    c1, c2   = st.columns(2)
    category = c1.selectbox("CATEGORY", ["general","technology","science","health","business","entertainment"], label_visibility="visible")
    country  = c2.selectbox("COUNTRY",  ["us","gb","in","au","ca"], label_visibility="visible")
    if st.button("📡 Fetch & Analyse", type="primary", key="live_btn", disabled=not api_key):
        try:
            from newsapi import NewsApiClient
            with st.spinner("Fetching live headlines..."):
                client   = NewsApiClient(api_key=api_key)
                response = client.get_top_headlines(language="en",country=country,category=category,page_size=15)
            articles = response.get("articles",[])
            if not articles: st.warning("No articles returned.")
            else:
                st.success(f"Fetched {len(articles)} headlines")
                fake_count = 0
                for article in articles:
                    title  = article.get("title") or ""
                    source = article.get("source",{}).get("name","Unknown")
                    url    = article.get("url","#")
                    if not title or title=="[Removed]": continue
                    r = predict(title, vectoriser, model)
                    if r["label"]=="FAKE": fake_count+=1
                    bc = "#f96b6b" if r["label"]=="FAKE" else "#fbbf24" if r["label"]=="UNCERTAIN" else "#6bf9a0"
                    bg = "rgba(249,107,107,0.05)" if r["label"]=="FAKE" else "rgba(251,191,36,0.05)" if r["label"]=="UNCERTAIN" else "rgba(107,249,160,0.05)"
                    ic = "❌" if r["label"]=="FAKE" else "⚠️" if r["label"]=="UNCERTAIN" else "✅"
                    st.markdown(f"""
                    <div style='background:{bg};border:1px solid {bc}33;border-radius:14px;
                         padding:14px 18px;margin-bottom:10px;display:flex;
                         justify-content:space-between;align-items:center;gap:16px'>
                      <div style='flex:1'>
                        <div style='font-size:0.9rem;color:#f0eeff;font-weight:500;margin-bottom:4px'>{title}</div>
                        <div style='font-size:0.75rem;color:#5a5470'>{source} · <a href="{url}" target="_blank" style="color:#b06ef3;text-decoration:none">Read ↗</a></div>
                      </div>
                      <div style='text-align:center;min-width:60px'>
                        <div style='font-size:1.4rem'>{ic}</div>
                        <div style='font-size:0.7rem;font-weight:600;color:{bc}'>{r["label"]}</div>
                        <div style='font-size:0.68rem;color:#5a5470'>{r["confidence"]:.0%}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center;color:#5a5470;font-size:0.8rem;padding:16px 0;font-family:monospace'>{len(articles)} headlines · {fake_count} flagged FAKE</div>", unsafe_allow_html=True)
        except ImportError:
            st.error("Run: pip install newsapi-python")
        except Exception as e:
            st.error(f"Error: {e}")
    elif not api_key:
        st.info("Enter your NewsAPI key above to enable live fetching.", icon="🔑")


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  TRUTHLENS AI &nbsp;·&nbsp; Built with Python · scikit-learn · NLTK · Streamlit
  &nbsp;·&nbsp; For educational purposes only &nbsp;·&nbsp; Always verify from primary sources
</div>
""", unsafe_allow_html=True)
