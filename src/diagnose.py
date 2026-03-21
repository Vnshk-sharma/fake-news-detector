"""
Dataset Diagnosis Script
Run this FIRST before trying any model improvements.
It prints a full health report of your dataset.

Usage:
    python src/diagnose.py
"""

import pandas as pd
import numpy as np
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
from preprocess import preprocess_text

DATA_PATH = "data/news.csv"   # ← change if your file is named differently


# ─── Load ─────────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  DATASET HEALTH REPORT")
print("="*55)

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"\n❌  File not found: {DATA_PATH}")
    print("    Check your DATA_PATH at the top of this script.")
    sys.exit(1)

print(f"\n📄  File loaded: {DATA_PATH}")
print(f"    Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"    Columns: {list(df.columns)}")


# ─── 1. Column check ──────────────────────────────────────────────────────────

print("\n── 1. Column check ──────────────────────────────")

text_col  = None
label_col = None

for c in df.columns:
    cl = c.lower().strip()
    if cl in ["text", "content", "article", "body", "title", "headline"]:
        text_col = c
    if cl in ["label", "class", "target", "fake", "category", "type"]:
        label_col = c

if text_col:
    print(f"  ✅  Text column found  : '{text_col}'")
else:
    print(f"  ❌  No text column found!")
    print(f"      Rename your text column to 'text' or 'title'.")

if label_col:
    print(f"  ✅  Label column found : '{label_col}'")
else:
    print(f"  ❌  No label column found!")
    print(f"      Rename your label column to 'label'.")

if not text_col or not label_col:
    print("\n  Fix columns first, then re-run this script.")
    sys.exit(1)


# ─── 2. Missing values ────────────────────────────────────────────────────────

print("\n── 2. Missing values ────────────────────────────")

text_nulls  = df[text_col].isna().sum()
label_nulls = df[label_col].isna().sum()
empty_texts = (df[text_col].astype(str).str.strip() == "").sum()

print(f"  Text  nulls  : {text_nulls:,}")
print(f"  Text  empty  : {empty_texts:,}")
print(f"  Label nulls  : {label_nulls:,}")

if text_nulls + empty_texts > df.shape[0] * 0.05:
    print("  ⚠️   More than 5% of text is missing — this hurts accuracy a lot.")
    print("       Run: df = df.dropna(subset=['text', 'label'])")
elif text_nulls == 0 and empty_texts == 0:
    print("  ✅  No missing text values.")

if label_nulls > 0:
    print(f"  ⚠️   {label_nulls} rows have no label — drop these rows.")


# ─── 3. Label distribution ────────────────────────────────────────────────────

print("\n── 3. Label distribution (class balance) ────────")

label_counts = df[label_col].value_counts()
print(f"\n  {label_counts.to_string()}\n")

# Detect imbalance
total = len(df)
for label, count in label_counts.items():
    pct = count / total * 100
    bar = "█" * int(pct / 2)
    print(f"  {str(label):10s}  {bar:25s}  {count:,} ({pct:.1f}%)")

minority_pct = label_counts.min() / total * 100
if minority_pct < 30:
    print(f"\n  ⚠️   IMBALANCED DATASET — minority class is only {minority_pct:.1f}%")
    print("       Fix: use class_weight='balanced' in LogisticRegression")
    print("       Fix: try oversampling with imbalanced-learn (SMOTE)")
elif minority_pct < 40:
    print(f"\n  ⚠️   Slightly imbalanced ({minority_pct:.1f}%). Use class_weight='balanced'.")
else:
    print(f"\n  ✅  Classes are reasonably balanced ({minority_pct:.1f}% minority).")


# ─── 4. Label format check ────────────────────────────────────────────────────

print("\n── 4. Label format ──────────────────────────────")

unique_labels = df[label_col].unique()
print(f"  Unique labels: {unique_labels}")

valid_label_sets = [
    {0, 1}, {"fake", "real"}, {"FAKE", "REAL"},
    {"Fake", "Real"}, {"0", "1"}, {0.0, 1.0},
    {"true", "false"}, {"TRUE", "FALSE"},
]

label_set = set(df[label_col].dropna().unique())
is_valid = any(label_set == s or label_set.issubset(s) for s in valid_label_sets)

if is_valid:
    print("  ✅  Labels look valid.")
else:
    print(f"  ⚠️   Unexpected label format: {label_set}")
    print("       The code expects 0/1 or 'fake'/'real'.")
    print("       Add a mapping in train.py to convert your labels.")


# ─── 5. Text quality ──────────────────────────────────────────────────────────

print("\n── 5. Text quality ──────────────────────────────")

df["_len"] = df[text_col].astype(str).str.split().str.len()

print(f"  Avg  word count : {df['_len'].mean():.0f}")
print(f"  Min  word count : {df['_len'].min()}")
print(f"  Max  word count : {df['_len'].max()}")

very_short = (df["_len"] < 10).sum()
if very_short > 0:
    print(f"\n  ⚠️   {very_short} articles have fewer than 10 words.")
    print("       Very short texts give the model almost no signal.")
    if very_short > total * 0.1:
        print("       Consider dropping them: df = df[df['_len'] >= 10]")

if df["_len"].mean() < 30:
    print("\n  ⚠️   Average article length is very short.")
    print("       Headlines-only datasets are harder to classify.")
    print("       Try to include full article body if possible.")
elif df["_len"].mean() >= 100:
    print("  ✅  Text length looks good.")


# ─── 6. Duplicate check ───────────────────────────────────────────────────────

print("\n── 6. Duplicate rows ────────────────────────────")

dupes = df[text_col].duplicated().sum()
if dupes > 0:
    print(f"  ⚠️   {dupes:,} duplicate texts found!")
    print("       Fix: df = df.drop_duplicates(subset=['{text_col}'])")
    print("       Duplicates in both train and test = data leakage!")
else:
    print("  ✅  No duplicate texts.")


# ─── 7. Dataset size verdict ──────────────────────────────────────────────────

print("\n── 7. Dataset size verdict ──────────────────────")

if total < 500:
    print(f"  ❌  Only {total:,} rows — this is too small for reliable ML.")
    print("       Minimum recommended: 2,000+ rows per class.")
    print("       SOLUTION: Merge with the Kaggle dataset (free, 44K articles).")
    print("       Download: kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
elif total < 2000:
    print(f"  ⚠️   {total:,} rows — small dataset. Accuracy will be limited.")
    print("       Recommended: augment with Kaggle data (see below).")
elif total < 5000:
    print(f"  ✅  {total:,} rows — acceptable. Could be better with more data.")
else:
    print(f"  ✅  {total:,} rows — good dataset size.")


# ─── 8. Language/encoding check ───────────────────────────────────────────────

print("\n── 8. Sample text preview ───────────────────────")

sample = df[[text_col, label_col]].dropna().sample(min(3, total), random_state=42)
for _, row in sample.iterrows():
    text_preview = str(row[text_col])[:120].replace("\n", " ")
    print(f"\n  Label: {row[label_col]}")
    print(f"  Text : {text_preview}...")


# ─── 9. Quick preprocessing test ─────────────────────────────────────────────

print("\n── 9. Preprocessing check ───────────────────────")

sample_text = str(df[text_col].dropna().iloc[0])
try:
    cleaned = preprocess_text(sample_text)
    token_count = len(cleaned.split())
    print(f"  Original  ({len(sample_text.split())} words): {sample_text[:80]}...")
    print(f"  Processed ({token_count} tokens): {cleaned[:80]}...")
    if token_count < 5:
        print("  ⚠️   Preprocessing is removing too many words!")
        print("       Check your stopwords list and lemmatiser.")
    else:
        print("  ✅  Preprocessing looks fine.")
except Exception as e:
    print(f"  ❌  Preprocessing error: {e}")


# ─── Summary ──────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  SUMMARY — WHAT TO FIX")
print("="*55)

issues = []
if total < 2000:
    issues.append(f"🔴  Dataset too small ({total:,} rows) → merge with Kaggle data")
if minority_pct < 35:
    issues.append(f"🔴  Class imbalance ({minority_pct:.0f}% minority) → use class_weight='balanced'")
if dupes > 0:
    issues.append(f"🟡  {dupes} duplicate rows → drop_duplicates()")
if text_nulls + empty_texts > 0:
    issues.append(f"🟡  {text_nulls+empty_texts} missing texts → dropna()")
if very_short > total * 0.1:
    issues.append(f"🟡  {very_short} very short articles → filter < 10 words")

if issues:
    for issue in issues:
        print(f"  {issue}")
else:
    print("  ✅  Dataset looks healthy — problem is likely in the model/features.")
    print("      Run the accuracy improvement guide for model-level fixes.")

print("\n  Run this after fixing: python src/train.py")
print("="*55 + "\n")