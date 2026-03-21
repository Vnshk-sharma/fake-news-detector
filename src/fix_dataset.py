"""
Dataset Fixer + Kaggle Merger
Fixes common dataset problems and optionally merges your data
with the Kaggle fake news dataset to boost accuracy.

Usage:
    python src/fix_dataset.py
"""

import pandas as pd
import numpy as np
import os, sys

YOUR_DATA    = "data/news.csv"        # ← your current dataset
KAGGLE_FAKE  = "data/Fake.csv"        # ← download from Kaggle (optional)
KAGGLE_REAL  = "data/True.csv"        # ← download from Kaggle (optional)
OUTPUT_PATH  = "data/news_fixed.csv"  # ← cleaned output file


# ─── Step 1: Load your dataset ────────────────────────────────────────────────

print("\n[LOAD] Reading your dataset...")
df = pd.read_csv(YOUR_DATA)
print(f"       Loaded {len(df):,} rows")

original_size = len(df)


# ─── Step 2: Standardise column names ─────────────────────────────────────────

print("\n[FIX]  Standardising columns...")
df.columns = [c.lower().strip() for c in df.columns]

# Auto-detect and rename text column
for candidate in ["content", "article", "body", "headline", "title"]:
    if candidate in df.columns and "text" not in df.columns:
        df = df.rename(columns={candidate: "text"})
        print(f"       Renamed '{candidate}' → 'text'")

# Auto-detect and rename label column
for candidate in ["class", "target", "category", "type", "fake"]:
    if candidate in df.columns and "label" not in df.columns:
        df = df.rename(columns={candidate: "label"})
        print(f"       Renamed '{candidate}' → 'label'")

if "text" not in df.columns:
    print("❌  Could not find a text column. Rename it to 'text' manually.")
    sys.exit(1)
if "label" not in df.columns:
    print("❌  Could not find a label column. Rename it to 'label' manually.")
    sys.exit(1)


# ─── Step 3: Standardise label values ─────────────────────────────────────────

print("\n[FIX]  Standardising labels...")
print(f"       Before: {df['label'].unique()}")

label_map = {
    # Text variants → 0=real, 1=fake
    "real": 0, "true": 0, "TRUE": 0, "Real": 0, "REAL": 0,
    "fake": 1, "false": 1, "FALSE": 1, "Fake": 1, "FAKE": 1,
    # Numeric variants
    0: 0, 1: 1, 0.0: 0, 1.0: 1,
    "0": 0, "1": 1,
}

# Try to map; if all values are already 0/1 integers, skip
if set(df["label"].dropna().unique()).issubset({0, 1}):
    print("       Labels already 0/1 — no change needed.")
else:
    before = df["label"].isna().sum()
    df["label"] = df["label"].map(label_map)
    after = df["label"].isna().sum()
    unmapped = after - before
    if unmapped > 0:
        print(f"       ⚠️  {unmapped} rows have unrecognised labels → will be dropped")
    print(f"       After:  {df['label'].dropna().unique()}")


# ─── Step 4: Drop bad rows ────────────────────────────────────────────────────

print("\n[FIX]  Dropping bad rows...")

before = len(df)
df = df.dropna(subset=["text", "label"])
df = df[df["text"].astype(str).str.strip() != ""]
df["label"] = df["label"].astype(int)
after = len(df)
print(f"       Dropped {before - after:,} rows with missing text or label")


# ─── Step 5: Drop very short texts ───────────────────────────────────────────

print("\n[FIX]  Filtering very short texts...")
df["_word_count"] = df["text"].astype(str).str.split().str.len()
before = len(df)
df = df[df["_word_count"] >= 8]   # minimum 8 words
df = df.drop(columns=["_word_count"])
after = len(df)
print(f"       Dropped {before - after:,} articles with < 8 words")


# ─── Step 6: Remove duplicates ────────────────────────────────────────────────

print("\n[FIX]  Removing duplicates...")
before = len(df)
df = df.drop_duplicates(subset=["text"])
after = len(df)
print(f"       Removed {before - after:,} duplicate articles")


# ─── Step 7: Merge with Kaggle dataset (optional but STRONGLY recommended) ────

kaggle_available = os.path.exists(KAGGLE_FAKE) and os.path.exists(KAGGLE_REAL)

if kaggle_available:
    print("\n[MERGE] Kaggle dataset found — merging...")

    fake_df = pd.read_csv(KAGGLE_FAKE)
    real_df = pd.read_csv(KAGGLE_REAL)
    fake_df["label"] = 1
    real_df["label"] = 0

    # Standardise Kaggle columns
    for kdf in [fake_df, real_df]:
        kdf.columns = [c.lower().strip() for c in kdf.columns]
        if "title" in kdf.columns and "text" in kdf.columns:
            kdf["text"] = kdf["title"].fillna("") + " " + kdf["text"].fillna("")
        elif "title" in kdf.columns:
            kdf["text"] = kdf["title"].fillna("")

    kaggle_df = pd.concat(
        [fake_df[["text","label"]], real_df[["text","label"]]],
        ignore_index=True
    )
    kaggle_df = kaggle_df.dropna(subset=["text","label"])

    # Combine with your dataset (yours gets double weight)
    combined = pd.concat([
        df[["text","label"]],
        df[["text","label"]],   # your data repeated → gives it more weight
        kaggle_df[["text","label"]],
    ], ignore_index=True)

    combined = combined.drop_duplicates(subset=["text"])
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"       Your data   : {len(df):,} rows")
    print(f"       Kaggle data : {len(kaggle_df):,} rows")
    print(f"       Combined    : {len(combined):,} rows")
    df = combined

else:
    print("\n[SKIP] Kaggle files not found in data/ folder.")
    print("       To boost accuracy significantly, download:")
    print("       kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    print("       Place Fake.csv and True.csv in your data/ folder and re-run.")
    df = df[["text","label"]]


# ─── Step 8: Final shuffle and save ───────────────────────────────────────────

print("\n[SAVE] Finalising dataset...")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Class balance report
counts = df["label"].value_counts()
print(f"\n  Final class balance:")
print(f"    Real (0): {counts.get(0,0):,} ({counts.get(0,0)/len(df)*100:.1f}%)")
print(f"    Fake (1): {counts.get(1,0):,} ({counts.get(1,0)/len(df)*100:.1f}%)")
print(f"    Total   : {len(df):,} rows")

df.to_csv(OUTPUT_PATH, index=False)
print(f"\n  ✅  Saved to: {OUTPUT_PATH}")
print(f"\n  Next step: update DATA_PATH in src/train.py to '{OUTPUT_PATH}'")
print("  Then run:  python src/train.py\n")
