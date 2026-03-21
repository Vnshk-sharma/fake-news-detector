"""
Fake News Detection System - Text Preprocessing
Cleans raw news text: lowercase, remove noise, tokenise, lemmatise, remove stopwords.
"""

import re
import string
import nltk

# Download NLTK resources on first run (silent if already present)
for resource in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt"
                       else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ─── Constants ────────────────────────────────────────────────────────────────

STOP_WORDS = set(stopwords.words("english"))

# Keep negation words — they matter for sentiment ("not true", "no evidence")
NEGATION = {"no", "not", "nor", "never", "neither", "none"}
STOP_WORDS -= NEGATION

LEMMATISER = WordNetLemmatizer()

# Patterns to strip before tokenising
URL_PATTERN     = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN   = re.compile(r"\S+@\S+\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
NUMBER_PATTERN  = re.compile(r"\b\d+\b")


# ─── Pipeline Steps ───────────────────────────────────────────────────────────

def remove_noise(text: str) -> str:
    """Strip URLs, emails, mentions, hashtags, and standalone numbers."""
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(" ", text)
    text = NUMBER_PATTERN.sub(" ", text)
    return text


def remove_punctuation(text: str) -> str:
    """Remove all punctuation characters."""
    return text.translate(str.maketrans("", "", string.punctuation))


def tokenise_and_filter(text: str) -> list[str]:
    """
    Tokenise, remove stopwords, and filter very short tokens.
    word_tokenize handles contractions better than split() — 
    e.g. "don't" → ["do", "n't"] so both parts are considered.
    """
    tokens = word_tokenize(text)
    return [
        token for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    ]


def lemmatise(tokens: list[str]) -> list[str]:
    """
    Lemmatisation converts words to their dictionary base form.
    Unlike stemming, it produces real words: "running"→"run", "better"→"good".
    We try both verb and noun forms to maximise coverage.
    """
    return [
        LEMMATISER.lemmatize(LEMMATISER.lemmatize(token, pos="v"), pos="n")
        for token in tokens
    ]


# ─── Public API ───────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline.  Returns a clean, space-joined string
    ready for TF-IDF vectorisation.

    Steps:
      1. Lowercase
      2. Remove noise (URLs, emails, etc.)
      3. Remove punctuation
      4. Tokenise + remove stopwords
      5. Lemmatise
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = remove_noise(text)
    text = remove_punctuation(text)
    tokens = tokenise_and_filter(text)
    tokens = lemmatise(tokens)

    return " ".join(tokens)


# ─── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = (
        "BREAKING: Scientists say COVID-19 vaccines NOT effective! "
        "Visit http://fakenews.com for the truth. #BigPharma @researcher"
    )
    print("Original :", sample)
    print("Processed:", preprocess_text(sample))
