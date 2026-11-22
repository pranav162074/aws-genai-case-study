"""
NLP text preprocessing for the Fake News Detector.

Steps (as covered in the ML for NLP course):
  1. Remove "source giveaways" (e.g. "WASHINGTON (Reuters) -") so the model
     learns from the writing itself and not from the news agency's name.
  2. Lower-casing
  3. Normalising contractions, removing URLs, @handles, image credits, numbers and punctuation
  4. Removing stop words
  5. (Tokenisation + TF-IDF vectorisation happen in train.py)
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Phrases that reveal the source of an article instead of its content.
# Leaving them in lets a model "cheat" (data leakage), so we strip them.
_LEAK_PATTERNS = [
    r"^.{0,120}?\(reuters\)\s*[-–—]\s*",        # "WASHINGTON (Reuters) - "
    r"\breuters\b",
    r"featured image[^.\n]*",
    r"image via[^.\n]*",
    r"photo by[^.\n]*",
    r"getty images",
    r"21st century wire",
    r"pic\.twitter\.com/\S+",
]

_URL = re.compile(r"https?://\S+|www\.\S+")
_HANDLE = re.compile(r"@\w+")
_NON_LETTERS = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")
_APOS = re.compile(r"[’'`´]")
# Some sources store "didn't" as "didn t" -> normalise both to "didnt"
_SPLIT_CONTRACTION = re.compile(r"\b([a-z]+) (s|t|re|ve|ll|d|m)\b")

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def remove_leakage(text: str) -> str:
    t = text or ""
    for pat in _LEAK_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)
    return t


def clean_text(text: str) -> str:
    """Full cleaning pipeline used for both training and prediction."""
    t = remove_leakage(text).lower()
    t = _APOS.sub("", t)
    t = _SPLIT_CONTRACTION.sub(r"\1\2", t)
    t = _URL.sub(" ", t)
    t = _HANDLE.sub(" ", t)
    t = _NON_LETTERS.sub(" ", t)
    tokens = [w for w in t.split() if w not in STOP_WORDS and len(w) > 1]
    return _SPACES.sub(" ", " ".join(tokens)).strip()


def combine(title: str, text: str) -> str:
    """Title + body: headlines carry a lot of the signal for fake news."""
    return f"{title or ''}. {text or ''}"
