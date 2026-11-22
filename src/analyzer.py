"""
Prediction + explanation layer.

Given a headline and article body it returns:
  * the ML verdict (FAKE / REAL) with a confidence score
  * which words pushed the model towards each side (explainability)
  * writing-style signals (NLP features such as clickbait phrases, ALL-CAPS, '!')
  * a locally extracted list of entities (used when Amazon Comprehend is unavailable)
"""

import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from src.preprocess import clean_text, combine

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "fake_news_model.joblib"

CLICKBAIT = [
    "you won't believe", "you wont believe", "shocking", "breaking", "watch",
    "exposed", "must see", "must-see", "unbelievable", "destroys", "slams",
    "epic", "bombshell", "what happened next", "the truth about", "share this",
    "mainstream media", "wake up", "hoax", "secret", "they don't want you to know",
]
ATTRIBUTION = [
    "said", "according to", "told reporters", "spokesman", "spokeswoman",
    "statement", "officials", "announced", "confirmed", "reported",
]


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run `python train.py` first (run.bat does this automatically)."
        )
    return joblib.load(MODEL_PATH)


def predict(title: str, text: str) -> dict:
    bundle = load_model()
    vec, model, coefs = bundle["vectorizer"], bundle["model"], bundle["coefs"]

    raw = combine(title, text)
    cleaned = clean_text(raw)
    X = vec.transform([cleaned])
    proba = model.predict_proba(X)[0]           # [P(fake), P(real)]
    p_real = float(proba[1])
    label = "REAL" if p_real >= 0.5 else "FAKE"
    confidence = p_real if label == "REAL" else 1 - p_real

    # Contribution of each word = its TF-IDF value x the model's weight for it
    row = X.tocoo()
    vocab = vec.get_feature_names_out()
    contrib = [(vocab[j], float(v * coefs[j])) for j, v in zip(row.col, row.data)]
    contrib.sort(key=lambda t: t[1])
    fake_words = [(w, round(-c, 4)) for w, c in contrib[:10] if c < 0]
    real_words = [(w, round(c, 4)) for w, c in contrib[::-1][:10] if c > 0]

    return {
        "label": label,
        # Responsible AI: never display absolute certainty
        "confidence": min(round(confidence * 100, 1), 99.9),
        "prob_fake": min(max(round((1 - p_real) * 100, 1), 0.1), 99.9),
        "prob_real": min(max(round(p_real * 100, 1), 0.1), 99.9),
        "model_name": bundle["model_name"],
        "fake_words": fake_words,
        "real_words": real_words,
        "highlight": highlight(raw, fake_words, real_words),
        "cleaned_preview": cleaned[:400],
        "token_count": len(cleaned.split()),
    }


def highlight(raw: str, fake_words, real_words):
    """Split the original text into segments tagged 'fake' / 'real' / ''."""
    fake = {w for w, _ in fake_words if " " not in w}
    real = {w for w, _ in real_words if " " not in w}
    out = []
    for tok in re.findall(r"[A-Za-z’']+|[^A-Za-z’']+", raw):
        key = re.sub(r"[’']", "", tok.lower())
        tag = "fake" if key in fake else "real" if key in real else ""
        if out and out[-1][1] == tag and tag == "":
            out[-1] = (out[-1][0] + tok, "")
        else:
            out.append((tok, tag))
    return out


def style_signals(title: str, text: str) -> dict:
    full = f"{title}\n{text}"
    low = full.lower()
    words = re.findall(r"[A-Za-z']+", full)
    caps = [w for w in words if len(w) > 2 and w.isupper()]
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    clickbait = sorted({c for c in CLICKBAIT if c in low})
    attribution = sum(len(re.findall(r"\b" + re.escape(a) + r"\b", low)) for a in ATTRIBUTION)
    return {
        "word_count": len(words),
        "exclamations": full.count("!"),
        "questions": full.count("?"),
        "caps_words": len(caps),
        "caps_ratio": round(100 * len(caps) / max(len(words), 1), 1),
        "clickbait_phrases": clickbait,
        "attribution_count": attribution,
        "avg_sentence_len": round(len(words) / max(len(sentences), 1), 1),
        "title_caps": bool(title) and sum(1 for w in title.split() if w.isupper() and len(w) > 2) >= 2,
    }


_MONTHS = r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
_DATE = re.compile(rf"\b{_MONTHS}\s+\d{{1,2}}(,\s*\d{{4}})?|\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b|\b(19|20)\d{{2}}\b")
_QUANTITY = re.compile(r"\$\s?\d[\d,.]*\s?(million|billion|trillion)?|\b\d[\d,.]*\s?(%|percent|million|billion|people|dollars)\b", re.I)
_PROPER = re.compile(r"\b[A-Z][a-z]+(?:\s+(?:of\s+|the\s+)?[A-Z][a-z]+){1,3}\b")
_SKIP_START = {"The", "This", "That", "These", "Those", "But", "And", "In", "On", "At", "It", "He", "She", "We", "They", "If", "When", "So", "As", "Now", "Then", "Also", "Why", "What", "How", "Fancy", "Even"}


def local_entities(text: str, limit: int = 15):
    """Lightweight rule-based entity extraction (offline fallback for Amazon Comprehend)."""
    found = []
    seen = set()

    def add(t, kind):
        t = t.strip(" ,.")
        if t and t.lower() not in seen:
            seen.add(t.lower())
            found.append({"text": t, "type": kind})

    for m in _DATE.finditer(text):
        add(m.group(0), "DATE")
    for m in _QUANTITY.finditer(text):
        add(m.group(0), "QUANTITY")
    for m in _PROPER.finditer(text):
        phrase = m.group(0)
        first = phrase.split()[0]
        if first in _SKIP_START:
            phrase = phrase[len(first):].strip()
        if len(phrase.split()) >= 2:
            add(phrase, "NAME / PLACE / ORG")
    return found[:limit]


def guess_language(text: str) -> str:
    common = {"the", "and", "of", "to", "in", "is", "that", "for", "it", "was", "on", "with"}
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return "unknown"
    share = sum(w in common for w in words) / len(words)
    return "en" if share > 0.08 else "unknown (possibly not English)"


def top_global_words(n=10):
    bundle = load_model()
    coefs, vocab = bundle["coefs"], bundle["vectorizer"].get_feature_names_out()
    order = np.argsort(coefs)
    return [vocab[i] for i in order[:n]], [vocab[i] for i in order[::-1][:n]]
