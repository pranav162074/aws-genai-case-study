"""
Fake News Detector — web app (Flask).

Run:  python app.py      then open http://127.0.0.1:5000
"""

import json
import random
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.analyzer import guess_language, load_model, local_entities, predict, style_signals
from src.aws_services import aws_status, comprehend_analyze
from src.genai import explain

ROOT = Path(__file__).parent
app = Flask(__name__)

MIN_WORDS, MAX_CHARS = 20, 20000


@lru_cache(maxsize=1)
def sample_pool():
    df = pd.read_csv(ROOT / "data" / "test.csv.gz").fillna("")
    df = df[(df["text"].str.len().between(500, 2200)) & (df["title"].str.len() > 20)]
    return {
        "FAKE": df[df["label"] == 0].sample(200, random_state=1)[["title", "text"]].to_dict("records"),
        "REAL": df[df["label"] == 1].sample(200, random_state=1)[["title", "text"]].to_dict("records"),
    }


@lru_cache(maxsize=1)
def metrics():
    p = ROOT / "model" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


@app.route("/")
def index():
    return render_template("index.html", status=aws_status(), metrics=metrics())


@app.route("/about")
def about():
    return render_template("about.html", status=aws_status(), metrics=metrics())


@app.route("/api/sample")
def api_sample():
    kind = request.args.get("kind", "FAKE").upper()
    item = random.choice(sample_pool()["REAL" if kind == "REAL" else "FAKE"])
    return jsonify({"title": item["title"].strip(), "text": item["text"].strip(), "actual": kind})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(force=True) or {}
    title = (data.get("title") or "").strip()[:500]
    text = (data.get("text") or "").strip()[:MAX_CHARS]

    # Guardrail: reject inputs that are too short to judge
    if len((title + " " + text).split()) < MIN_WORDS:
        return jsonify({"error": f"Please enter at least {MIN_WORDS} words (headline + article)."}), 400

    t0 = time.time()
    pred = predict(title, text)                                   # ML + NLP
    style = style_signals(title, text)                            # NLP style features
    aws_nlp = comprehend_analyze(f"{title}. {text}")              # Amazon Comprehend
    if aws_nlp:
        nlp = {**aws_nlp, "source": "Amazon Comprehend"}
    else:
        nlp = {"language": guess_language(text), "entities": local_entities(text),  # body only: headlines are Title Case
               "sentiment": None, "sentiment_scores": None,
               "source": "Offline rule-based NLP (Comprehend not connected)"}
    gen = explain(title, text, pred, style, nlp["entities"])      # Generative AI

    pred["highlight"] = [[t, tag] for t, tag in pred["highlight"]]
    return jsonify({"prediction": pred, "style": style, "nlp": nlp, "genai": gen,
                    "elapsed_ms": int((time.time() - t0) * 1000)})


@app.route("/api/status")
def api_status():
    return jsonify(aws_status())


if __name__ == "__main__":
    load_model()  # fail fast with a clear message if train.py was not run
    print("\n  Fake News Detector running at  http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
