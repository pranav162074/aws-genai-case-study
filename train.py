"""
Train the Fake News Detector.

Pipeline
  data/*.csv.gz  ->  preprocessing (src/preprocess.py)
                 ->  TF-IDF features (unigrams + bigrams)
                 ->  4 ML classifiers trained and compared
                 ->  best model saved to model/fake_news_model.joblib
                 ->  metrics + charts saved for the web app

Run:  python train.py
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.preprocess import clean_text, combine

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODEL_DIR = ROOT / "model"
PLOTS = ROOT / "static" / "plots"
MODEL_DIR.mkdir(exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

LABELS = {0: "FAKE", 1: "REAL"}  # dataset convention: 0 = fake, 1 = real


def load(split):
    df = pd.read_csv(DATA / f"{split}.csv.gz").fillna("")
    df["clean"] = [clean_text(combine(t, x)) for t, x in zip(df["title"], df["text"])]
    df = df[df["clean"].str.len() > 20]  # drop empty / near-empty articles
    return df


def main():
    t0 = time.time()
    print("STEP 1: Loading and preprocessing data ...")
    train, val, test = load("train"), load("validation"), load("test")
    print(f"  train={len(train)}  validation={len(val)}  test={len(test)}")
    print("  class balance (train):", train["label"].map(LABELS).value_counts().to_dict())

    print("STEP 2: TF-IDF feature extraction ...")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                          max_features=60000, sublinear_tf=True)
    Xtr = vec.fit_transform(train["clean"])
    Xva = vec.transform(val["clean"])
    Xte = vec.transform(test["clean"])
    ytr, yva, yte = train["label"].values, val["label"].values, test["label"].values
    print(f"  vocabulary size = {len(vec.vocabulary_)}")

    print("STEP 3: Training and comparing models ...")
    candidates = {
        "Logistic Regression": LogisticRegression(C=10, max_iter=2000),
        "Naive Bayes": MultinomialNB(alpha=0.05),
        "Linear SVM": LinearSVC(C=0.5),
        "SGD Classifier": SGDClassifier(loss="modified_huber", alpha=2e-5, max_iter=50, random_state=42),
    }
    results, fitted = [], {}
    for name, clf in candidates.items():
        s = time.time()
        clf.fit(Xtr, ytr)
        pv = clf.predict(Xva)
        r = {
            "model": name,
            "val_accuracy": accuracy_score(yva, pv),
            "val_f1": f1_score(yva, pv),
            "train_seconds": round(time.time() - s, 2),
        }
        results.append(r)
        fitted[name] = clf
        print(f"  {name:<20} val_acc={r['val_accuracy']:.4f}  val_f1={r['val_f1']:.4f}")

    best_name = max(results, key=lambda r: r["val_f1"])["model"]
    print(f"STEP 4: Best model on validation set -> {best_name}")

    # Calibrate so the app can show a confidence percentage for every model type.
    best = candidates[best_name].__class__(**candidates[best_name].get_params())
    final = CalibratedClassifierCV(best, cv=3, method="sigmoid")
    final.fit(Xtr, ytr)

    print("STEP 5: Evaluating on the unseen test set ...")
    pt = final.predict(Xte)
    test_metrics = {
        "accuracy": accuracy_score(yte, pt),
        "precision_fake": precision_score(yte, pt, pos_label=0),
        "recall_fake": recall_score(yte, pt, pos_label=0),
        "f1_fake": f1_score(yte, pt, pos_label=0),
        "precision_real": precision_score(yte, pt, pos_label=1),
        "recall_real": recall_score(yte, pt, pos_label=1),
        "f1_real": f1_score(yte, pt, pos_label=1),
    }
    cm = confusion_matrix(yte, pt, labels=[0, 1])
    for k, v in test_metrics.items():
        print(f"  {k:<16} {v:.4f}")

    # Word weights for explanations (average over calibrated folds)
    def weights(est):
        if hasattr(est, "coef_"):
            return est.coef_[0]
        return est.feature_log_prob_[1] - est.feature_log_prob_[0]  # Naive Bayes

    coefs = np.mean([weights(c.estimator) for c in final.calibrated_classifiers_], axis=0)

    vocab = np.array(vec.get_feature_names_out())
    order = np.argsort(coefs)
    top_fake = [(vocab[i], float(coefs[i])) for i in order[:20]]
    top_real = [(vocab[i], float(coefs[i])) for i in order[::-1][:20]]

    joblib.dump({"vectorizer": vec, "model": final, "coefs": coefs,
                 "model_name": best_name, "labels": LABELS},
                MODEL_DIR / "fake_news_model.joblib", compress=3)

    metrics = {
        "best_model": best_name,
        "comparison": results,
        "test": test_metrics,
        "confusion_matrix": cm.tolist(),
        "sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "vocabulary": len(vec.vocabulary_),
        "top_fake_words": top_fake,
        "top_real_words": top_real,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("STEP 6: Saving charts ...")
    make_plots(results, cm, top_fake, top_real)
    print(f"Done in {time.time() - t0:.1f}s. Model saved to model/fake_news_model.joblib")


def make_plots(results, cm, top_fake, top_real):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    # Model comparison
    fig, ax = plt.subplots(figsize=(8, 4.6))
    names = [r["model"] for r in results]
    acc = [r["val_accuracy"] * 100 for r in results]
    f1 = [r["val_f1"] * 100 for r in results]
    x = np.arange(len(names))
    ax.bar(x - 0.2, acc, 0.4, label="Accuracy", color="#2563eb")
    ax.bar(x + 0.2, f1, 0.4, label="F1 score", color="#f59e0b")
    for i, (a, f) in enumerate(zip(acc, f1)):
        ax.text(i - 0.2, a + 0.3, f"{a:.1f}", ha="center", fontsize=9)
        ax.text(i + 0.2, f + 0.3, f"{f:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylim(min(acc + f1) - 5, 100)
    ax.set_ylabel("Validation score (%)")
    ax.set_title("Model comparison (validation set)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS / "model_comparison.png", dpi=150)
    plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["FAKE", "REAL"])
    ax.set_yticks([0, 1], ["FAKE", "REAL"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(PLOTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Most influential words
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, words, color, title in [
        (axes[0], top_fake[:12], "#dc2626", "Words pushing towards FAKE"),
        (axes[1], top_real[:12], "#16a34a", "Words pushing towards REAL"),
    ]:
        w = [a for a, _ in words][::-1]
        v = [abs(b) for _, b in words][::-1]
        ax.barh(w, v, color=color)
        ax.set_title(title)
        ax.set_xlabel("Model weight")
    fig.tight_layout()
    fig.savefig(PLOTS / "top_words.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
