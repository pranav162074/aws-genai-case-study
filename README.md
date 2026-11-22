# TruthLens – Fake News Detector
**AWS Gen AI Virtual Internship – Case Study**
G. Sai Pranav Reddy · 324103311015 · Dept. of Information Technology, GVPCE (A)

A web app that takes a news headline and article and:
1. **predicts REAL or FAKE** with a confidence score (Machine Learning),
2. **shows why**: the words and writing-style signals behind the decision (NLP),
3. **explains the result in plain English**, with steps to verify it, using a foundation model on **Amazon Bedrock** (Generative AI).

It works **offline** too. If AWS credentials are not set, Amazon Comprehend and Amazon Bedrock are replaced by built-in fallbacks, so the demo never breaks.

---

## How to run (Windows)

**Easiest:** double-click `run.bat`.
The first time, it creates a virtual environment, installs the packages, trains the model (~1 min) and opens the browser at **http://127.0.0.1:5000**.

**Manually (VS Code terminal):**
```bat
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
python app.py
```
Then open http://127.0.0.1:5000. Press `Ctrl + C` in the terminal to stop.

Requires Python 3.10 or newer.

### Optional: connect AWS (Comprehend + Bedrock)
1. Copy `.env.example` to `.env`.
2. Paste your AWS keys into it. In the AWS Academy Learner Lab: **Start Lab → AWS Details → AWS CLI: Show**.
3. Make sure the Bedrock model in `BEDROCK_MODEL_ID` is enabled in your account/region.
4. Restart `app.py`. The badge at the top right changes from **Offline mode** to **AWS connected**.

---

## Project structure
```
aws-genai-case-study/
├── app.py               Flask web server + /api/analyze endpoint
├── train.py             Preprocess → TF-IDF → train & compare 4 models → save best
├── src/
│   ├── preprocess.py    NLP cleaning: leakage removal, contractions, stop words
│   ├── analyzer.py      Prediction, word contributions, style signals, offline NER
│   ├── aws_services.py  Amazon Comprehend + Amazon Bedrock (with fallbacks)
│   └── genai.py         Prompt engineering + offline template generator
├── templates/           Web pages (detector + how-it-works/results)
├── static/              CSS, JavaScript, generated charts
├── data/                Labelled news dataset (train / validation / test)
├── model/               Trained model + metrics (created by train.py)
├── requirements.txt
├── run.bat              One-click setup & run
└── .env.example         AWS settings template
```

## Dataset
`GonzaloA/fake_news` (Hugging Face): about 40,600 English news articles labelled **real (1)** or **fake (0)**, already split into
train (24,353) / validation (8,117) / test (8,117).
Real articles come mainly from Reuters; fake ones from sites flagged by fact-checkers (2016–2017).

**Data-leakage fix:** almost every real article starts with *"CITY (Reuters) –"*. A model could simply learn "Reuters = real".
`preprocess.py` removes these datelines, image credits and similar source giveaways, so the model learns from the writing itself.

## Pipeline
| Step | What happens | Course |
|---|---|---|
| 1. Input | Headline + article from the web page; guardrail rejects texts under 20 words | Gen AI (Responsible AI) |
| 2. Preprocessing | Leakage removal, lower-case, URL/handle removal, contraction normalising, stop-word removal | ML for NLP |
| 3. Feature extraction | TF-IDF, unigrams + bigrams, 60,000 features | ML for NLP |
| 4. Model training | Logistic Regression, Naive Bayes, Linear SVM and SGD Classifier compared on the validation set; best one calibrated for probabilities | ML Foundations |
| 5. Evaluation | Accuracy, precision, recall, F1 and a confusion matrix on the unseen test set | ML Foundations |
| 6. NLP analysis | Amazon Comprehend: language, entities, sentiment (offline: rule-based) | ML for NLP |
| 7. Explanation | Amazon Bedrock Converse API with an engineered prompt (offline: template) | Gen AI Foundations |

## Results (test set, 8,116 unseen articles)
| Metric | Score |
|---|---|
| Accuracy | ~97.4 % |
| F1 (fake) | ~97.2 % |
| F1 (real) | ~97.5 % |

Exact numbers are printed by `train.py` and shown on the **How it works & results** page.

## Demo script (for the review)
1. Run `run.bat` and show the **Detector** page.
2. Click **Load a FAKE sample** → **Analyze**. Point out the verdict, the confidence bar, *"Actual label … ✓ correct"*, the red words and the Gen AI explanation.
3. Click **Load a REAL sample** → **Analyze**. Point out the green words and the source attributions ("said", officials).
4. Type your own clickbait headline, e.g. *"BREAKING: SHOCKING video EXPOSES what the government is hiding!!!"*, with a short body, and show how the style signals react.
5. Open **View the prompt sent to the foundation model** to explain prompt engineering.
6. Open the **How it works & results** page: course mapping, model comparison, confusion matrix.

## Limitations
- The model judges writing **style and vocabulary**, not whether the facts are true.
- It was trained on 2016–2017, mostly US political news, so it is less reliable on other topics, very short texts or other languages.
- Treat it as a decision-support tool; verify important claims with trusted sources.
