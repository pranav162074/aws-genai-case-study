"""
AWS service layer.

  * Amazon Comprehend  -> language detection, entity recognition (NER), sentiment
  * Amazon Bedrock     -> Generative AI explanation (Converse API)

Both are optional. If no AWS credentials are configured (or a call fails),
the app falls back to offline logic so the demo always works.

Credentials are read the normal boto3 way: environment variables,
~/.aws/credentials, or the .env file in the project folder
(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_REGION).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v and k not in os.environ:
            os.environ[k] = v


_load_dotenv()

REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
USE_AWS = os.environ.get("USE_AWS", "auto").lower()  # auto | off

try:
    import boto3
    from botocore.config import Config

    _CFG = Config(connect_timeout=4, read_timeout=30, retries={"max_attempts": 1})
except ImportError:  # boto3 not installed -> offline only
    boto3 = None


def _client(service):
    if boto3 is None or USE_AWS == "off":
        return None
    session = boto3.Session(region_name=REGION)
    if session.get_credentials() is None:
        return None
    return session.client(service, config=_CFG)


def aws_status():
    if boto3 is None:
        return {"available": False, "reason": "boto3 not installed"}
    if USE_AWS == "off":
        return {"available": False, "reason": "disabled in .env (USE_AWS=off)"}
    if boto3.Session(region_name=REGION).get_credentials() is None:
        return {"available": False, "reason": "no AWS credentials configured"}
    return {"available": True, "region": REGION, "bedrock_model": BEDROCK_MODEL_ID}


def comprehend_analyze(text: str):
    """Returns language, entities and sentiment from Amazon Comprehend, or None."""
    client = _client("comprehend")
    if client is None:
        return None
    sample = text.encode("utf-8")[:4800].decode("utf-8", "ignore")
    try:
        lang = client.detect_dominant_language(Text=sample)["Languages"][0]
        code = lang["LanguageCode"] if lang["LanguageCode"] in ("en", "es", "fr", "de", "it", "pt") else "en"
        ents = client.detect_entities(Text=sample, LanguageCode=code)["Entities"]
        sent = client.detect_sentiment(Text=sample, LanguageCode=code)
        seen, entities = set(), []
        for e in sorted(ents, key=lambda e: -e["Score"]):
            if e["Text"].lower() not in seen and e["Score"] > 0.6:
                seen.add(e["Text"].lower())
                entities.append({"text": e["Text"], "type": e["Type"]})
        return {
            "language": f'{lang["LanguageCode"]} ({lang["Score"]*100:.0f}%)',
            "entities": entities[:15],
            "sentiment": sent["Sentiment"],
            "sentiment_scores": {k: round(v * 100, 1) for k, v in sent["SentimentScore"].items()},
        }
    except Exception as exc:  # noqa: BLE001 - any AWS error -> fallback
        print(f"[Comprehend] falling back to offline NLP: {exc}")
        return None


def bedrock_generate(system: str, prompt: str, max_tokens: int = 500):
    """Calls an Amazon Bedrock foundation model through the Converse API, or returns None."""
    client = _client("bedrock-runtime")
    if client is None:
        return None
    try:
        resp = client.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.3, "topP": 0.9},
        )
        return resp["output"]["message"]["content"][0]["text"].strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[Bedrock] falling back to offline generator: {exc}")
        return None
