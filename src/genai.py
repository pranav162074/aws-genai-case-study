"""
Generative AI explanation.

Prompt engineering (Generative AI Foundations course):
  * Role / persona          -> "You are a careful fact-checking assistant"
  * Context                 -> the ML verdict, confidence, key words, style signals, entities
  * Instructions            -> explain the verdict, list red flags / credibility signals
  * Constraints (guardrails)-> no invented facts, no certainty claims, neutral tone
  * Output format           -> fixed headings so the web page can display it

If Amazon Bedrock is not reachable, a template-based generator produces the
same structure offline, so the demo always works.
"""

from src.aws_services import BEDROCK_MODEL_ID, bedrock_generate

SYSTEM_PROMPT = (
    "You are a careful, neutral fact-checking assistant that explains the output of a "
    "machine-learning fake-news classifier to students. You never claim to know whether "
    "the events in the article really happened, you never invent facts, sources or URLs, "
    "and you avoid political opinions. Keep the answer under 180 words."
)

DISCLAIMER = ("This is an automated estimate from a model trained on 2016–2017 news. "
              "Always verify important claims with trusted sources.")


def build_prompt(title, text, pred, style, entities):
    fake_w = ", ".join(w for w, _ in pred["fake_words"][:6]) or "none"
    real_w = ", ".join(w for w, _ in pred["real_words"][:6]) or "none"
    ents = ", ".join(e["text"] for e in entities[:8]) or "none"
    return f"""Explain this classifier result to a reader.

<article>
Headline: {title or '(no headline)'}
Body (first 1500 characters): {text[:1500]}
</article>

<classifier_output>
Verdict: {pred['label']} (confidence {pred['confidence']}%)
Words that pushed towards FAKE: {fake_w}
Words that pushed towards REAL: {real_w}
Exclamation marks: {style['exclamations']}, ALL-CAPS words: {style['caps_words']}
Clickbait phrases: {', '.join(style['clickbait_phrases']) or 'none'}
Source-attribution phrases (said, according to ...): {style['attribution_count']}
Entities mentioned: {ents}
</classifier_output>

Write the answer in exactly this format:
Summary: <one sentence explaining the verdict in plain English>
Red flags:
- <up to 3 bullet points, or "- None found">
Credibility signals:
- <up to 3 bullet points, or "- None found">
How to verify:
- <2 practical steps the reader can take>
"""


def template_explanation(title, pred, style, entities):
    """Offline generator: builds the same structured explanation from rules."""
    label, conf = pred["label"], pred["confidence"]
    strength = "strongly" if conf >= 90 else "moderately" if conf >= 70 else "weakly"
    red, good = [], []

    if style["clickbait_phrases"]:
        red.append("Uses sensational / clickbait wording: " + ", ".join(f'"{c}"' for c in style["clickbait_phrases"][:4]) + ".")
    if style["exclamations"] >= 2:
        red.append(f"Contains {style['exclamations']} exclamation marks, typical of emotional rather than factual writing.")
    if style["caps_words"] >= 3 or style["title_caps"]:
        red.append(f"Uses ALL-CAPS words ({style['caps_words']}) to grab attention.")
    if pred["fake_words"]:
        red.append("Vocabulary common in fake articles of the training data: " + ", ".join(w for w, _ in pred["fake_words"][:5]) + ".")
    if style["attribution_count"] == 0:
        red.append("No clear source attribution (no \"said\", \"according to\", officials, statements).")

    if style["attribution_count"] >= 2:
        good.append(f"Attributes information to sources {style['attribution_count']} times (\"said\", \"according to\", officials).")
    if pred["real_words"]:
        good.append("Words the model associates with real news reports: " + ", ".join(w for w, _ in pred["real_words"][:5]) + ".")
    dated = [e["text"] for e in entities if e["type"] in ("DATE", "QUANTITY")]
    if dated:
        good.append("Gives concrete details such as dates or figures (" + ", ".join(dated[:3]) + ").")
    if not style["clickbait_phrases"] and style["exclamations"] == 0:
        good.append("Neutral tone: no clickbait phrases or exclamation marks.")

    red = red[:3] or ["None found"]
    good = good[:3] or ["None found"]
    names = [e["text"] for e in entities if e["type"] not in ("DATE", "QUANTITY")][:2]
    verify = [
        "Search the headline on trusted news sites (e.g. Reuters, AP, BBC, PTI) to see if they report the same event.",
        ("Check official statements from " + " / ".join(names) + ".") if names
        else "Look for the original source, author and publication date of the article.",
    ]

    summary = (f"The model {strength} classifies this article as {label} ({conf}% confidence), "
               f"mainly because of its {'sensational style and wording' if label == 'FAKE' else 'factual, source-based reporting style'}.")
    lines = [f"Summary: {summary}", "Red flags:"] + [f"- {r}" for r in red] + \
            ["Credibility signals:"] + [f"- {g}" for g in good] + \
            ["How to verify:"] + [f"- {v}" for v in verify]
    return "\n".join(lines)


def parse_sections(text):
    """Turn the structured answer into a dict for the web page."""
    sections = {"summary": "", "red_flags": [], "credibility": [], "verify": []}
    current = None
    for line in text.splitlines():
        s = line.strip().strip("*").strip()
        low = s.lower()
        if low.startswith("summary"):
            sections["summary"] = s.split(":", 1)[1].strip() if ":" in s else ""
            current = "summary"
        elif low.startswith("red flags"):
            current = "red_flags"
        elif low.startswith("credibility"):
            current = "credibility"
        elif low.startswith("how to verify"):
            current = "verify"
        elif s.startswith(("-", "•")) and current in ("red_flags", "credibility", "verify"):
            sections[current].append(s.lstrip("-• ").strip())
        elif s and current == "summary":
            sections["summary"] += " " + s
    return sections


def explain(title, text, pred, style, entities):
    prompt = build_prompt(title, text, pred, style, entities)
    answer = bedrock_generate(SYSTEM_PROMPT, prompt)
    if answer:
        source = f"Amazon Bedrock ({BEDROCK_MODEL_ID})"
    else:
        answer = template_explanation(title, pred, style, entities)
        source = "Offline template generator (Bedrock not connected)"
    sections = parse_sections(answer)
    if not sections["summary"]:  # guardrail: unexpected format -> show raw text
        sections["summary"] = answer
    return {"source": source, "sections": sections, "raw": answer,
            "prompt": prompt, "disclaimer": DISCLAIMER}
