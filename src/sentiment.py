"""
Step 14 — Sentiment analysis. Classical lexicon-based, no ML/LLM needed.
"""

import re

POSITIVE_WORDS = {
    "great", "good", "excellent", "thanks", "thank", "helpful", "appreciate",
    "love", "awesome", "perfect", "happy", "pleased", "wonderful", "nice",
    "fantastic",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "useless", "frustrated", "frustrating", "angry",
    "annoyed", "annoying", "hate", "worst", "horrible", "disappointed",
    "broken", "confusing", "confused", "slow", "unhelpful", "unclear",
    "difficult", "unsure",
}
NEGATIONS = {"not", "no", "never", "n't", "without"}


def analyze_sentiment(text: str) -> dict:
    words = re.findall(r"[a-z']+", text.lower())
    score = 0
    hits = []

    for i, word in enumerate(words):
        polarity = 0
        if word in POSITIVE_WORDS:
            polarity = 1
        elif word in NEGATIVE_WORDS:
            polarity = -1
        if polarity == 0:
            continue

        window = words[max(0, i - 2):i]
        if any(w in NEGATIONS or w.endswith("n't") for w in window):
            polarity *= -1

        score += polarity
        hits.append((word, polarity))

    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"

    return {"label": label, "score": score, "matched_words": hits}


if __name__ == "__main__":
    for sample in [
        "This is really helpful, thank you!",
        "This is not helpful at all, very frustrating.",
        "What documents are required to register a partnership deed?",
    ]:
        print(sample, "->", analyze_sentiment(sample))
