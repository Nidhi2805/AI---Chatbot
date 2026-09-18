"""
Step 13 — Conversational gate.

Greetings and small talk get an instant response with zero delay — no
input checkpoint, no topic gate, no retrieval, no LLM. Pure pattern
matching, deliberately not even a classical ML model. Each category has
several response variants, chosen at random per call. Runs FIRST in the
pipeline, before every other checkpoint.
"""

import random
import re

PATTERNS = {
    "greeting": {
        # "what's up"/"whats up"/"yo" added after a real miss: "What's up?"
        # fell through to the topic gate instead of being recognised here.
        # "good day" added after the same category of miss.
        "patterns": [r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bgood morning\b",
                     r"\bgood afternoon\b", r"\bgood evening\b", r"\bgood day\b",
                     r"\bwhat's up\b", r"\bwhats up\b", r"\byo\b"],
        "responses": [
            "Hello! How can I help you today?",
            "Hi there! What can I help you with?",
            "Hey! What would you like to know?",
            "Hi! Happy to help — what's on your mind?",
        ],
    },
    "wellbeing": {
        # "how have you been"/"how you doing"/"how you been" added after a
        # real miss — the original patterns only covered present-tense
        # "how are you", not the equally common past-tense phrasing.
        "patterns": [r"how are you\b", r"how's it going\b", r"how are things\b",
                     r"how have you been\b", r"how you doin'?g?\b", r"how you been\b"],
        "responses": [
            "I'm doing well, thanks for asking! How can I help you today?",
            "Doing great, thanks! What can I do for you?",
            "All good on my end! What can I help with?",
        ],
    },
    "thanks": {
        # broadened from the literal phrase "appreciate it" to the word
        # "appreciate" alone, after "I really appreciate your quick
        # response" was missed — this category is low-stakes (a false
        # match just means an unrelated sentence gets a "you're welcome"),
        # so a broader net is a safe trade here.
        "patterns": [r"\bthank(s| you)\b", r"\bappreciate\b"],
        "responses": [
            "You're welcome! Let me know if there's anything else you need.",
            "Happy to help! Anything else I can do for you?",
            "Anytime! Feel free to ask if you have more questions.",
        ],
    },
    "farewell": {
        # "catch you"/"take care"/"heading out" added after "Catch you
        # later" was missed by the original, narrower pattern list.
        "patterns": [r"\bbye\b", r"\bgoodbye\b", r"\bsee you\b", r"\btalk later\b",
                     r"\bcatch you\b", r"\btake care\b", r"\bheading out\b"],
        "responses": [
            "Goodbye! Feel free to come back if you have more questions.",
            "See you! Come back anytime.",
            "Take care! I'm here if you need anything else.",
        ],
    },
}

_compiled = [
    (category, re.compile(p, re.IGNORECASE))
    for category, spec in PATTERNS.items()
    for p in spec["patterns"]
]


def check_greeting(text: str) -> dict:
    for category, pattern in _compiled:
        if pattern.search(text):
            response = random.choice(PATTERNS[category]["responses"])
            return {"is_smalltalk": True, "category": category, "response": response}
    return {"is_smalltalk": False, "category": None, "response": None}


if __name__ == "__main__":
    for sample in [
        "Hi there!", "What's up?", "How are you doing today?",
        "Thanks a lot for the help", "Goodbye for now",
        "What documents are required to register a partnership deed?",
    ]:
        print(sample, "->", check_greeting(sample))
