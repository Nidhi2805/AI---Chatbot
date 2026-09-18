"""
Step 4 — Topic / answerability gate.

Decides whether a query is even answerable from this knowledge base before
retrieval runs. Trains on BOTH document sentences and short conversational
queries. Decision combines TF-IDF cosine similarity against the in-domain
centroid with a small classifier, both trained on the combined corpus.
"""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_PATH = BASE_DIR / "data" / "documents.csv"
QUERIES_PATH = BASE_DIR / "data" / "topic_queries.csv"
MODEL_PATH = BASE_DIR / "models" / "topic_gate.joblib"

SIMILARITY_THRESHOLD = 0.02
# Lowered from 0.05 after a real miss: a correctly-labelled in-domain
# compliance question scored 0.036-0.045 similarity (just under the old
# threshold) despite the classifier itself being confident (0.67-0.69) —
# the AND-gate was vetoing a right answer. Every out-of-domain case tested
# so far (capital-of-country, weather, recipes, trivia) scored similarity
# 0.0 exactly, well clear of this lower bar, so this fixes the false
# negative without reopening the earlier false-positive bugs.

STRONG_SIMILARITY_OVERRIDE = 0.15
# Added after a real miss when custom knowledge bases arrived: with only a
# handful (or one) positive training example, the classifier can't learn a
# reliable boundary at all — a genuinely on-topic question ("what is your
# return policy?" against a KB that's ONLY about return policy) scored
# 0.422 classifier confidence despite 0.253 similarity, well above every
# out-of-domain case tested so far. A similarity this strong is treated as
# sufficient on its own, since it reflects direct lexical overlap with the
# actual knowledge base rather than a classifier's shaky guess.


class TopicGate:
    def __init__(self, vectorizer=None, classifier=None, centroid=None):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.centroid = centroid

    def train(self, docs_path=DOCS_PATH, queries_path=QUERIES_PATH):
        docs_df = pd.read_csv(docs_path)[["text", "is_in_domain"]]
        queries_df = pd.read_csv(queries_path)[["text", "is_in_domain"]]
        df = pd.concat([docs_df, queries_df], ignore_index=True)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        X = self.vectorizer.fit_transform(df["text"])

        in_domain_mask = df["is_in_domain"] == 1
        doc_in_domain_mask = (docs_df["is_in_domain"] == 1).values
        doc_X = self.vectorizer.transform(docs_df["text"])
        self.centroid = np.asarray(doc_X[doc_in_domain_mask].mean(axis=0))

        self.classifier = LogisticRegression(max_iter=5000, class_weight="balanced")
        self.classifier.fit(X, df["is_in_domain"])

        return (f"Trained on {len(df)} examples ({len(docs_df)} documents + {len(queries_df)} queries): "
                f"{in_domain_mask.sum()} in-domain, {(~in_domain_mask).sum()} out-of-domain")

    def train_from_texts(self, in_domain_texts: list, out_domain_texts: list):
        """Same training logic as train(), but from in-memory text lists
        instead of CSV files — used when a user uploads their own knowledge
        base, so a fresh gate can be built around whatever domain they
        bring, rather than reusing the bundled finance/legal training data.

        A real problem this has to handle: a short uploaded document might
        be only 1-3 chunks, against 10 generic out-of-domain examples. The
        classifier is then badly outnumbered and skews toward "out of
        domain" — scoring even correct in-domain questions below 0.5 (a
        vendor-policy upload scored 0.468 and 0.378 on questions its own
        text clearly answered). Splitting the in-domain text into sentences
        gives the classifier many more positive examples of that domain's
        vocabulary, without inventing any content that isn't in the
        document.
        """
        augmented = list(in_domain_texts)
        for text in in_domain_texts:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
                sentence = sentence.strip()
                if len(sentence.split()) >= 4:  # skip fragments/headers
                    augmented.append(sentence)

        docs_df = pd.DataFrame({"text": augmented, "is_in_domain": 1})
        queries_df = pd.DataFrame({"text": out_domain_texts, "is_in_domain": 0})
        df = pd.concat([docs_df, queries_df], ignore_index=True)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        X = self.vectorizer.fit_transform(df["text"])

        # centroid stays built from the original chunks, not the augmented
        # sentence splits, so similarity scoring is unchanged
        doc_X = self.vectorizer.transform(in_domain_texts)
        self.centroid = np.asarray(doc_X.mean(axis=0))

        self.classifier = LogisticRegression(max_iter=5000, class_weight="balanced")
        self.classifier.fit(X, df["is_in_domain"])

        return (f"Trained on {len(in_domain_texts)} chunk(s) expanded to {len(augmented)} "
                f"in-domain examples + {len(queries_df)} out-of-domain examples")

    def predict(self, query: str) -> dict:
        X = self.vectorizer.transform([query])
        similarity = float(cosine_similarity(X, self.centroid)[0][0])
        proba = self.classifier.predict_proba(X)[0]
        classes = self.classifier.classes_
        in_domain_idx = list(classes).index(1)
        classifier_confidence = float(proba[in_domain_idx])

        in_domain = similarity >= SIMILARITY_THRESHOLD and (
            classifier_confidence >= 0.5 or similarity >= STRONG_SIMILARITY_OVERRIDE
        )

        return {
            "in_domain": bool(in_domain),
            "similarity_to_domain": round(similarity, 3),
            "classifier_confidence": round(classifier_confidence, 3),
        }

    def save(self, path=MODEL_PATH):
        path.parent.mkdir(exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "classifier": self.classifier, "centroid": self.centroid}, path)

    @classmethod
    def load(cls, path=MODEL_PATH):
        obj = joblib.load(path)
        return cls(vectorizer=obj["vectorizer"], classifier=obj["classifier"], centroid=obj["centroid"])


if __name__ == "__main__":
    gate = TopicGate()
    print(gate.train())
    gate.save()
    print(f"Saved trained gate to {MODEL_PATH}")

    for sample in [
        "What documents are required to register a partnership deed?",
        "What's the weather like in Paris this weekend?",
        "What is the capital of Italy?",
        "How is CIN validated during document verification?",
    ]:
        print(sample, "->", gate.predict(sample))
