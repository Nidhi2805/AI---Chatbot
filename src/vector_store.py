"""
Step 5 — Vector store.

A minimal TF-IDF-based stand-in for a production vector database. Stores
vectors plus a tag ("clean"/"flagged") per chunk.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
STORE_PATH = BASE_DIR / "models" / "vector_store.joblib"


class TfidfVectorStore:
    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.records = []

    def build(self, texts, doc_ids):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(texts)
        self.records = [{"doc_id": d, "text": t, "tag": None, "reason": None} for d, t in zip(doc_ids, texts)]

    def tag(self, doc_id, tag, reason):
        for r in self.records:
            if r["doc_id"] == doc_id:
                r["tag"] = tag
                r["reason"] = reason
                return
        raise KeyError(doc_id)

    def search(self, query, top_k=5, only_clean=True):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        order = np.argsort(-sims)
        results = []
        for idx in order:
            record = self.records[idx]
            if only_clean and record["tag"] != "clean":
                continue
            results.append({**record, "score": round(float(sims[idx]), 3)})
            if len(results) >= top_k:
                break
        return results

    def save(self, path=STORE_PATH):
        path.parent.mkdir(exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "matrix": self.matrix, "records": self.records}, path)

    @classmethod
    def load(cls, path=STORE_PATH):
        obj = joblib.load(path)
        store = cls()
        store.vectorizer = obj["vectorizer"]
        store.matrix = obj["matrix"]
        store.records = obj["records"]
        return store
