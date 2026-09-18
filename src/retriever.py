"""
Step 7 — Retriever. Thin wrapper around the tagged vector store: retrieves
the top-K *clean* candidates for a query.
"""

from src.vector_store import TfidfVectorStore


class Retriever:
    def __init__(self, store: TfidfVectorStore):
        self.store = store

    def retrieve(self, query: str, top_k: int = 5):
        return self.store.search(query, top_k=top_k, only_clean=True)
