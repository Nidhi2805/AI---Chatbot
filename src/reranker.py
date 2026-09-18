"""
Step 8 — Reranker. BM25 reranking of the retriever's shortlist.

BM25's index is built once over the FULL corpus, not re-fit from scratch
on just the small candidate subset each call. Real bug found in testing:
rebuilding BM25 on only ~6 retrieved candidates makes its IDF statistics
wildly unstable — a generic word like "typically" that happens to appear
in only 1 of those 6 documents gets treated as rare and highly
discriminative, inflating an irrelevant document's score above the one a
human (and TF-IDF retrieval itself) would clearly call correct. Scoring
against a corpus-wide index, then just looking up scores for the specific
candidates, gives IDF statistics that reflect the whole knowledge base
rather than whatever handful of documents happened to be retrieved.
"""

import re

from rank_bm25 import BM25Okapi

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
    "how", "do", "does", "did", "of", "in", "on", "for", "to", "and", "or",
    "this", "that", "it", "be", "can", "you", "your", "i", "my",
}

# common suffixes stripped so word-form variants match ("requires" and
# "required" should both match "require") -- a real miss found in testing:
# "What is required to set up a private trust?" outranked the correct
# trust-deed document with an unrelated one, partly because "requires" in
# the document never matched "required" in the query at all.
_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(word: str) -> str:
    if len(word) <= 4:
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def _tokenize(text: str):
    # hyphenated compounds ("paid-up") are kept as one token instead of
    # splitting into fragments -- "up" alone was coincidentally matching
    # unrelated queries like "set up a trust", inflating irrelevant scores
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return [_stem(w) for w in words if w not in STOPWORDS]


class Reranker:
    def __init__(self, corpus_texts: list = None):
        self._bm25 = None
        self._corpus_texts = list(corpus_texts) if corpus_texts else []
        if self._corpus_texts:
            tokenized = [_tokenize(t) for t in self._corpus_texts]
            self._bm25 = BM25Okapi(tokenized)

    def rerank(self, query: str, candidates: list) -> list:
        if not candidates:
            return []

        if self._bm25 is not None:
            # score against the corpus-wide index, then pick out just the
            # scores for the given candidates
            full_scores = self._bm25.get_scores(_tokenize(query))
            text_to_score = dict(zip(self._corpus_texts, full_scores))
            reranked = [
                {**c, "rerank_score": round(float(text_to_score.get(c["text"], 0.0)), 3)}
                for c in candidates
            ]
        else:
            # fallback for when no corpus was supplied: old, less stable
            # behaviour (fit fresh on just the candidates)
            tokenized_corpus = [_tokenize(c["text"]) for c in candidates]
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(_tokenize(query))
            reranked = [
                {**c, "rerank_score": round(float(s), 3)}
                for c, s in zip(candidates, scores)
            ]

        reranked.sort(key=lambda r: r["rerank_score"], reverse=True)
        return reranked
