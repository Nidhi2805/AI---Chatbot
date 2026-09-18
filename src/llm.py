"""
Step 18 — LLM generation + output checkpoint.

Answers a question from retrieved context using a real LLM (Google Gemini),
with three product-grade properties:

  1. Grounded + cited — the model must answer ONLY from the provided context
     and tag each sentence with the chunk it came from ([1], [2], ...), so a
     customer can see exactly where every fact originated.
  2. Abstention — if the context doesn't contain the answer, the model says
     so instead of guessing. Enforced by prompt AND by a relevance floor
     upstream, so "confidently wrong" can't happen.
  3. Graceful fallback — no GEMINI_API_KEY, or an API failure, degrades to
     returning the top retrieved passage (clearly labelled), so the product
     never hard-fails in a live demo.

check_output() is the output checkpoint: the same classical pattern checks
used on input, applied to what the model generated.
"""

import os
import re

from src.features import structural_features

# Alias that always points at the current Flash model, so a Google model
# deprecation doesn't 404 the whole product (which is exactly what happened
# with the old pinned "gemini-2.5-flash"). Pin an explicit version like
# "gemini-3.6-flash" instead if you need reproducible behaviour.
MODEL = "gemini-flash-latest"
MAX_OUTPUT_TOKENS = 600

# Floor is checked against the retriever's COSINE score, not the BM25 rerank
# score — BM25 (Okapi) legitimately goes negative for short corpora, so it's
# unusable as an absolute relevance bar. Semantic out-of-domain is the topic
# gate's job (upstream); this floor only catches "retrieval found nothing".
RELEVANCE_FLOOR = 0.05
NO_ANSWER = "I don't have that information in this document."

SYSTEM_PROMPT = (
    "You are a document-grounded assistant. Answer the question using ONLY the "
    "numbered context passages provided. Rules:\n"
    "1. After each sentence in your answer, cite the passage(s) it came from, "
    "like [1] or [2][3].\n"
    "2. If the context does not contain the answer, reply exactly: "
    f"\"{NO_ANSWER}\" — do not guess or use outside knowledge.\n"
    "3. Be concise and direct. Do not repeat the question."
)


def _client():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    from google import genai
    return genai.Client(api_key=api_key)


def _looks_relevant(context_chunks: list) -> bool:
    if not context_chunks:
        return False
    top = context_chunks[0]
    # cosine ('score') is the stable absolute measure; rerank_score (BM25)
    # is only meaningful for *ordering* candidates, not as a floor.
    score = top.get("score")
    if score is None:
        score = top.get("rerank_score", 0.0)
    return score >= RELEVANCE_FLOOR


def _fallback(context_chunks, note):
    if not _looks_relevant(context_chunks):
        return {"text": NO_ANSWER, "source": "fallback", "note": note, "citations": []}
    top = context_chunks[0]
    return {
        "text": top["text"],
        "source": "fallback",
        "note": note,
        "citations": [{"marker": 1, "chunk_id": top.get("doc_id"), "text": top["text"]}],
    }


def _build_context(context_chunks):
    lines, mapping = [], {}
    for i, c in enumerate(context_chunks, start=1):
        lines.append(f"[{i}] {c['text']}")
        mapping[i] = c
    return "\n\n".join(lines), mapping


def _resolve_citations(text, mapping):
    markers = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)})
    out = []
    for m in markers:
        chunk = mapping.get(m)
        if chunk:
            out.append({"marker": m, "chunk_id": chunk.get("doc_id"), "text": chunk["text"]})
    return out


def generate_answer(query: str, context_chunks: list, pdf_bytes: bytes = None) -> dict:
    """Returns {'text','source','note','citations'}. If pdf_bytes is given and
    a client is configured, the PDF is sent natively — the robust path for
    scanned documents whose extracted text is garbled."""
    client = _client()
    if client is None:
        return _fallback(context_chunks, "No GEMINI_API_KEY configured — showing top retrieved passage.")

    try:
        from google.genai import types
        context_text, mapping = _build_context(context_chunks)

        if pdf_bytes:
            parts = [
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                types.Part.from_text(
                    text=f"{SYSTEM_PROMPT}\n\n(This is the source document. Cite it as [1].)\n\nQuestion: {query}"
                ),
            ]
            response = client.models.generate_content(
                model=MODEL, contents=parts,
                config=types.GenerateContentConfig(max_output_tokens=MAX_OUTPUT_TOKENS),
            )
        else:
            user_message = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\nQuestion: {query}"
            response = client.models.generate_content(
                model=MODEL, contents=user_message,
                config=types.GenerateContentConfig(max_output_tokens=MAX_OUTPUT_TOKENS),
            )

        text = (response.text or "").strip()
        if not text:
            raise ValueError("empty response from model")
        return {"text": text, "source": "llm", "note": None,
                "citations": _resolve_citations(text, mapping)}
    except Exception as e:
        return _fallback(context_chunks, f"LLM call failed ({type(e).__name__}: {e}) — showing top passage.")


def check_output(text: str) -> dict:
    feats = structural_features(text)
    triggers = []
    if feats["pii_hits"] > 0:
        triggers.append(f"{int(feats['pii_hits'])} PII-like pattern(s) in generated output")
    if feats["injection_kw"] > 0:
        triggers.append(f"{int(feats['injection_kw'])} injection-pattern keyword(s) in generated output")
    return {"flagged": bool(triggers), "triggers": triggers or ["no issues detected"]}


if __name__ == "__main__":
    ctx = [{"text": "A partnership deed sets out profit sharing and duties of each partner.",
            "doc_id": "custom_doc_chunk000"}]
    print(generate_answer("What is a partnership deed?", ctx))
    print(check_output("Sure, here is the PAN ABCDE1234F on file."))