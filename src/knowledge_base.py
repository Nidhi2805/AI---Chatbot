"""
Step 17 — Knowledge base manager.

Addresses a real gap the mentor flagged: a user has no way to know what
the chatbot actually knows about, so they don't know what's reasonable to
ask. Two things fix that:

  1. A plain-language description of the active knowledge base, shown in
     the product UI before the user types anything.
  2. Letting a user upload their OWN documents to replace the bundled
     finance/legal one — the chatbot then answers from whatever they
     brought, not a fixed demo domain.

Uploading a custom KB rebuilds both the vector store (what gets retrieved)
and the topic gate (what counts as "in domain") around the new content —
reusing the same chunker/scanner/gate code, just pointed at different text.
Nothing here is security-domain-specific, so the input checkpoint and
action firewall are untouched; only retrieval-side components change.
"""

from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

from src.chunker import chunk_document
from src.features import structural_features
from src.topic_gate import TopicGate
from src.vector_store import TfidfVectorStore

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_PATH = BASE_DIR / "data" / "documents.csv"

# generic, domain-independent examples of "clearly not about any specific
# knowledge base" — used as the negative class whenever a custom KB is
# trained, since a user's own documents won't come with matching negative
# examples the way the bundled finance/legal dataset does.
GENERIC_OUT_OF_DOMAIN_EXAMPLES = [
    "What's the weather like today?",
    "Can you recommend a good movie to watch?",
    "Who won the football match last night?",
    "What's a good recipe for dinner tonight?",
    "How far away is the nearest star?",
    "What's the capital of Australia?",
    "Can you help me plan a trip to the mountains?",
    "What's a good workout routine for beginners?",
    "Tell me a joke.",
    "What's the population of Brazil?",
]

DEFAULT_KB_DESCRIPTION = (
    "This assistant answers questions about Indian company and business "
    "documentation — partnership deeds, trust deeds, KYC verification, "
    "certificates of incorporation, society registration, share and "
    "authorised capital, and annual compliance filings. Questions outside "
    "this area (general knowledge, other topics) will get a clear "
    "\"I don't have an answer to this question\" instead of a guess."
)


def default_kb_description() -> str:
    return DEFAULT_KB_DESCRIPTION


def build_custom_kb(raw_text: str, source_name: str = "uploaded document"):
    """Chunks + scans + indexes user-supplied text, and trains a fresh topic
    gate around it. Returns (vector_store, topic_gate, description)."""
    # 150-word chunks turn a short document (a one-page certificate) into a
    # single blob, so every query retrieves the same undifferentiated chunk
    # and the retriever can't point at the sentence that answers a specific
    # question. Smaller chunks with overlap let retrieval actually
    # discriminate ("who signed this" vs "which Act" vs "what CIN").
    chunks = chunk_document("custom_doc", raw_text, chunk_size=45, overlap=12)
    if not chunks:
        raise ValueError("No text found to build a knowledge base from.")

    store = TfidfVectorStore()
    store.build(texts=[c["text"] for c in chunks], doc_ids=[c["chunk_id"] for c in chunks])

    flagged = 0
    for c in chunks:
        feats = structural_features(c["text"])
        triggers = []
        if feats["jailbreak_kw"] > 0:
            triggers.append(f"{int(feats['jailbreak_kw'])} jailbreak-pattern keyword(s)")
        if feats["injection_kw"] > 0:
            triggers.append(f"{int(feats['injection_kw'])} injection-pattern keyword(s)")
        if feats["pii_hits"] > 0:
            triggers.append(f"{int(feats['pii_hits'])} PII-like pattern(s)")
        if triggers:
            store.tag(c["chunk_id"], "flagged", triggers)
            flagged += 1
        else:
            store.tag(c["chunk_id"], "clean", "no injection/PII patterns detected")

    gate = TopicGate()
    in_domain_texts = [c["text"] for c in chunks]
    gate.train_from_texts(in_domain_texts, GENERIC_OUT_OF_DOMAIN_EXAMPLES)

    description = (
        f"This assistant is answering from a custom knowledge base you uploaded "
        f"({source_name}, {len(chunks)} chunk(s), {flagged} flagged and excluded). "
        f"Questions outside what that document covers will get "
        f"\"I don't have an answer to this question\" instead of a guess."
    )
    return store, gate, description


if __name__ == "__main__":
    print(default_kb_description())
    print()
    sample_text = (
        "Our return policy allows returns within 30 days of purchase with a valid receipt. "
        "Refunds are issued to the original payment method within 5-7 business days. "
        "Items must be unused and in original packaging to qualify for a refund."
    )
    store, gate, desc = build_custom_kb(sample_text, source_name="return-policy.txt")
    print(desc)
    print()
    for q in ["What is your return policy?", "What is the capital of Japan?"]:
        print(q, "->", gate.predict(q))
