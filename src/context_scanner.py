"""
Step 6 — Context scanner (ingestion-time).

Chunks then scans every document ONCE, at ingestion, tagging each chunk
clean/flagged via content-pattern checks (not the prompt classifier — a
model trained on prompts doesn't generalise to document-style text, which
was a real bug found during testing). Only in-domain documents go into the
retrieval store.
"""

from pathlib import Path

import pandas as pd

from src.chunker import chunk_document
from src.features import structural_features
from src.vector_store import TfidfVectorStore

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_PATH = BASE_DIR / "data" / "documents.csv"


def scan_and_build_store(chunk_size: int = 150, overlap: int = 20) -> TfidfVectorStore:
    df = pd.read_csv(DOCS_PATH)
    df = df[df["is_in_domain"] == 1].reset_index(drop=True)

    all_chunks = []
    for _, row in df.iterrows():
        all_chunks.extend(chunk_document(row["doc_id"], row["text"], chunk_size=chunk_size, overlap=overlap))

    store = TfidfVectorStore()
    store.build(texts=[c["text"] for c in all_chunks], doc_ids=[c["chunk_id"] for c in all_chunks])

    flagged_count = 0
    for chunk in all_chunks:
        feats = structural_features(chunk["text"])
        triggers = []
        if feats["jailbreak_kw"] > 0:
            triggers.append(f"{int(feats['jailbreak_kw'])} jailbreak-pattern keyword(s)")
        if feats["injection_kw"] > 0:
            triggers.append(f"{int(feats['injection_kw'])} injection-pattern keyword(s)")
        if feats["pii_hits"] > 0:
            triggers.append(f"{int(feats['pii_hits'])} PII-like pattern(s)")

        if triggers:
            store.tag(chunk["chunk_id"], "flagged", triggers)
            flagged_count += 1
        else:
            store.tag(chunk["chunk_id"], "clean", "no injection/PII patterns detected")

    print(f"Chunked {len(df)} documents into {len(all_chunks)} chunks (size={chunk_size}, overlap={overlap}) "
          f"at ingestion — {flagged_count} flagged, {len(all_chunks) - flagged_count} clean.")
    return store


if __name__ == "__main__":
    store = scan_and_build_store()
    store.save()
    print(f"Vector store saved with {len(store.records)} tagged chunks.")
    results = store.search("What documents are required for KYC verification?", top_k=3)
    for r in results:
        print(f"  [{r['tag']}] {r['score']:.3f}  {r['text'][:70]}...")
