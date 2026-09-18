"""
Step 15 — Document chunker.

Splits documents into ~150-word chunks (with overlap) before ingestion.
"""

DEFAULT_CHUNK_SIZE = 150
DEFAULT_OVERLAP = 20


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def chunk_document(doc_id: str, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list:
    pieces = chunk_text(text, chunk_size, overlap)
    return [
        {"chunk_id": f"{doc_id}_chunk{i:03d}", "doc_id": doc_id, "text": piece}
        for i, piece in enumerate(pieces)
    ]


if __name__ == "__main__":
    long_doc = " ".join(["clause"] + [f"word{i}" for i in range(400)])
    chunks = chunk_document("doc_long_001", long_doc, chunk_size=150, overlap=20)
    print(f"Document of {len(long_doc.split())} words -> {len(chunks)} chunks")
    for c in chunks:
        print(f"  {c['chunk_id']}: {len(c['text'].split())} words")
