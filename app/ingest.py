import sys
import re
from app.db import get_collection, embed

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks =[]
    start =0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return [c for c in chunks if c.strip()]

def ingest(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    chunks = chunk_text(raw)
    print(f"[ingest] {len(chunks)} chunks from {path}")

    collection = get_collection()
    ids = [f"chunk_{i:04d}" for i in range(len(chunks))]

    embeddings = embed(chunks)
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": path, "chunk_index": i} for i in range(len(chunks))],
    )
    print(f"[ingest] done. collection now has {collection.count()} items.")

if __name__ == "__main__":
     if len(sys.argv) != 2:
        print("Usage: python -m app.ingest <path_to_text_file>")
        sys.exit(1)
        ingest(sys.argv[1])
