import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS
from corpus import CORPUS


def fixed_size_chunk(text: str, size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS):
    words = text.split()
    if len(words) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks


def main():
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Fresh collection each run, so re-running ingest.py doesn't duplicate data
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, texts = [], []
    for doc_id, text in CORPUS:
        chunks = fixed_size_chunk(text)
        for i, chunk in enumerate(chunks):
            chunk_id = doc_id if len(chunks) == 1 else f"{doc_id}_{i}"
            ids.append(chunk_id)
            texts.append(chunk)

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(ids=ids, documents=texts, embeddings=embeddings)
    print(f"Ingested {len(texts)} chunks into collection '{COLLECTION_NAME}' at {CHROMA_PATH}")


if __name__ == "__main__":
    main()   




    
