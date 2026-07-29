import chromadb
from sentence_transformers import SentenceTransformer
from app.config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, EMDEDDING_MODEL

_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
_embedder = SentenceTransformer(EMBEDDING_MODEL)

def get_collection():
    return _client.get_or_create_collection(
        name = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"},

    )

def embed(texts: list[str]) -> list[list[float]]:
    return _embedder.encode(texts, normalize_embeddings=True).tolist()
