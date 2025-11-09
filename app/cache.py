from sentence_transformers import SentenceTransformer
import numpy as np

# 1️⃣ 80 MB model, runs on CPU, English only
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# 2️⃣ in-memory key-value stores
_cache_emb = []      # list of 384-dim vectors (np.array)
_cache_hint = []     # list of strings

SIMILARITY_THRESHOLD = 0.93   # cosine similarity above this = “same question”

def _cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_cached_hint(question: str) -> str | None:
    """Return hint if we already have a semantically identical question"""
    if not _cache_emb:
        return None
    q_vec = encoder.encode(question, normalize_embeddings=True)
    similarities = [_cos_sim(q_vec, c) for c in _cache_emb]
    best = max(similarities)
    if best >= SIMILARITY_THRESHOLD:
        idx = similarities.index(best)
        return _cache_hint[idx]
    return None

def store_hint(question: str, hint: str):
    """Cache a new (question, hint) pair"""
    q_vec = encoder.encode(question, normalize_embeddings=True)
    _cache_emb.append(q_vec)
    _cache_hint.append(hint)