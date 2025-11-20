from typing import List, Dict, Any
from app.vector_db import ChromaDB
from scripts.embedding_provider import EmbeddingProvider

chroma = ChromaDB()
embedder = EmbeddingProvider()

def get_top_k_similar_proposals(text: str, k: int = 5):
    emb = embedder.embed_text(text[:2000])
    hits = chroma.query_similar_proposals(query_embedding=emb, top_k=k)
    grouped = {}
    for h in hits:
        meta = h["metadata"]
        pid = meta.get("proposal_id")
        if pid not in grouped or h["distance"] < grouped[pid]["distance"]:
            grouped[pid] = {
                "proposal_id": pid,
                "best_chunk": h["text"],
                "distance": h["distance"],
                "metadata": meta
            }
    
    out = sorted(grouped.values(), key=lambda x: x["distance"])[:k]
    return out

def get_relevant_sandt_sections(text: str, k: int = 5):
    emb = embedder.embed_text(text[:2000])
    hits = chroma.query_sandt_sections(query_embedding=emb, top_k=k)
    return hits
