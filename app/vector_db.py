import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

load_dotenv() 

db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db_data")

client = chromadb.PersistentClient(path=db_path)

guidelines_collection = client.get_or_create_collection(
    name="s_and_t_guidelines",
    metadata={"hnsw:space": "cosine"}
)

proposals_collection = client.get_or_create_collection(
    name="naccer_proposals",
    metadata={"hnsw:space": "cosine"}
)

class ChromaDB:
    def __init__(self, persistent_directory: str = db_path):
        try:
            self.client = chromadb.PersistentClient(path=persistent_directory)
        except Exception:
            self.client = chromadb.Client(
                ChromaSettings(chroma_db_impl="duckdb+parquet", persist_directory=persistent_directory)
            )
        
        self.sandt = self.client.get_or_create_collection(
            name="s_and_t_guidelines", metadata={"hnsw:space": "cosine"}
        )
        self.proposals = self.client.get_or_create_collection(
            name="naccer_proposals", metadata={"hnsw:space": "cosine"}
        )

    def add_guideline_sections(self, sections: List[Dict[str, Any]]):
        """
        sections: list of dicts with keys: section_id, title, text, embedding (list[float]), doc_id, doc_title, last_updated
        Stores each section as a document in s_and_t_guidelines collection.
        """
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        embs: List[List[float]] = []

        for s in sections:
            sid = s.get("section_id") or s.get("id")
            if sid is None:
                raise ValueError("Each section must have a 'section_id' or 'id' field.")
            if "text" not in s or "embedding" not in s:
                raise ValueError("Each section must include 'text' and 'embedding' keys.")

            ids.append(str(sid))
            docs.append(s["text"])
            metas.append({
                "section_id": sid,
                "title": s.get("title"),
                "doc_id": s.get("doc_id"),
                "doc_title": s.get("doc_title"),
                "last_updated": s.get("last_updated"),
                **(s.get("metadata") or {})
            })
            embs.append(s["embedding"])

        self.sandt.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    def add_proposal_chunks(
        self,
        proposal_id: str,
        title: str,
        authors: List[str],
        chunks: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Each chunk dict must contain: chunk_id, text, embedding(list), offset (int), optional metadata.
        We'll add each chunk as a document with metadata referencing the parent proposal.
        """
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        embs: List[List[float]] = []

        for ch in chunks:
            if "chunk_id" not in ch or "text" not in ch or "embedding" not in ch:
                raise ValueError("Each chunk must contain 'chunk_id', 'text', and 'embedding' keys.")
            cid = ch["chunk_id"]
            ids.append(f"{proposal_id}__{cid}")
            docs.append(ch["text"])
            meta = {
                "proposal_id": proposal_id,
                "proposal_title": title,
                "authors": authors,
                "chunk_id": cid,
                "offset": ch.get("offset"),
            }
            # merge any chunk-level metadata
            meta.update(ch.get("metadata") or {})
            # include optional top-level metadata once per chunk for easy retrieval
            if metadata:
                meta.update(metadata)
            metas.append(meta)
            embs.append(ch["embedding"])

        self.proposals.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    def query_similar_proposals(self, query_embedding: List[float], top_k: int = 5):
        """
        Returns top_k matching chunks from proposals collection, together with their metadata and distances.
        """
        res = self.proposals.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        results: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            results.append({"text": doc, "metadata": meta, "distance": float(dist)})
        return results
    
    def query_sandt_sections(self, query_embedding: List[float], top_k: int = 5):
        """
        Query guidelines collection (s_and_t_guidelines)
        """
        res = self.sandt.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        results: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            results.append({"text": doc, "metadata": meta, "distance": float(dist)})
        return results
    
    def persist(self):
        if hasattr(self.client, "persist"):
            self.client.persist()
        if hasattr(client, "persist"):
            try:
                client.persist()
            except Exception:
                pass

        

