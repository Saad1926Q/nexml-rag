# app/temp_vector.py
import os
import uuid
from typing import List, Dict, Any
from langchain_core.documents import Document
from dotenv import load_dotenv
import chromadb

from utils.utils import chunk_text_and_generate_embeddings

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db_data")

# Use PersistentClient so your chroma data directory stays consistent with your project
_chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)


def create_temp_collection(prefix: str = "temp_proposal_") -> chromadb.api.models.Collection:
    """
    Create a temporary chroma collection with a unique name and return it.
    """
    name = f"{prefix}{uuid.uuid4().hex[:8]}"
    collection = _chroma_client.get_or_create_collection(name=name, metadata={"ephemeral": True})
    return collection


def add_documents_to_collection(collection, docs: List[Document]) -> Dict[str, Any]:
    """
    Chunk documents, compute embeddings and add them into the given chroma collection.

    Returns a dict:
      {
        "ids_added": int,
        "collection_name": str,
        "embeddings_model": embeddings_model_object
      }

    Note: embeddings_model is returned (object) so the caller can use the same model to embed queries.
    """
    if not docs:
        return {"ids_added": 0, "collection_name": collection.name, "embeddings_model": None}

    chunked_docs, embeddings_model = chunk_text_and_generate_embeddings(docs)

    ids = []
    documents = []
    metadatas = []

    for i, doc in enumerate(chunked_docs):
        ids.append(f"{collection.name}_doc_{i}")
        documents.append(doc.page_content)
        md = doc.metadata if hasattr(doc, "metadata") and doc.metadata else {}
        md = dict(md)
        md.setdefault("source_collection", "ephemeral")
        metadatas.append(md)

    # generate embeddings for all documents (list of strings)
    all_embeddings = embeddings_model.embed_documents(documents)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=all_embeddings,
        metadatas=metadatas,
    )

    return {
        "ids_added": len(ids),
        "collection_name": collection.name,
        "embeddings_model": embeddings_model
    }


def delete_temp_collection(collection) -> bool:
    """
    Delete the named chroma collection. Returns True on success.
    """
    try:
        _chroma_client.delete_collection(name=collection.name)
        return True
    except Exception:
        return False
