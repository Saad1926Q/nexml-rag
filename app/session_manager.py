# app/session_manager.py
import threading
from typing import Dict, Any
from app.temp_vector import create_temp_collection, delete_temp_collection, add_documents_to_collection
from langchain_core.documents import Document

_lock = threading.Lock()

# stores: session_id -> dict { collection, embeddings_model, metadata }
SESSIONS: Dict[str, Dict[str, Any]] = {}


def create_session_with_document(doc: Document) -> Dict[str, Any]:
    """
    Create a new ephemeral session, store the collection and embeddings model,
    return session info (session_id, collection_name).
    """
    with _lock:
        collection = create_temp_collection()
        add_result = add_documents_to_collection(collection, [doc])
        embeddings_model = add_result.get("embeddings_model")
        session_id = collection.name  # using collection.name as session id
        SESSIONS[session_id] = {
            "collection": collection,
            "embeddings_model": embeddings_model
        }
        return {"session_id": session_id, "collection_name": collection.name, "chunks_added": add_result["ids_added"]}


def get_session(session_id: str):
    with _lock:
        return SESSIONS.get(session_id)


def end_session(session_id: str) -> bool:
    with _lock:
        data = SESSIONS.pop(session_id, None)
    if not data:
        return False
    return delete_temp_collection(data["collection"])
