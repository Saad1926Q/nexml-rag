# app/talk_to_proposals.py
import os
import tempfile
import asyncio
import uuid
import logging
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
load_dotenv()

# Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TEXT_EMBEDDING_ID = os.getenv("TEXT_EMBEDDING_ID", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db_data")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Required optional libs (guarded imports)
try:
    import chromadb
except Exception as e:
    raise RuntimeError("chromadb required. Install with: pip install chromadb") from e

# FastAPI imports
try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
except Exception as e:
    raise RuntimeError("fastapi required. Install with: pip install fastapi uvicorn") from e

# Preferred embeddings (langchain_huggingface)
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # optional
except Exception:
    HuggingFaceEmbeddings = None

# Fallback: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Optional: langchain text splitter (preferred)
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

# Optional: LLM (ChatGroq) — if you want LLM answers
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
except Exception:
    ChatGroq = None
    PromptTemplate = None

# Document extractor wrapper (requires your scripts/doc_extractor.py)
def extract_text_from_pdf(path: str) -> List[Any]:
    try:
        from scripts.doc_extractor import extract_text_images_tables
    except Exception as e:
        raise RuntimeError("Missing scripts/doc_extractor.py (extract_text_images_tables)") from e
    docs = extract_text_images_tables(path)
    # extract_text_images_tables may return (doc_list, imgs, tables) or a doc_list
    if isinstance(docs, tuple):
        return docs[0]
    return docs

# Chroma client
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def create_temp_collection(prefix: str = "temp_proposal_"):
    name = f"{prefix}{uuid.uuid4().hex[:8]}"
    return client.get_or_create_collection(name=name, metadata={"ephemeral": True})

def _split_documents(documents: List[Any]) -> List[Any]:
    """Return list of langchain Document-like chunks (must have page_content & metadata)."""
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return splitter.split_documents(documents)
    # simple fallback naive chunker:
    out = []
    for d in documents:
        text = getattr(d, "page_content", str(d))
        meta = getattr(d, "metadata", {}) or {}
        i = 0
        while i < len(text):
            chunk = text[i : i + CHUNK_SIZE]
            out.append(type(d)(page_content=chunk, metadata=meta) if hasattr(d, "__class__") else type("Doc", (), {"page_content": chunk, "metadata": meta})())
            i += CHUNK_SIZE - CHUNK_OVERLAP
    return out

class EmbeddingsWrapper:
    """Unified wrapper with embed_documents and embed_query to normalize APIs."""
    def __init__(self, backend, backend_type: str):
        self.backend = backend
        self.type = backend_type  # "langchain" or "sentence-transformers"

    def embed_documents(self, texts: List[str]):
        if self.type == "langchain":
            return self.backend.embed_documents(texts)
        # sentence-transformers: return numpy array (chromadb accepts lists/arrays)
        return self.backend.encode(texts, convert_to_numpy=True)

    def embed_query(self, text: str):
        if self.type == "langchain":
            return self.backend.embed_query(text)
        return self.backend.encode([text], convert_to_numpy=True)[0]

def chunk_and_embed_documents(documents: List[Any]):
    """
    Returns (chunks, embeddings_wrapper)
    - chunks: list of Documents (with page_content & metadata)
    - embeddings_wrapper: has embed_documents(texts) and embed_query(text)
    """
    chunks = _split_documents(documents)

    # Prefer langchain_huggingface if available
    if HuggingFaceEmbeddings is not None:
        try:
            emb = HuggingFaceEmbeddings(model_name=TEXT_EMBEDDING_ID)
            return chunks, EmbeddingsWrapper(emb, "langchain")
        except Exception as e:
            log.warning("HuggingFaceEmbeddings initialization failed, falling back: %s", e)

    # Fallback to sentence-transformers
    if SentenceTransformer is not None:
        try:
            model_name = TEXT_EMBEDDING_ID or "sentence-transformers/all-MiniLM-L6-v2"
            st = SentenceTransformer(model_name)
            return chunks, EmbeddingsWrapper(st, "sentence-transformers")
        except Exception as e:
            log.warning("SentenceTransformer initialization failed: %s", e)

    raise RuntimeError(
        "No embeddings backend available. Install 'langchain-huggingface' or 'sentence-transformers', "
        "and ensure TEXT_EMBEDDING_ID is set correctly."
    )

def add_documents_to_collection(collection, docs: List[Any]):
    if not docs:
        return {"ids_added": 0, "collection_name": collection.name, "embeddings_model": None}
    chunked_docs, embeddings_wrapper = chunk_and_embed_documents(docs)
    ids, documents, metadatas = [], [], []
    for i, d in enumerate(chunked_docs):
        doc_id = f"{collection.name}_doc_{i}"
        ids.append(doc_id)
        documents.append(d.page_content)
        md = getattr(d, "metadata", {}) or {}
        md = dict(md)
        md.setdefault("source_collection", "ephemeral")
        metadatas.append(md)
    # compute embeddings in thread to avoid blocking event loop
    all_embeddings = embeddings_wrapper.embed_documents(documents)
    collection.add(ids=ids, documents=documents, embeddings=all_embeddings, metadatas=metadatas)
    return {"ids_added": len(ids), "collection_name": collection.name, "embeddings_model": embeddings_wrapper}

def delete_collection(collection) -> bool:
    try:
        client.delete_collection(name=collection.name)
        return True
    except Exception as e:
        log.warning("Failed to delete collection %s: %s", getattr(collection, "name", "<unknown>"), e)
        return False

# In-memory session store
SESSIONS: Dict[str, Dict[str, Any]] = {}

def create_session_from_pdf_path(path: str) -> Dict[str, Any]:
    docs = extract_text_from_pdf(path)
    if not docs:
        raise RuntimeError("No document extracted from PDF.")
    proposal_doc = docs[0]
    collection = create_temp_collection()
    add_result = add_documents_to_collection(collection, [proposal_doc])
    session_id = collection.name
    SESSIONS[session_id] = {"collection": collection, "embeddings_model": add_result["embeddings_model"]}
    log.info("Created ephemeral session %s (chunks: %d)", session_id, add_result["ids_added"])
    return {"session_id": session_id, "chunks_added": add_result["ids_added"]}

def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    return SESSIONS.get(session_id)

def end_session(session_id: str) -> bool:
    data = SESSIONS.pop(session_id, None)
    if not data:
        return False
    return delete_collection(data["collection"])

# LLM setup (optional)
LLM = None
if ChatGroq is not None and GROQ_API_KEY:
    try:
        LLM = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", max_tokens=2048)
    except Exception as e:
        log.warning("Could not initialize ChatGroq LLM: %s", e)
        LLM = None

def build_answer_prompt(context: str, question: str) -> str:
    template = """
You are an assistant. Answer ONLY from the provided context (the proposal chunks).
If the answer is not in the context, reply exactly: "I don't know (not in the proposal)."

Context:
{context}

Question:
{question}

Answer:
"""
    if PromptTemplate is not None:
        return PromptTemplate(template=template, input_variables=["context", "question"]).invoke({"context": context, "question": question})
    return template.format(context=context, question=question)

# FastAPI app
app = FastAPI(title="Ephemeral Proposal QA")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read()); tmp_path = tmp.name
    try:
        session_info = create_session_from_pdf_path(tmp_path)
        return JSONResponse(session_info)
    finally:
        try: os.remove(tmp_path)
        except: pass

@app.post("/chat")
async def chat(payload: Dict[str, str]):
    session_id = payload.get("session_id"); question = payload.get("question")
    if not session_id or not question:
        raise HTTPException(status_code=400, detail="session_id and question required")
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    collection = session["collection"]; embeddings_wrapper = session["embeddings_model"]
    # embed the query in background thread
    query_embedding = await asyncio.to_thread(embeddings_wrapper.embed_query, question)
    results = collection.query(query_embeddings=[query_embedding], n_results=5, include=["documents","metadatas","distances"])
    docs = results.get("documents", [[]])[0]
    context = "\n\n---\n\n".join(docs) if docs else ""
    prompt_text = build_answer_prompt(context, question)
    if LLM is None:
        # return contextual answer fallback (no LLM): return retrieved context so caller can see text
        return JSONResponse({"answer": "LLM not configured (GROQ_API_KEY missing).", "context": context}, status_code=503)
    response = await asyncio.to_thread(LLM.invoke, prompt_text)
    answer = getattr(response, "content", str(response)).strip()
    return JSONResponse({"answer": answer})

@app.post("/end")
async def api_end(payload: Dict[str, str]):
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    ok = end_session(session_id)
    return JSONResponse({"ended": ok})

@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return HTMLResponse(open(index_file, "r", encoding="utf-8").read())
    return HTMLResponse("<h3>Ephemeral Proposal QA</h3><p>Use CLI or POST to /upload and /chat.</p>")
