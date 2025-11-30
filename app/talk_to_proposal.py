# app/server.py
import os
import tempfile
import asyncio
import uuid
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document


from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TEXT_EMBEDDING_ID = os.getenv("TEXT_EMBEDDING_ID", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db_data")


try:
    import chromadb
except Exception as e:
    raise RuntimeError("chromadb required. pip install chromadb") from e

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
except Exception as e:
    raise RuntimeError("fastapi and uvicorn required. pip install fastapi uvicorn") from e

try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
except Exception:
    ChatGroq = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
except Exception:
    HuggingFaceEmbeddings = None

# doc extractor wrapper (expects your scripts/doc_extractor.py)
def extract_text_from_pdf(path: str) -> List[Document]:
    try:
        from scripts.doc_extractor import extract_text_images_tables
    except Exception as e:
        raise RuntimeError("Could not import scripts.doc_extractor. Ensure scripts/doc_extractor.py exists.") from e
    docs = extract_text_images_tables(path)
    if isinstance(docs, tuple):
        return docs[0]
    return docs

# Chroma client
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def create_temp_collection(prefix: str = "temp_proposal_"):
    name = f"{prefix}{uuid.uuid4().hex[:8]}"
    return client.get_or_create_collection(name=name, metadata={"ephemeral": True})

def chunk_and_embed_documents(documents: List[Document]):
    if HuggingFaceEmbeddings is None:
        raise RuntimeError("langchain_huggingface not available. pip install langchain-huggingface or set TEXT_EMBEDDING_ID.")
    embeddings = HuggingFaceEmbeddings(model_name=TEXT_EMBEDDING_ID)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return chunks, embeddings

def add_documents_to_collection(collection, docs: List[Document]):
    if not docs:
        return {"ids_added": 0, "collection_name": collection.name, "embeddings_model": None}
    chunked_docs, embeddings_model = chunk_and_embed_documents(docs)
    ids, documents, metadatas = [], [], []
    for i, d in enumerate(chunked_docs):
        ids.append(f"{collection.name}_doc_{i}")
        documents.append(d.page_content)
        md = d.metadata if hasattr(d, "metadata") and d.metadata else {}
        md = dict(md)
        md.setdefault("source_collection", "ephemeral")
        metadatas.append(md)
    all_embeddings = embeddings_model.embed_documents(documents)
    collection.add(ids=ids, documents=documents, embeddings=all_embeddings, metadatas=metadatas)
    return {"ids_added": len(ids), "collection_name": collection.name, "embeddings_model": embeddings_model}

def delete_collection(collection) -> bool:
    try:
        client.delete_collection(name=collection.name)
        return True
    except Exception:
        return False

# simple in-memory session manager
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
    return {"session_id": session_id, "chunks_added": add_result["ids_added"]}

def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    return SESSIONS.get(session_id)

def end_session(session_id: str) -> bool:
    data = SESSIONS.pop(session_id, None)
    if not data:
        return False
    return delete_collection(data["collection"])

# LLM setup (guarded)
if ChatGroq is None or not GROQ_API_KEY:
    LLM = None
else:
    LLM = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", max_tokens=2048)

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
    return PromptTemplate(template=template, input_variables=["context", "question"]).invoke({"context": context, "question": question})

# FastAPI app
app = FastAPI(title="Minimal Ephemeral Proposal QA")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
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
    collection = session["collection"]; embeddings_model = session["embeddings_model"]
    query_embedding = await asyncio.to_thread(embeddings_model.embed_query, question)
    results = collection.query(query_embeddings=[query_embedding], n_results=5, include=["documents","metadatas","distances"])
    docs = results.get("documents", [[]])[0]
    context = "\n\n---\n\n".join(docs) if docs else ""
    prompt_text = build_answer_prompt(context, question)
    if LLM is None:
        return JSONResponse({"answer": "LLM not configured (GROQ_API_KEY missing)."}, status_code=503)
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
    return HTMLResponse("<h3>Minimal Ephemeral Proposal QA</h3><p>Use CLI or add static/index.html for browser UI.</p>")
