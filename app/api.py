# app/api.py
import os
import asyncio
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from scripts.doc_extractor import extract_text_images_tables
from app.session_manager import create_session_with_document, get_session, end_session
from utils.utils import chunk_text_and_generate_embeddings, query_collection

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY must be set in .env")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    max_tokens=2048
)

app = FastAPI(title="NaCCER Ephemeral QA API")

# Serve static (client) from app/static
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Allow local dev CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class ChatRequest(BaseModel):
    session_id: str
    question: str


def build_answer_prompt(context: str, question: str) -> str:
    prompt_template = PromptTemplate(
        template="""
You are an assistant that answers questions using ONLY the provided context excerpts from a single proposal. 
If the context does not contain the answer, reply exactly: "I don't know (not in the proposal)."

Context:
{context}

Question: {question}

Answer (concise, grounded in context):
""",
        input_variables=["context", "question"]
    )
    return prompt_template.invoke({"context": context, "question": question})


@app.post("/upload")
async def upload_proposal(file: UploadFile = File(...)):
    """
    Uploads a proposal PDF, creates an ephemeral session (collection), and returns session_id.
    """
    suffix = os.path.splitext(file.filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        docs = extract_text_images_tables(tmp_path)
        if isinstance(docs, tuple):
            doc_list = docs[0]
        else:
            doc_list = docs

        if not doc_list:
            raise HTTPException(status_code=400, detail="Could not extract proposal from file.")

        doc = doc_list[0]
        session_info = create_session_with_document(doc)
        return JSONResponse({"session_id": session_info["session_id"], "chunks_added": session_info["chunks_added"]})
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Ask question for a given session_id (ephemeral collection).
    """
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    collection = session["collection"]
    embeddings_model = session["embeddings_model"]
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")

    # embed query using embeddings_model in a thread (embedding may be blocking)
    query_embedding = await asyncio.to_thread(embeddings_model.embed_query, question)

    # Query the collection with existing query_collection util (it handles reranking)
    results = await asyncio.to_thread(query_collection, collection, query_embedding, question, 5)

    doc_texts = results.get("documents", [[]])[0]
    if not doc_texts:
        context_text = ""
    else:
        context_text = "\n\n---\n\n".join(doc_texts)

    final_prompt = build_answer_prompt(context_text, question)

    # call llm.invoke in a thread
    response = await asyncio.to_thread(llm.invoke, final_prompt)

    # llm.invoke returns an object with .content in your code earlier
    text = getattr(response, "content", None)
    if text is None:
        # maybe response is str
        text = str(response)

    return JSONResponse({"answer": text.strip()})


@app.post("/end")
async def end_session_endpoint(payload: dict):
    """
    End session and delete ephemeral collection.
    payload: {"session_id": "<id>"}
    """
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    success = end_session(session_id)
    return JSONResponse({"ended": success})


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the minimal chat client.
    """
    index_file = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_file):
        # provide minimal fallback HTML (should not normally happen if you add static file)
        return HTMLResponse("<html><body><h1>Create static/index.html</h1></body></html>")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
