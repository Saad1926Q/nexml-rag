import os
import sys
import json
from pypdf import PdfReader
from tqdm import tqdm
from dotenv import load_dotenv

from app.vector_db import ChromaDB
from helpers.chunker import simple_chunk_text
from scripts.embedding_provider import EmbeddingProvider

load_dotenv()

def extract_text_from_pdf(path: str) -> str:
    r = PdfReader(path)
    texts = []
    for p in r.pages:
        texts.append(p.extract_text() or "")
    return "\n\n".join(texts)

def split_into_sections(text: str, chunk_size=1500, overlap=200):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    sections = []
    for para in paras:
        if len(para) <= chunk_size:
            sections.append(para)
        else:
            sections.extend([c["text"] for c in simple_chunk_text(para, chunk_size=chunk_size, overlap=overlap)])
    return sections

def main(pdf_path: str, doc_id: str = None):
    if not os.path.exists(pdf_path):
        print("PDF not found:", pdf_path); sys.exit(1)
    text = extract_text_from_pdf(pdf_path)
    sections = split_into_sections(text, chunk_size=1500, overlap=200)
    print(f"Found {len(sections)} sections/chunks")

    embedder = EmbeddingProvider()
    chroma = ChromaDB()

    payload = []
    for i, sec in enumerate(tqdm(sections)):
        emb = embedder.embed_text(sec)
        payload.append({
            "section_id": f"{doc_id or os.path.basename(pdf_path)}__{i}",
            "title": f"Section {i}",
            "text": sec,
            "embedding": emb,
            "doc_id": doc_id or os.path.basename(pdf_path),
            "doc_title": os.path.basename(pdf_path),
            "last_updated": None,
            "metadata": {}
        })

    chroma.add_guideline_sections(payload)
    chroma.persist()
    print("S&T guideline sections added to Chroma.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/create_sandt.py path/to/guidelines.pdf [doc_id]"); sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
