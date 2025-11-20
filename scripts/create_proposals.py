# scripts/create_proposals.py
import os, sys, json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from app.vector_db import ChromaDB
from helpers.chunker import simple_chunk_text
from scripts.embedding_provider import EmbeddingProvider

load_dotenv()

def load_csv(path: str):
    return pd.read_csv(path)

def main(csv_path: str, text_column: str = "text", id_column: str = "proposal_id"):
    df = load_csv(csv_path)
    print("Loaded", len(df), "rows")
    embedder = EmbeddingProvider()
    chroma = ChromaDB()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        proposal_id = str(row.get(id_column) or row.get("Proposal_ID") or row.get("id") or pd.NA)
        title = str(row.get("Title") or row.get("title") or "")
        authors = []
        if "Authors" in row and not pd.isna(row["Authors"]):
            authors = [a.strip() for a in str(row["Authors"]).split(";") if a.strip()]
        text = str(row.get(text_column) or row.get("Text") or "")
        chunks = simple_chunk_text(text, chunk_size=1200, overlap=200)
        for ch in chunks:
            ch["embedding"] = embedder.embed_text(ch["text"])
        doc_emb = embedder.embed_text(text[:2000])
        metadata = {
            "Research_Area": row.get("Research_Area"),
            "Status": row.get("Status"),
            "source_row": int(_)
        }
        chroma.add_proposal_chunks(proposal_id=proposal_id, title=title, authors=authors, chunks=chunks, metadata=metadata)
    chroma.persist()
    print("Proposals database populated.")
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/create_proposals.py path/to/proposals.csv"); sys.exit(1)
    main(sys.argv[1])
