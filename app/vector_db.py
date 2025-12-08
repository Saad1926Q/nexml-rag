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
budget_collection = client.get_or_create_collection(
    name="budget_guidelines",
    metadata={"hnsw:space": "cosine"}
)

proposals_collection = client.get_or_create_collection(
    name="naccer_proposals",
    metadata={"hnsw:space": "cosine"}
)

talk2proposal_collection = client.get_or_create_collection(
    name="talk2proposal_collection",
    metadata={"hnsw:space": "cosine"}
)