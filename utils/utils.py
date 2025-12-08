from typing import Dict, Any, List
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch
from sentence_transformers import CrossEncoder
from scripts.doc_extractor import extract_text_images_tables
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import Chroma
from app.vector_db import talk2proposal_collection, proposals_collection
from app.prompts import SCORE_PROMPT
from langchain_core.prompts import PromptTemplate
from utils.schema import Score
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
import requests
import re


load_dotenv()
SUPERMEMORY_API_KEY = os.getenv('SUPERMEMORY_API_KEY')
EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_ID")
IMAGE_EMBEDDING_MODEL = os.getenv("CLIP_MODEL")

    
clip_model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device) 


def extract_abstract(text: str) -> str:
    pattern = r'abstract(.*?)(?=keywords:)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

    if match:
        return match.group(1).strip()
    return ""

def chunk_text_and_generate_embeddings(docs):
    
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    text_splitter = SemanticChunker(
        embeddings=embeddings_model
    )
    doc = text_splitter.split_documents(docs)
    return doc, embeddings_model

def get_image_embeddings(image):
    inputs = clip_processor(images=image, return_tensors="pt")

    # Generate the embedding vector
    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])


    embedding = image_features.cpu().numpy().tolist()[0]
    return embedding


def rerank(query: str, results: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    """
    Performs 'Rank CoT' retrieval:
    1. Takes initial results from ChromaDB.
    2. Reranks them using the CrossEncoder.
    3. Returns the top_k most relevant results.
    """
    if not results['documents'][0]:
        return results

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    pairs = [[query, doc] for doc in documents]
    scores = reranker_model.predict(pairs)

    ranked = sorted(zip(documents, metadatas, distances, scores), key=lambda x: x[3], reverse=True)

    final_docs = []
    final_metas = []
    final_dists = []

    for doc, meta, dist, score in ranked[:top_k]:
        meta['relevance_score'] = float(score)
        final_docs.append(doc)
        final_metas.append(meta)
        final_dists.append(dist)

    return {
        'documents': [final_docs],
        'metadatas': [final_metas],
        'distances': [final_dists]
    }


def query_collection(collection: Any, query_embedding: List[float], query_text: str,
                     n_results: int = 5, fetch_k: int = 10) -> Dict[str, Any]:
    """
    Query a ChromaDB collection with an embedding vector and reranking.

    Args:
        collection: ChromaDB collection to query
        query_embedding: Embedding vector for the query
        query_text: Original query text for reranking
        n_results: Number of results to return (default: 5)
        fetch_k: Number of initial results to fetch before reranking (default: 20)

    Returns:
        Dictionary containing query results with documents, metadatas, and distances
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"]
    )

    results = rerank(query_text, results, top_k=n_results)

    return results


def store_proposal_for_chat(file_path: str):
    """
    Process and store a proposal document in the talk2proposal collection.
    Clears existing collection before adding new proposal.
    """

    existing_items = talk2proposal_collection.get()  # Get previously stored embeddings and their associate data from the talk2proposal_collection.
    if existing_items['ids']:
        talk2proposal_collection.delete(ids=existing_items['ids'])

    text, _ = extract_text_images_tables(file_path=file_path)
    chunked_docs, embeddings_model = chunk_text_and_generate_embeddings(text)


    ids = []
    documents = []
    metadatas = []

    for i, doc in enumerate(chunked_docs):
        ids.append(f"proposal_chunk_{i}")
        documents.append(doc.page_content)
        metadatas.append(doc.metadata)

    all_embeddings = embeddings_model.embed_documents([doc.page_content for doc in chunked_docs])

    print(f"\nAdding {len(chunked_docs)} proposal chunks to ChromaDB collection...")
    talk2proposal_collection.add(
        ids=ids,
        documents=documents,
        embeddings=all_embeddings,
        metadatas = metadatas
    )
    


def score(proposal_text, context_text, answer_text, llm):
    parser = PydanticOutputParser(pydantic_object= Score)

    response_score = PromptTemplate(
        template = SCORE_PROMPT,
        input_variables = ['question', 'context', 'answer'],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = response_score | llm | parser

    score_result = chain.invoke({
    "question": proposal_text,
    "context": context_text,
    "answer": answer_text
})
    return score_result


def save_file(tmp_file_path, metadata):
    print("save file started")
    doc,_ = extract_text_images_tables(tmp_file_path)
    chunked_docs, embeddings_model = chunk_text_and_generate_embeddings(doc)
    ids = []
    documents = []
    metadatas = []
    for i, doc in enumerate(tqdm(chunked_docs, desc="Preparing Proposal for ChromaDB")):
        ids.append(f"{tmp_file_path}_{i}")
        documents.append(doc.page_content)
        
        chunk_meta = metadata.model_dump() 
        metadatas.append(chunk_meta)

    print("\nGenerating embeddings for the proposal chunks...")
    all_embeddings = embeddings_model.embed_documents([doc.page_content for doc in chunked_docs])
    proposals_collection.add(
    ids=ids,
    documents=documents,
    embeddings=all_embeddings,
    metadatas=metadatas
)

def delete_memory():
    
    url = "https://api.supermemory.ai/v3/documents/bulk"

    payload = { "containerTags": ["Talk_2_Proposal"] }
    headers = {
        "Authorization": f"Bearer {SUPERMEMORY_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.delete(url, json=payload, headers=headers)

    return response.text

    