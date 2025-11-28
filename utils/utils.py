from typing import Dict, Any, List
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from transformers import CLIPProcessor, CLIPModel
import torch
from sentence_transformers import CrossEncoder  
from langchain_core.documents import Document

load_dotenv()
EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_ID")
IMAGE_EMBEDDING_MODEL = os.getenv("CLIP_MODEL")


clip_model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL) 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device) 


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


def query_collection(collection: Any, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
    """
    Query a ChromaDB collection with an embedding vector.

    Args:
        collection: ChromaDB collection to query
        query_embedding: Embedding vector for the query
        n_results: Number of results to return (default: 5)

    Returns:
        Dictionary containing query results with documents, metadatas, and distances
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results


def smart_search(vectorstore, query: str, top_k: int = 5, fetch_k: int = 20) -> List[Document]:
    """
    Performs 'Rank CoT' retrieval: 
    1. Fetches a broad set of documents (fetch_k) using the vectorstore.
    2. Reranks them using the CrossEncoder.
    3. Returns the top_k most relevant Documents.
    
    Use this in llm.py instead of retriever.invoke()
    """
    
    initial_docs = vectorstore.similarity_search(query, k=fetch_k)
    
    if not initial_docs:
        return []

   
    pairs = [[query, doc.page_content] for doc in initial_docs]
 
    scores = reranker_model.predict(pairs)
    
    ranked_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    
    # 5. Extract top_k
    final_docs = []
    for doc, score in ranked_docs[:top_k]:
        # Optional: Save score to metadata for debugging
        doc.metadata['relevance_score'] = float(score)
        final_docs.append(doc)
        
    return final_docs


# def rules_storage(doc, embeddings):
#     docs = file_loader(doc)
#     chunks = chunk_data(doc)
    
    