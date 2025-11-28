from typing import Dict, Any, List
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from transformers import CLIPProcessor, CLIPModel
import torch

load_dotenv()
EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_ID")
IMAGE_EMBEDDING_MODEL = os.getenv("CLIP_MODEL")


clip_model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL) 


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


# def rules_storage(doc, embeddings):
#     docs = file_loader(doc)
#     chunks = chunk_data(doc)
    
    