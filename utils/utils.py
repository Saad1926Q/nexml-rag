from typing import Dict, Any, List
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from transformers import CLIPProcessor, CLIPModel
import torch
from sentence_transformers import CrossEncoder

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


def query_collection(collection: Any, query_embedding: List[float], query_text: str, n_results: int = 5) -> Dict[str, Any]:
    """
    Query a ChromaDB collection with an embedding vector AND Rerank with text.

    Args:
        collection: ChromaDB collection to query
        query_embedding: Embedding vector for the initial broad search
        query_text: Raw text string for the Reranker (Cross-Encoder)
        n_results: Number of final results to return
    """
    
    # FETCH BROADLY (Fetch 4x the requested amount to ensure recall)
    fetch_k = n_results * 4
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"]
    )

   
    if not results['documents'] or not results['documents'][0]:
        return results

   
    retrieved_docs = results['documents'][0]
    retrieved_metas = results['metadatas'][0]
    retrieved_ids = results['ids'][0]

   
    pairs = [[query_text, doc_text] for doc_text in retrieved_docs]

    # 3. PREDICT RELEVANCE SCORES
    scores = reranker_model.predict(pairs)

   
    zipped_results = list(zip(retrieved_ids, retrieved_docs, retrieved_metas, scores))
    
    # Sort by score (index 3)
    ranked_results = sorted(zipped_results, key=lambda x: x[3], reverse=True)

    top_results = ranked_results[:n_results]

    # 6. RECONSTRUCT CHROMA FORMAT
    
    final_ids = []
    final_docs = []
    final_metas = []
    final_distances = [] # We will put the Rerank Score here instead of distance

    for r_id, r_doc, r_meta, r_score in top_results:
        final_ids.append(r_id)
        final_docs.append(r_doc)
        
        r_meta['rerank_score'] = float(r_score)
        final_metas.append(r_meta)
        final_distances.append(float(r_score))

    return {
        "ids": [final_ids],
        "documents": [final_docs],
        "metadatas": [final_metas],
        "distances": [final_distances] # These are now Relevance Scores, not cosine dist
    }


# def rules_storage(doc, embeddings):
#     docs = file_loader(doc)
#     chunks = chunk_data(doc)
    
    