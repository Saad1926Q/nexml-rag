from typing import Dict, Any, List
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from transformers import CLIPProcessor, CLIPModel
import torch
from sentence_transformers import CrossEncoder
from scripts.doc_extractor import extract_text_images_tables
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import Chroma
from app.vector_db import talk2proposal_collection
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
                     n_results: int = 5, fetch_k: int = 20) -> Dict[str, Any]:
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


def talk2proposal_vectorDB(file_path: str):
    text, img_emb = extract_text_images_tables(file_path=file_path)
    chunked_docs, embeddings_model = chunk_text_and_generate_embeddings(text)

    
    ids = []
    documents = []
    metadatas = []

    for i, doc in enumerate(chunked_docs):
        ids.append(f"guideline_{i}")
        documents.append(doc.page_content)
        

    all_embeddings = embeddings_model.embed_documents([doc.page_content for doc in chunked_docs])

    print("\nAdding guidelines to ChromaDB collection...")
    talk2proposal_collection.add(
        ids=ids,
        documents=documents,
        embeddings=all_embeddings,
    )
    




#TODO: Make it have memory of prev conv
vectorestore = talk2proposal('documents/NACCER_2023_RD_8968.pdf')
def talk2proposal(question):
    retriever = vectorestore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    question = input("Enter a question which you want to ask from the user regarding the research? ")


    
    retriever_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retriever_docs)


    prompt = PromptTemplate(
        template=TALK2PROPOSAL_PROMPT,
        input_variables = ['context', 'question']
    )
   
    
 
    final_prompt = prompt.invoke({"context": context_text, "question": question})
    
    answer = llm.invoke(final_prompt)
    print(answer.content)

    





# def rules_storage(doc, embeddings):
#     docs = file_loader(doc)
#     chunks = chunk_data(doc)
    
    