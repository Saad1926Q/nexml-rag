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
clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL) # type    : ignore


def chunk_text_and_generate_embeddings(docs):
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    text_splitter = SemanticChunker(
        embeddings=embeddings
    )
    doc = text_splitter.split_documents(docs)
    return doc, embeddings

def get_image_embeddings(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    
    # Generate the embedding vector
    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
    

    embedding = image_features.cpu().numpy().tolist()[0]
    return embedding


# def rules_storage(doc, embeddings):
#     docs = file_loader(doc)
#     chunks = chunk_data(doc)
    
    