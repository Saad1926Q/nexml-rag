import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from app.vector_db import proposals_collection,guidelines_collection, budget_collection
from utils.utils import chunk_text_and_generate_embeddings
from scripts.doc_extractor import extract_text_images_tables
from langchain_core.documents import Document
from tqdm import tqdm

df = pd.read_csv('documents/naccer_proposals_100_cleaned.csv')

print(f"Loading {len(df)} proposals into vector database...")

ids = []
documents = []
metadatas = []

# Create Document objects from CSV
doc_list = []
for index, row in df.iterrows():
    page_content = (
        f"Proposal Title: {row['Title']}\n"
        f"Principal Investigator: {row['PI_Name']}\n"
        f"Institution: {row['Institution']}\n"
        f"Research Area: {row['Research_Area']}\n"
        f"Keywords: {row['Keywords']}\n"
        f"Abstract: {row['Abstract']}"
    )

    doc = Document(
        page_content=page_content,
        metadata={
            'proposal_id': str(row['Proposal_ID']),
            'title': row['Title'],
            'pi_name': row['PI_Name'],
            'institution': row['Institution'],
            'research_area': row['Research_Area'],
            'keywords': row['Keywords']
        }
    )
    doc_list.append(doc)

# Chunk and generate embeddings
print("Chunking documents and generating embeddings...")
chunked_docs, embeddings_model = chunk_text_and_generate_embeddings(doc_list)

# Prepare data for ChromaDB
for i, doc in enumerate(tqdm(chunked_docs, desc="Preparing for ChromaDB")):
    ids.append(f"proposal_{i}")
    documents.append(doc.page_content)
    metadatas.append(doc.metadata)

print("\nGenerating embeddings for all chunks...")
all_embeddings = embeddings_model.embed_documents([doc.page_content for doc in chunked_docs])

print("\nAdding proposals to ChromaDB collection...")
proposals_collection.add(
    ids=ids,
    documents=documents,
    embeddings=all_embeddings,
    metadatas=metadatas
)

print(f"\n Successfully loaded {len(df)} proposals into the vector database!")
print(f"Collection name: naccer_proposals")
print(f"Total items in collection: {proposals_collection.count()}")


print("\n" + "="*50)
print("Processing Guidelines PDF...")
print("="*50)


#------------------------GUIDELINES DB----------------------------------------------------
guidlines_list,_ = extract_text_images_tables("documents/S&T-Guidelines-MoC.pdf")

print("Chunking guidelines and generating embeddings...")
chunked_docs, embeddings_model = chunk_text_and_generate_embeddings(guidlines_list)

ids = []
documents = []
metadatas = []

for i, doc in enumerate(tqdm(chunked_docs, desc="Preparing Guidelines for ChromaDB")):
    ids.append(f"guideline_{i}")
    documents.append(doc.page_content)
    metadatas.append(doc.metadata)

print("\nGenerating embeddings for all guideline chunks...")
all_embeddings = embeddings_model.embed_documents([doc.page_content for doc in chunked_docs])

print("\nAdding guidelines to ChromaDB collection...")
guidelines_collection.add(
    ids=ids,
    documents=documents,
    embeddings=all_embeddings,
    metadatas=metadatas
)

print(f"\nSuccessfully loaded guidelines into the vector database!")
print(f"Total items in collection: {guidelines_collection.count()}")


print(metadatas)
#----------------------BUDGET VECTOR DB-------------------------------------
budget_list,_ = extract_text_images_tables("documents/S&T Budget.pdf")

print("Chunking guidelines and generating embeddings...")
chunked_docs, embeddings_model = chunk_text_and_generate_embeddings(budget_list)

ids = []
documents = []
metadatas = []

for i, doc in enumerate(tqdm(chunked_docs, desc="Preparing Budget Guidelines for ChromaDB")):
    ids.append(f"Budget_guideline_{i}")
    documents.append(doc.page_content)
    metadatas.append(doc.metadata)

print("\nGenerating embeddings for all budget guideline chunks...")
all_embeddings = embeddings_model.embed_documents([doc.page_content for doc in chunked_docs])

print("\nAdding budget guidelines to ChromaDB collection...")
budget_collection.add(
    ids=ids,
    documents=documents,
    embeddings=all_embeddings,
    metadatas=metadatas
)

print(f"\nSuccessfully loaded budget guidelines into the vector database!")
print(f"Total items in collection: {budget_collection.count()}")
