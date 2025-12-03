# nexml-rag

RAG app built for SIH to automate and streamline the evaluation of research proposals.

## Project Setup (using uv)

```bash
# Install uv if you haven't already
pip install uv

# Sync dependencies
uv sync
```

## Environment Configuration

Create a `.env` file in the project root and add the following content:

```env
# Required: API Key for LLM (Groq)
GROQ_API_KEY="gsk_your_groq_api_key_here"
MODEL_NAME="llama3-70b-8192"
CHROMA_DB_PATH="./chroma_db_data"
CLIP_MODEL = "openai/clip-vit-base-patch32"
TEXT_EMBEDDING_ID = "sentence-transformers/clip-ViT-B-32"
```

## Initialization (First Run Only)

Setup vector database by populating it with proposals and guidelines:

```bash

uv run app/vector_db.py
uv run scripts/create_vector_db.py

```

## File Structure

```
nexml-rag/
├── app/                          # Core application code
│   ├── llm.py                    # LLM functions (novelty, compliance checks)
│   └── vector_db.py              # Vector database collections
├── scripts/                      # Setup and processing scripts
│   ├── doc_extractor.py          # PDF extraction using Docling
│   ├── create_vector_db.py       # Populate proposal database
│   └── guidelines_collection.py  # Load S&T guidelines
├── utils/                        # Utility functions
│   └── utils.py                  # Embeddings and chunking
├── exp/                          # Experiments and notebooks
│   ├── test_novelty_check.ipynb
│   └── test_compliance_check.ipynb
├── documents/                    # Input documents
│   ├── naccer_proposals_100_cleaned.csv
│   └── S&T-Guidelines-MoC.pdf
├── chroma_db_data/               # Vector database storage
└── main.py                       # FastAPI application
```


## Implemented
- Novelty Analysis 
- S&T Guidelines Compliance

## TODO
- Currently I have commented out the logic for handling images for simplicity
- Final evaluation score paprt pending
- Talk to proposal
- Arxiv feature
- check things for a better llm response with minimal hallucination
- regarding scores make sure that you always get a score
