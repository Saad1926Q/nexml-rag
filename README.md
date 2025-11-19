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
```

## Initialization (First Run Only)

```bash
uv run app/vector_db.py
uv run scripts/create_vector_db.py
```

## Running the FastAPI Server

```bash
uv run uvicorn main:app --reload
```
