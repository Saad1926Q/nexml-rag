# scripts/cli_chat.py
"""
Terminal (CLI) chat for ephemeral proposal QA.

Usage:
    python scripts/cli_chat.py --pdf /path/to/proposal.pdf

Behaviour:
 - Creates an ephemeral collection (session_id)
 - Adds the proposal chunks to the collection
 - Lets you type questions in terminal, gets answers from the LLM grounded
   in the ephemeral collection contents.
 - Type 'exit' or Ctrl+C to finish; the script will delete the ephemeral collection.
"""

import argparse
import asyncio
import signal
import sys
from typing import Optional

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from scripts.doc_extractor import extract_text_images_tables
from app.session_manager import create_session_with_document, get_session, end_session
from utils.utils import chunk_text_and_generate_embeddings, query_collection

import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment. Set it in .env")

# Configure LLM (same as your other scripts)
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    max_tokens=2048
)


def build_answer_prompt(context: str, question: str) -> str:
    prompt_template = PromptTemplate(
        template="""
You are an assistant that answers questions using ONLY the provided context excerpts from a single proposal. 
If the context does not contain the answer, reply exactly: "I don't know (not in the proposal)."

Context:
{context}

Question: {question}

Answer (concise, grounded in context):
""",
        input_variables=["context", "question"]
    )
    return prompt_template.invoke({"context": context, "question": question})


def safe_input(prompt_text: str) -> Optional[str]:
    try:
        return input(prompt_text)
    except EOFError:
        return None
    except KeyboardInterrupt:
        return None


def embed_query_threadsafe(embeddings_model, text):
    # embeddings_model.embed_query is likely blocking; run in thread wrapper when needed
    return embeddings_model.embed_query(text)


def llm_invoke_threadsafe(prompt_text: str):
    # llm.invoke is likely blocking; run in a background thread when called via asyncio.to_thread
    return llm.invoke(prompt_text)


def interactive_loop(session_id: str):
    session = get_session(session_id)
    if not session:
        print("Session not found. Exiting.")
        return

    embeddings_model = session["embeddings_model"]
    collection = session["collection"]

    print("\nChat ready. Type questions and press Enter.")
    print("Type 'exit' (without quotes) to finish and delete ephemeral session.\n")

    try:
        while True:
            q = safe_input("Question> ")
            if q is None:
                print("\n(EOF/interrupt received)")
                break
            q = q.strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                print("Ending session...")
                break

            # 1) embed query (blocking) — run in a thread
            query_embedding = asyncio.run(asyncio.to_thread(embed_query_threadsafe, embeddings_model, q))

            # 2) query collection + rerank (blocking) — run in thread
            results = asyncio.run(asyncio.to_thread(query_collection, collection, query_embedding, q, 5))

            docs = results.get("documents", [[]])[0]
            if docs:
                context_text = "\n\n---\n\n".join(docs)
            else:
                context_text = ""

            final_prompt = build_answer_prompt(context_text, q)

            # 3) call llm (blocking) run in thread
            llm_response = asyncio.run(asyncio.to_thread(llm_invoke_threadsafe, final_prompt))
            answer = getattr(llm_response, "content", str(llm_response)).strip()

            print("\nAnswer:\n")
            print(answer)
            print("\n" + ("-" * 60) + "\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user, ending session.")
    finally:
        try:
            ended = end_session(session_id)
            if ended:
                print(f"Ephemeral session {session_id} deleted.")
            else:
                print(f"Could not delete ephemeral session {session_id} (may have been removed).")
        except Exception as e:
            print("Error while ending session:", e)


def main():
    parser = argparse.ArgumentParser(description="CLI ephemeral proposal chat")
    parser.add_argument("--pdf", required=True, help="Path to proposal PDF")
    args = parser.parse_args()

    pdf_path = args.pdf
    if not os.path.exists(pdf_path):
        print("ERROR: PDF path does not exist:", pdf_path)
        sys.exit(1)

    # Extract text
    print("Extracting proposal text from PDF...")
    docs = extract_text_images_tables(pdf_path)
    # extract_text_images_tables returns a list of Document objects in your repo
    if isinstance(docs, tuple):
        doc_list = docs[0]
    else:
        doc_list = docs

    if not doc_list:
        print("Failed to extract proposal from PDF.")
        sys.exit(1)

    proposal_doc = doc_list[0]

    # Create ephemeral session
    print("Creating ephemeral session and uploading chunks...")
    session_info = create_session_with_document(proposal_doc)
    session_id = session_info["session_id"]
    chunks_added = session_info.get("chunks_added", 0)

    print(f"Session created: {session_id} (chunks added: {chunks_added})")

    # Enter interactive loop
    interactive_loop(session_id)


if __name__ == "__main__":
    main()
