import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from app.vector_db import proposals_collection, guidelines_collection
from app.prompts import (
    NOVELTY_ANALYSIS_PROMPT,
    COMPLIANCE_CHECK_PROMPT,
    FINAL_EVALUATION_PROMPT,
    TALK2PROPOSAL_PROMPT
)
from utils.utils import chunk_text_and_generate_embeddings, query_collection



# GROQ_API_KEY = os.getenv('GROQ_API_KEY')
llm = ChatGroq(
    # groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    max_tokens=4096
)


async def check_novelty(proposal_text: str) -> str:
    """
    Check novelty of a proposal by comparing with similar past proposals.
    """
    proposal_doc = Document(page_content=proposal_text)
    _, embeddings_model = chunk_text_and_generate_embeddings([proposal_doc])
    query_embedding = embeddings_model.embed_query(proposal_text)

    print("Embedding type:", type(query_embedding))

    results = query_collection(proposals_collection, query_embedding, proposal_text, n_results=5)

    context_text = ""
    for i in range(len(results['documents'][0])):
        doc_text = results['documents'][0][i]
        metadata = results['metadatas'][0][i]

        context_text += f"\n{'='*80}\n"
        context_text += f"Proposal ID: {metadata.get('proposal_id', 'N/A')}\n"
        context_text += f"Title: {metadata.get('title', 'N/A')}\n"
        context_text += f"PI: {metadata.get('pi_name', 'N/A')}\n"
        context_text += f"Institution: {metadata.get('institution', 'N/A')}\n"
        context_text += f"Research Area: {metadata.get('research_area', 'N/A')}\n"
        context_text += f"\nContent:\n{doc_text}\n"
        context_text += f"{'='*80}\n"

    novelty_prompt = PromptTemplate(
        template=NOVELTY_ANALYSIS_PROMPT,
        input_variables=['proposal', 'context']
    )

    final_prompt = novelty_prompt.invoke({
        "proposal": proposal_text,
        "context": context_text
    })

    response = llm.invoke(final_prompt)

    return response.content


async def check_compliance(proposal_text: str) -> str:
    """
    Check compliance of a proposal with S&T guidelines.
    """

    proposal_doc = Document(page_content=proposal_text)
    _, embeddings_model = chunk_text_and_generate_embeddings([proposal_doc])
    query_embedding = embeddings_model.embed_query(proposal_text)

    results = query_collection(guidelines_collection, query_embedding, proposal_text, n_results=5)

    context_text = ""
    for i in range(len(results['documents'][0])):
        doc_text = results['documents'][0][i]
        metadata = results['metadatas'][0][i]

        context_text += f"\n{'='*80}\n"
        context_text += f"Guideline {i+1}:\n"
        context_text += f"Section ID: {metadata.get('section_id', 'N/A')}\n"
        context_text += f"Title: {metadata.get('title', 'N/A')}\n"
        context_text += f"Document: {metadata.get('doc_title', 'N/A')}\n"
        context_text += f"\nContent:\n{doc_text}\n"
        context_text += f"{'='*80}\n"

    compliance_prompt = PromptTemplate(
        template=COMPLIANCE_CHECK_PROMPT,
        input_variables=['proposal', 'context']
    )

    final_prompt = compliance_prompt.invoke({
        "proposal": proposal_text,
        "context": context_text
    })

    response = llm.invoke(final_prompt)

    return response.content

async def final_evaluation(proposal_text: str, novelty: str, compliance: str) -> str:
    """
    Perform detailed evaluation of a proposal based on novelty and compliance assessments.
    Returns a text summary (LLM content). Optionally returns a small JSON summary embedded in the text.
    """

    evaluation_prompt = PromptTemplate(
        template=FINAL_EVALUATION_PROMPT,
        input_variables=['proposal', 'novelty', 'compliance']
    )


    final_prompt = evaluation_prompt.invoke({
        "proposal": proposal_text,
        "novelty": novelty,
        "compliance": compliance
    })

    
    response = llm.invoke(final_prompt)
    
    return response.content




