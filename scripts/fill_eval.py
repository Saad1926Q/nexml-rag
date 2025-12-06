import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from utils.utils import chunk_text_and_generate_embeddings, query_collection
from app.vector_db import proposals_collection
from app.prompts import TALK2PROPOSAL_PROMPT

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model_name="llama-3.1-8b-instant",
    max_tokens=4096,
    temperature=0
)

with open('documents/ragas_eval.json', 'r') as f:
    eval_data = json.load(f)

talk2proposal_prompt = PromptTemplate(
    template=TALK2PROPOSAL_PROMPT,
    input_variables=['question', 'context']
)
chain = talk2proposal_prompt | llm

for item in eval_data:
    question = item['question']

    question_doc = Document(page_content=question)
    _, embeddings_model = chunk_text_and_generate_embeddings([question_doc])
    query_embedding = embeddings_model.embed_query(question)

    results = query_collection(proposals_collection, query_embedding, question, n_results=5, fetch_k=20)

    context_chunks = []
    for i in range(len(results['documents'][0])):
        doc_text = results['documents'][0][i]
        metadata = results['metadatas'][0][i]

        context_text = f"\n{'='*80}\n"
        context_text += f"Proposal ID: {metadata.get('proposal_id', 'N/A')}\n"
        context_text += f"Title: {metadata.get('title', 'N/A')}\n"
        context_text += f"PI: {metadata.get('pi_name', 'N/A')}\n"
        context_text += f"Institution: {metadata.get('institution', 'N/A')}\n"
        context_text += f"Research Area: {metadata.get('research_area', 'N/A')}\n"
        context_text += f"\nContent:\n{doc_text}\n"
        context_text += f"{'='*80}\n"

        context_chunks.append(context_text)

    full_context = "\n".join(context_chunks)

    answer = chain.invoke({"question": question, "context": full_context})

    item['answer'] = answer.content
    item['contexts'] = full_context

with open('documents/ragas_eval_filled.json', 'w') as f:
    json.dump(eval_data, f, indent=4, ensure_ascii=False)