import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from app.vector_db import proposals_collection, guidelines_collection, talk2proposal_collection, budget_collection
from app.prompts import (
    BUDGET_CHECK_PROMPT,
    BUDGET_CONTEXT,
    NOVELTY_ANALYSIS_PROMPT,
    COMPLIANCE_CHECK_PROMPT,
    FINAL_EVALUATION_PROMPT,
    TALK2PROPOSAL_PROMPT
)
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from utils.utils import chunk_text_and_generate_embeddings, query_collection, score
from utils.schema import BudgetAnalysis, Score, EvaluationScore
from utils.schema import Score
from langchain_core.messages import SystemMessage, HumanMessage
from supermemory import Supermemory
from langchain_openai import ChatOpenAI
SUPERMEMORY_API_KEY = os.getenv('SUPERMEMORY_API_KEY')
client = Supermemory(
    api_key=SUPERMEMORY_API_KEY,
)


GROQ_API_KEY = os.getenv('GROQ_API_KEY')
# llm = ChatGroq(
#     groq_api_key=GROQ_API_KEY,
#     model_name="llama-3.1-8b-instant",
#     max_tokens=4096,
#     temperature= 0
# )
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature= 0
)



parser = StrOutputParser()

async def check_novelty(proposal_text: str) -> tuple[str,Score,set[str]]:
    """
    Check novelty of a proposal by comparing with similar past proposals.
    """
    proposal_doc = Document(page_content=proposal_text)
    _, embeddings_model = chunk_text_and_generate_embeddings([proposal_doc])
    query_embedding = embeddings_model.embed_query(proposal_text)

    print("Embedding type:", type(query_embedding))

    results = query_collection(proposals_collection, query_embedding, proposal_text, n_results=5, fetch_k=10)

    context_text = ""
    p_id = set()
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

        p_id.add(metadata.get('proposal_id', 'N/A'))
    novelty_prompt = PromptTemplate(
        template=NOVELTY_ANALYSIS_PROMPT,
        input_variables=['proposal', 'context']
    )
    
    chain = novelty_prompt | llm | parser
    response = chain.invoke({
        "proposal": proposal_text,
        "context": context_text
    })
    # final_response = "{}".format(response)
    response_score = score(proposal_text, context_text,response, llm)
    return response, response_score, p_id

async def check_compliance(proposal_text: str) -> tuple[str, Score]:
    """
    Check compliance of a proposal with S&T guidelines.
    """

    proposal_doc = Document(page_content=proposal_text)
    _, embeddings_model = chunk_text_and_generate_embeddings([proposal_doc])
    query_embedding = embeddings_model.embed_query(proposal_text)
    print("Embedding type S&T Guidelines:", type(query_embedding))
    results = query_collection(guidelines_collection, query_embedding, proposal_text, n_results=5,fetch_k=10)
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

    chain = compliance_prompt | llm | parser
    response = chain.invoke({
        "proposal": proposal_text,
        "context": context_text
    })


    compliance_score = score(proposal_text, context_text,response, llm)
    return response, compliance_score



async def final_evaluation(proposal_text: str, novelty: str, compliance: str) -> EvaluationScore:
    """
    Perform detailed evaluation of a proposal based on novelty and compliance assessments.
    Returns a text summary (LLM content). Optionally returns a small JSON summary embedded in the text.
    """
    parser = PydanticOutputParser(pydantic_object = EvaluationScore)
    evaluation_prompt = PromptTemplate(
        template=FINAL_EVALUATION_PROMPT,
        input_variables=['proposal', 'novelty', 'compliance'],
        partial_variables = {'format_description': parser.get_format_instructions()}
    )

    chain = evaluation_prompt | llm | parser
    response = chain.invoke({
        "proposal": proposal_text,
        "novelty": novelty,
        "compliance": compliance
    })


    return response


async def talk2proposal(question: str) -> str:
    """
    Answer questions about a proposal by retrieving relevant chunks and using LLM.
    """

    question_doc = Document(page_content=question)
    _, embeddings_model = chunk_text_and_generate_embeddings([question_doc])
    query_embedding = embeddings_model.embed_query(question)

    results = query_collection(talk2proposal_collection, query_embedding, question, n_results=5)
    
    
    context_text = ""
    for i in range(len(results['documents'][0])):
        doc_text = results['documents'][0][i]
        context_text += f"\n{'='*80}\n"
        context_text += f"Chunk {i+1}:\n"
        context_text += f"{doc_text}\n"
        context_text += f"{'='*80}\n"

    response = client.memories.add(
        content=context_text,
        container_tags=["Talk_2_Proposal"],
        metadata={
            "note_id": "Retrieved_Chunks",
        }
    )
    
    
    
    talk2proposal_prompt = PromptTemplate(
        template=TALK2PROPOSAL_PROMPT,
        input_variables=['question', 'context']
    )
    
    result = client.search.documents(
        q = question,
        container_tags= ["Talk_2_Proposal"],
        limit = 7
        
    )
    chain = talk2proposal_prompt | llm | parser
    response = chain.invoke({
        "question": question,
        "context": result
    })

    client.memories.add(
    content=f"[User_Query]:{question}\n\n[AI Message]:{response} ",
    container_tags =["Talk_2_Proposal"],
    metadata={
        "note_id": "User_123",
    }
)
    
    return response



#--------------BUDGET------------------------------------------
# async def check_budget(proposal_text: str) -> tuple[str, Score]:
async def check_budget(proposal_text: str) -> BudgetAnalysis:

    """
    Check compliance of a proposal with S&T guidelines.
    """

    # proposal_doc = Document(page_content=proposal_text)
    # _, embeddings_model = chunk_text_and_generate_embeddings([proposal_doc])
    # query_embedding = embeddings_model.embed_query(proposal_text)
    # print("Embedding type Budget Guidelines:", type(query_embedding))
    # results = query_collection(budget_collection, query_embedding, proposal_text, n_results=5,fetch_k=10)
    # context_text = ""
    # for i in range(len(results['documents'][0])):
    #     doc_text = results['documents'][0][i]
    #     metadata = results['metadatas'][0][i]

    #     context_text += f"\n{'='*80}\n"
    #     context_text += f"Guideline {i+1}:\n"
    #     context_text += f"Section ID: {metadata.get('section_id', 'N/A')}\n"
    #     context_text += f"Title: {metadata.get('title', 'N/A')}\n"
    #     context_text += f"Document: {metadata.get('doc_title', 'N/A')}\n"
    #     context_text += f"\nContent:\n{doc_text}\n"
    #     context_text += f"{'='*80}\n"
    parser = PydanticOutputParser(pydantic_object=BudgetAnalysis)
    compliance_prompt = PromptTemplate(
        template=BUDGET_CHECK_PROMPT,
        input_variables=['proposal', 'context'],
        partial_variables= {'format_description': parser.get_format_instructions()}
    )
    context_text = BUDGET_CONTEXT
    chain = compliance_prompt | llm | parser
    response = chain.invoke({
        "proposal": proposal_text,
        "context": context_text
    })


    # compliance_score = score(proposal_text, context_text, response, llm)
    # return response, compliance_score
    return response




# ---------------------- RELEVANCE -------------------------------------------



@tool
def mark_relevant(explanation:str)->str:
    """
      Mark the proposal abstract as RELEVANT to Ministry of Coal.

      Args:
          explanation: 2-3 sentences explaining why it's relevant
      """
    
    pass

@tool
def mark_irrelevant(explanation: str) -> str:
      """
      Mark the proposal abstract as NOT RELEVANT to Ministry of Coal.

      Args:
          explanation: 2-3 sentences explaining why it's not relevant
      """
      pass


relevance_tools = [mark_relevant, mark_irrelevant]
llm_with_relevance_tools = llm.bind_tools(relevance_tools,tool_choice="required")


async def check_coal_relevance(abstract: str) -> dict:
    system_prompt = """Evaluate if this research proposal is relevant to Ministry of Coal, India."""

    messages = [
          SystemMessage(content=system_prompt),
          HumanMessage(content=abstract)
      ]
    
    response = llm_with_relevance_tools.invoke(messages)

    tool_call = response.tool_calls[0]
    is_relevant = (tool_call["name"] == "mark_relevant")
    explanation = tool_call["args"]["explanation"]


    return {"is_relevant": is_relevant, "explanation": explanation}