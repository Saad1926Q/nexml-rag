import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from app.vector_db import proposals_collection, guidelines_collection
from utils.utils import chunk_text_and_generate_embeddings, query_collection

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
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

    results = query_collection(proposals_collection, query_embedding, n_results=5)

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
        template="""You are an expert research evaluator for NACCER.

Your task is to carry out a NOVELTY ANALYSIS for the following research proposal.

**CURRENT PROPOSAL TO EVALUATE:**
{proposal}

**SIMILAR PROPOSALS SUBMITTED IN THE PAST:**
{context}

**INSTRUCTIONS:**
Analyze the novelty of the current proposal by comparing it with the similar past proposals above.

Provide your analysis covering:
1. Key similarities with past proposals
2. Novel aspects of the current proposal
3. Overall novelty assessment (High/Medium/Low)

**YOUR ANALYSIS:**
""",
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

    results = query_collection(guidelines_collection, query_embedding, n_results=5)

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
        template="""You are an expert research evaluator for NACCER.

Your task is to assess whether the following research proposal COMPLIES with the S&T (Science & Technology) Guidelines.

**CURRENT PROPOSAL TO EVALUATE:**
{proposal}

**RELEVANT S&T GUIDELINES:**
{context}

**INSTRUCTIONS:**
Carefully assess the proposal's compliance with each guideline provided above.

Provide your analysis covering:
1. Which guidelines the proposal COMPLIES with (include guideline number and reasoning)
2. Which guidelines the proposal DOES NOT COMPLY with (include guideline number and reasoning)
3. Overall compliance assessment (Fully Compliant/Partially Compliant/Non-Compliant)

**YOUR ASSESSMENT:**
""",
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
    """

    evaluation_prompt = PromptTemplate(
        template="""You are an expert research evaluator for NACCER.

Your task is to carry out a DETAILED EVALUATION of the following research proposal.

**CURRENT PROPOSAL:**
{proposal}

**NOVELTY ASSESSMENT:**
{novelty}

**COMPLIANCE WITH S&T GUIDELINES:**
{compliance}

**INSTRUCTIONS:**
Based on the proposal, novelty assessment, and compliance analysis above, evaluate the proposal on the following aspects. For each aspect, provide a score out of 10 along with detailed reasoning.

Evaluate on these aspects:
1. **Budget** - Appropriateness and justification of budget allocation
2. **Technical Novelty** - Originality and innovation of the proposed research
3. **Technical Feasibility** - Practicality and achievability of the proposed methods
4. **Expertise** - Qualifications and capability of the research team
5. **Compliance with Guidelines** - Adherence to S&T guidelines and requirements
6. **Industry Relevance** - Practical applications and industrial impact
7. **Scalability** - Potential for scaling the solution
8. **Sustainability** - Long-term viability and environmental considerations
9. **Impact** - Overall potential impact on the field and society

For each aspect, provide:
- Score (X/10)
- Detailed reasoning

Finally, provide:
- **Overall Final Score** (Average of all aspects)
- **Summary** of the proposal's strengths and weaknesses

**YOUR DETAILED EVALUATION:**
""",
        input_variables=['proposal', 'novelty', 'compliance']
    )

    pass