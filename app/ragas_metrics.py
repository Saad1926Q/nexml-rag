import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    max_tokens=4096,
    temperature=0
)

parser = StrOutputParser()


from app.prompts import (
    FAITHFULNESS_PROMPT,
    CONTEXT_PRECISION_PROMPT,
    CONTEXT_RECALL_PROMPT
)

def get_llm_score(prompt_text: str) -> float:
    """Get score from LLM (expects response to be 0-1 number)"""
    try:
        response = llm.invoke(prompt_text)
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    except:
        return 0.5

def faithfulness_score(response: str, contexts: list) -> float:
    """Measure: Does response only use facts from contexts?"""
    if not response or not contexts:
        return 0.0
    
    context_text = "\n".join(contexts) if isinstance(contexts, list) else contexts
    
    prompt = PromptTemplate(
        template=FAITHFULNESS_PROMPT,
        input_variables=['response', 'context']
    )
    
    chain = prompt | llm | parser
    result = chain.invoke({
        "response": response,
        "context": context_text
    })
    
    try:
        return float(result.strip())
    except:
        return 0.5

def context_precision_score(question: str, contexts: list) -> float:
    """Measure: What % of retrieved contexts are relevant to question?"""
    if not question or not contexts:
        return 0.0
    
    context_text = "\n".join(contexts) if isinstance(contexts, list) else contexts
    
    prompt = PromptTemplate(
        template=CONTEXT_PRECISION_PROMPT,
        input_variables=['question', 'context']
    )
    
    chain = prompt | llm | parser
    result = chain.invoke({
        "question": question,
        "context": context_text
    })
    
    try:
        return float(result.strip())
    except:
        return 0.5

def context_recall_score(question: str, ground_truth: str, contexts: list) -> float:
    """Measure: Were all necessary contexts retrieved?"""
    if not question or not ground_truth or not contexts:
        return 0.0
    
    context_text = "\n".join(contexts) if isinstance(contexts, list) else contexts
    
    prompt = PromptTemplate(
        template=CONTEXT_RECALL_PROMPT,
        input_variables=['question', 'ground_truth', 'context']
    )
    
    chain = prompt | llm | parser
    result = chain.invoke({
        "question": question,
        "ground_truth": ground_truth,
        "context": context_text
    })
    
    try:
        return float(result.strip())
    except:
        return 0.5
