from fastapi import FastAPI, UploadFile, File, status, Body
from typing import  Dict, Annotated
import tempfile
import os
import json

from app.llm import check_budget, check_novelty, check_compliance, final_evaluation, talk2proposal,check_coal_relevance
from scripts.doc_extractor import extract_text_images_tables
from fastapi.responses import JSONResponse
from utils.utils import store_proposal_for_chat, save_file, delete_memory,extract_abstract
from utils.schema import Assessment, EvaluationResponse, ProposalMetadata
app = FastAPI(title="NaCCER Auto-Evaluation")

@app.post("/check_relevance")
async def check_relevance(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        doc_list, _ = extract_text_images_tables(tmp_file_path)
        proposal_text = doc_list[0].page_content

        abstract = extract_abstract(proposal_text)

        response = await check_coal_relevance(abstract)

        return response

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

    

@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    """Get the evaluation JSON
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        doc_list,_ = extract_text_images_tables(tmp_file_path)
        proposal_text = doc_list[0].page_content
        
        novelty_res, novelty_score, p_id = await check_novelty(proposal_text)
        compliance_res, compliance_score = await check_compliance(proposal_text)
        # budget_res, budget_score = await check_budget(proposal_text)
        budget_res = await check_budget(proposal_text)
        final_res = await final_evaluation(proposal_text, novelty_res, compliance_res)

        return EvaluationResponse(
            novelty_assessment = Assessment(
                summary= novelty_res,
                score = novelty_score
            ),
            s_and_t_assessment = Assessment(
                summary=compliance_res,
                score = compliance_score
            ),
            # budget_assessment= Assessment(
            #     summary = budget_res,
            #     score = budget_score
            #     ),
            budget_assessment= budget_res,
            evaluation = final_res,
            proposal_ids = p_id
        )

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


@app.post("/proposal_save")
async def save_proposal(metadata: Annotated[str, Body(...)], file: UploadFile = File(...)) -> JSONResponse:
    """Saves Propsal in the Vector Database
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        metadata_obj = ProposalMetadata(**json.loads(metadata))
        save_file(tmp_file_path, metadata_obj)
    
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Resource Saved successfully!"})

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)



@app.post("/talk2proposal")
async def chat(question: str) -> Dict[str, str]:
    """
    Answer questions about uploaded proposals using RAG.
    """
    answer = await talk2proposal(question)
    return {"answer": answer}


@app.post("/talk2proposal/upload_proposal")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    """ 
    Upload Proposal before starting with the chat
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        store_proposal_for_chat(tmp_file_path)
        
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Resource created successfully!"})

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
    
@app.post("/talk2proposal/clear_memory")
async def clear_memory() -> JSONResponse:
    """
    Clears memory of the llm
    """
    delete_memory()
    return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Resource Deleted successfully!"})
