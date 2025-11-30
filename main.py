from fastapi import FastAPI, UploadFile, File, status
from typing import Dict
import tempfile
import os
from app.llm import check_novelty, check_compliance, final_evaluation
from scripts import talk2proposal
from scripts.doc_extractor import extract_text_images_tables
import asyncio
from fastapi.responses import JSONResponse

from app.talk import router as talk_router, start_cleanup_background_task, stop_cleanup_background_task
from utils.utils import talk2proposal_collection

app = FastAPI(title="NaCCER Auto-Evaluation")
app.include_router(talk_router, prefix="/talk")


@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)) -> Dict[str, str]:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        doc_list = extract_text_images_tables(tmp_file_path)
        proposal_text = doc_list[0].page_content
        
        novelty_res = await check_novelty(proposal_text)
        compliance_res = await check_compliance(proposal_text)
        final_res = await final_evaluation(proposal_text, novelty_res, compliance_res)

        return {
            "novelty_assessment": novelty_res,
            "s_and_t_assessment": compliance_res,
            "evaluation": final_res
        }

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


@app.post("/upload_talk2proposal")
async def evaluate(file: UploadFile = File(...)) -> JSONResponse:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        talk2proposal_collection(tmp_file_path)
        
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Resource created successfully!"})

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


@app.post("/talk2proposal_chat")
async def chat(question: str) -> str:
    vectore
    
@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_event_loop()
    start_cleanup_background_task(loop)


@app.on_event("shutdown")
async def on_shutdown():
    stop_cleanup_background_task()