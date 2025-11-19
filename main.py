from fastapi import FastAPI, UploadFile, File
from app.llm import check_novelty,check_compliance,final_evaluation


app = FastAPI(title="NaCCER Auto-Evaluation")

def extract_text_from_file_content(content):
    pass

@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    content = await file.read() #loads the entire file as a raw byte string

    text=extract_text_from_file_content(content)

    """
    Here add code to run novelty assessment,S&tguidelines assessment and evaluation

    and return  {
        "novelty_assessment": novelty_res,
        "s_and_t_assessment": compliance_res,
        "evaluation": final_res
    }

    """