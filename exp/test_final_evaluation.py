# tests/test_final_evaluation.py
import asyncio
import os
import sys

# make sure repo root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.doc_extractor import extract_text_images_tables
from app.llm import check_novelty, check_compliance, final_evaluation

async def run_test():
    # 1) Use a sample PDF you already have in documents/
    sample_pdf = r"C:\Users\Ariba\Documents\RAG\nexml-rag\documents\NACCER_2024_RD_8535.pdf"
    docs = extract_text_images_tables(sample_pdf)

    # extract_text_images_tables returns either a list or a tuple depending on your implementation
    # if it returns (doc_list, img_emb) handle accordingly:
    if isinstance(docs, tuple):
        doc_list = docs[0]
    else:
        doc_list = docs

    proposal_text = doc_list[0].page_content
    print("Proposal text length:", len(proposal_text))

    # 2) Run novelty & compliance
    novelty = await check_novelty(proposal_text)
    print("----- NOVELTY -----")
    print(novelty[:1000])   # print first 1000 chars for brevity

    compliance = await check_compliance(proposal_text)
    print("----- COMPLIANCE -----")
    print(compliance[:1000])

    # 3) Run final evaluation
    final_eval = await final_evaluation(proposal_text, novelty, compliance)
    print("----- FINAL EVALUATION -----")
    print(final_eval[:2000])  # print up to first 2000 chars

if __name__ == "__main__":
    asyncio.run(run_test())
