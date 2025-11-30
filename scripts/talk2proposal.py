import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import chunk_text_and_generate_embeddings
from scripts.doc_extractor import extract_text_images_tables
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from app.prompts import TALK2PROPOSAL_PROMPT
from collections import deque
load_dotenv()


llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    max_tokens=4096
)

def talk2proposal(file_path: str):
    text, img_emb = extract_text_images_tables(file_path=file_path)
    text_emb, e= chunk_text_and_generate_embeddings(text)

    persist_directory = "./ChromaDB"
    text_emb = filter_complex_metadata(text_emb)
    vectorstore = Chroma.from_documents(text_emb,
                                        embedding=e,
                                        collection_name="talk2proposal_collection",
                                        
                                        persist_directory=persist_directory)
    
    return vectorstore

vectorstore = talk2proposal('documents/NACCER_2023_RD_8968.pdf')

#TODO: Make it have memory of prev conv
while True:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    question = input("Enter a question which you want to ask from the user regarding the research? ")

    if question =='quit':
        break
    
    retriever_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retriever_docs)


    prompt = PromptTemplate(
        template=TALK2PROPOSAL_PROMPT,
        input_variables = ['context', 'question']
    )
   
    
 
    final_prompt = prompt.invoke({"context": context_text, "question": question})
    
    answer = llm.invoke(final_prompt)
    print(answer.content)

    

