import os
from scripts.guidelines_collection import GROQ_API_KEY, vectorstore
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    max_tokens=4096
)
question = "tell me about the budget allocation guidelines or something related to the s and t guideleines "
retriever_docs = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retriever_docs)


prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided document context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
final_prompt = prompt.invoke({"context": context_text, "question": question})

answer = llm.invoke(final_prompt)
print(answer.content)
