import os
import sys
import json
import time
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextRelevance, ContextRecall, ResponseRelevancy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

with open('documents/ragas_eval_filled.json', 'r') as file:
    eval_data = json.load(file)

eval_dataset = Dataset.from_list(eval_data)

llm = ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model_name="llama-3.1-8b-instant",
    max_tokens=4096,
    temperature=0
)

embedding_model = HuggingFaceEmbeddings(model_name=os.getenv("TEXT_EMBEDDING_ID"))

faithfulness_metric = Faithfulness()
context_relevance_metric = ContextRelevance()
context_recall_metric = ContextRecall()
response_relevancy_metric = ResponseRelevancy()

batch_size = 4
all_results = []

for i in tqdm(range(0, len(eval_dataset), batch_size)):
    batch = eval_dataset.select(range(i, min(i + batch_size, len(eval_dataset))))

    result = evaluate(
        dataset=batch,
        metrics=[
            faithfulness_metric,
            context_relevance_metric,
            context_recall_metric,
            response_relevancy_metric
        ],
        llm=llm,
        embeddings=embedding_model,
        raise_exceptions=False
    )

    all_results.append(result.to_pandas())

    time.sleep(60)

df_results = pd.concat(all_results, ignore_index=True)

df_results.to_csv("documents/eval_results.csv")
print(df_results)
