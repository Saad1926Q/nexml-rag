#!/usr/bin/env python3
import os, json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from app.ragas_metrics import faithfulness_score, context_precision_score, context_recall_score

load_dotenv()

with open("FIXED_ragas_eval_filled.json", encoding="cp1252") as f:
    data = json.load(f)

def normalize(sample):
    contexts = sample.get('contexts', [])
    if isinstance(contexts, str) and contexts.strip():
        contexts = [contexts]
    elif not contexts:
        contexts = []
    
    return {
        'user_input': sample.get('question', '').strip(),
        'response': sample.get('answer', '').strip(),
        'retrieved_contexts': contexts,
        'reference': sample.get('ground_truth', '').strip()
    }

data = [normalize(s) for s in data]
print(f"✓ Loaded {len(data)} samples")

empty_count = sum(1 for s in data if not s['response'] or not s['retrieved_contexts'])
print(f"✓ Valid samples: {len(data) - empty_count}/{len(data)}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("✓ Embeddings loaded")

def cosine_sim(a, b):
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def context_relevance_score(response, contexts):
    if not response or not contexts:
        return 0.0
    
    response_vec = np.array(embeddings.embed_query(response))
    context_vecs = [np.array(embeddings.embed_query(ctx)) for ctx in contexts if ctx.strip()]
    
    if not context_vecs:
        return 0.0
    
    similarities = [cosine_sim(response_vec, ctx_vec) for ctx_vec in context_vecs]
    normalized = [(sim + 1) / 2 for sim in similarities]
    return np.mean(normalized)

def answer_relevance_score(question, response):
    if not question or not response:
        return 0.0
    
    question_vec = np.array(embeddings.embed_query(question))
    response_vec = np.array(embeddings.embed_query(response))
    
    sim = cosine_sim(question_vec, response_vec)
    return (sim + 1) / 2

print("\n🔄 Computing metrics...")
results = []

for idx, sample in enumerate(data, 1):
    response = sample['response']
    contexts = sample['retrieved_contexts']
    question = sample['user_input']
    ground_truth = sample['reference']
    
    
    context_rel = context_relevance_score(response, contexts)
    answer_rel = answer_relevance_score(question, response)
    
    
    print(f"  Sample {idx}: Computing LLM metrics...", end=" ", flush=True)
    faithfulness = faithfulness_score(response, contexts)
    precision = context_precision_score(question, contexts)
    recall = context_recall_score(question, ground_truth, contexts)
    print(f"Done")
    
    results.append({
        'question': question,
        'response': response,
        'reference': ground_truth,
        'context_relevance': context_rel,
        'answer_relevance': answer_rel,
        'faithfulness': faithfulness,
        'context_precision': precision,
        'context_recall': recall
    })

df = pd.DataFrame(results)
df.to_csv("eval_results_simple.csv", index=False)

print(f"\n{'='*60}")
print(f"RESULTS SAVED: eval_results_simple1.csv")
print(f"{'='*60}")
print(f"\nMETRICS SUMMARY:")
print(f"  Context Relevance:   {df['context_relevance'].mean():.4f} ± {df['context_relevance'].std():.4f}")
print(f"  Answer Relevance:    {df['answer_relevance'].mean():.4f} ± {df['answer_relevance'].std():.4f}")
print(f"  Faithfulness:        {df['faithfulness'].mean():.4f} ± {df['faithfulness'].std():.4f}")
print(f"  Context Precision:   {df['context_precision'].mean():.4f} ± {df['context_precision'].std():.4f}")
print(f"  Context Recall:      {df['context_recall'].mean():.4f} ± {df['context_recall'].std():.4f}")
print(f"\nFirst 3 samples:")
print(df[['question', 'faithfulness', 'context_precision', 'context_recall']].head(3))
