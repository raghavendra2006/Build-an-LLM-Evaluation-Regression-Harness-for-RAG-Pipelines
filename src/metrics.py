import numpy as np
from typing import List, Dict, Any
from src.judge import LLMJudge

def calculate_recall(ground_truth_context_titles: List[str], retrieved_context_titles: List[str]) -> float:
    """Deterministic Recall@k calculation."""
    if not ground_truth_context_titles:
        return 1.0
    gt_set = set(ground_truth_context_titles)
    retrieved_set = set(retrieved_context_titles)
    matches = len(gt_set & retrieved_set)
    return float(round(matches / len(gt_set), 4))

def calculate_p95_latency(latencies: List[float]) -> float:
    """Calculate 95th percentile latency across all dataset samples."""
    if not latencies:
        return 0.0
    return float(round(np.percentile(latencies, 95), 2))

def evaluate_pipeline(pipeline, eval_set: List[Dict[str, Any]], judge: LLMJudge) -> List[Dict[str, Any]]:
    """Runs a RAG pipeline on eval_set and calculates question-by-question metrics matching exact JSON schema."""
    details = []
    for item in eval_set:
        question = item["question"]
        gt_answer = item["ground_truth_answer"]
        gt_titles = item["ground_truth_context_titles"]
        
        output = pipeline.retrieve_and_generate(question)
        
        gen_answer = output["answer"]
        retrieved_titles = output["context_titles"]
        retrieved_contexts = output["retrieved_contexts"]
        latency_ms = float(output["latency_ms"])
        
        recall_val = calculate_recall(gt_titles, retrieved_titles)
        correctness_val = int(judge.evaluate_correctness(question, gt_answer, gen_answer))
        groundedness_val = int(judge.evaluate_groundedness(question, retrieved_contexts, gen_answer))
        
        details.append({
            "id": item["id"],
            "generated_answer": gen_answer,
            "retrieved_context_titles": retrieved_titles,
            "metrics": {
                "recall": float(recall_val),
                "correctness": int(correctness_val),
                "groundedness": int(groundedness_val),
                "latency_ms": float(latency_ms)
            }
        })
        
    return details
