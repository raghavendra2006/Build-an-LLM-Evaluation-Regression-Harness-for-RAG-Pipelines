import json
import os
import sys
import pandas as pd
from src.dataset import load_eval_set, ensure_dataset_exists
from src.pipelines import PipelineA, PipelineB
from src.judge import LLMJudge
from src.metrics import evaluate_pipeline
from src.report import generate_regression_report

def main():
    print("==================================================")
    print(" Starting RAG Pipeline Regression Evaluation      ")
    print("==================================================")
    
    # Ensure dataset exists
    eval_set = load_eval_set()
    print(f"Loaded evaluation dataset with {len(eval_set)} questions.")
    
    # Initialize LLM Judge
    judge = LLMJudge()
    print(f"Initialized LLM Judge (Model: {judge.model}, Live API: {judge.use_live_api})")
    
    # Evaluate Pipeline A (Baseline)
    print("\n[1/3] Running Baseline Pipeline A...")
    pipeline_a = PipelineA()
    eval_a_details = evaluate_pipeline(pipeline_a, eval_set, judge)
    
    os.makedirs("results", exist_ok=True)
    eval_a_path = os.path.join("results", "eval_A_details.json")
    with open(eval_a_path, "w", encoding="utf-8") as f:
        json.dump(eval_a_details, f, indent=2)
    print(f"Saved Baseline Pipeline A details to {eval_a_path}")
    
    # Evaluate Pipeline B (Candidate)
    print("\n[2/3] Running Candidate Pipeline B...")
    pipeline_b = PipelineB()
    eval_b_details = evaluate_pipeline(pipeline_b, eval_set, judge)
    
    eval_b_path = os.path.join("results", "eval_B_details.json")
    with open(eval_b_path, "w", encoding="utf-8") as f:
        json.dump(eval_b_details, f, indent=2)
    print(f"Saved Candidate Pipeline B details to {eval_b_path}")
    
    # Generate Regression Report CSV
    print("\n[3/3] Generating Regression Report...")
    report_df = generate_regression_report(eval_a_details, eval_b_details)
    
    print("\n==================================================")
    print(" REGRESSION REPORT SUMMARY                       ")
    print("==================================================")
    print(report_df.to_string(index=False))
    print("==================================================")

if __name__ == "__main__":
    main()
