import csv
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any

OUTPUT_REPORT_PATH = os.path.join("results", "regression_report.csv")

def determine_flag(metric: str, baseline: float, candidate: float) -> str:
    """Calculates flag strictly based on metric direction and rules.
    - For Recall, Correctness, Groundedness: If candidate < baseline -> REGRESS, candidate > baseline -> IMPROVE, else NEUTRAL.
    - For P95_Latency: If candidate > baseline * 1.10 -> REGRESS, candidate < baseline -> IMPROVE, else NEUTRAL.
    """
    if metric in ["Recall", "Correctness", "Groundedness"]:
        if candidate < baseline:
            return "REGRESS"
        elif candidate > baseline:
            return "IMPROVE"
        else:
            return "NEUTRAL"
    elif metric == "P95_Latency":
        if candidate > (baseline * 1.10):
            return "REGRESS"
        elif candidate < baseline:
            return "IMPROVE"
        else:
            return "NEUTRAL"
    else:
        return "NEUTRAL"

def generate_regression_report(results_a: List[Dict[str, Any]], results_b: List[Dict[str, Any]], output_path: str = OUTPUT_REPORT_PATH) -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Aggregation for Baseline A
    rec_a = np.mean([item["metrics"]["recall"] for item in results_a])
    cor_a = np.mean([item["metrics"]["correctness"] for item in results_a])
    gro_a = np.mean([item["metrics"]["groundedness"] for item in results_a])
    lat_a = np.percentile([item["metrics"]["latency_ms"] for item in results_a], 95)
    
    # Aggregation for Candidate B
    rec_b = np.mean([item["metrics"]["recall"] for item in results_b])
    cor_b = np.mean([item["metrics"]["correctness"] for item in results_b])
    gro_b = np.mean([item["metrics"]["groundedness"] for item in results_b])
    lat_b = np.percentile([item["metrics"]["latency_ms"] for item in results_b], 95)
    
    metrics_summary = [
        ("Recall", float(round(rec_a, 4)), float(round(rec_b, 4))),
        ("Correctness", float(round(cor_a, 4)), float(round(cor_b, 4))),
        ("Groundedness", float(round(gro_a, 4)), float(round(gro_b, 4))),
        ("P95_Latency", float(round(lat_a, 2)), float(round(lat_b, 2)))
    ]
    
    rows = []
    for metric_name, base_val, cand_val in metrics_summary:
        delta = float(round(cand_val - base_val, 4))
        flag = determine_flag(metric_name, base_val, cand_val)
        rows.append({
            "metric": metric_name,
            "baseline": base_val,
            "candidate": cand_val,
            "delta": delta,
            "flag": flag
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Regression report written to {output_path}")
    return df
