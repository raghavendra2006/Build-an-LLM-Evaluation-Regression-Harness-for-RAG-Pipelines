import os
import pandas as pd
import pytest
from src.report import determine_flag

def test_regression_report_schema_and_math():
    report_path = os.path.join("results", "regression_report.csv")
    assert os.path.exists(report_path), f"File missing: {report_path}"
    
    df = pd.read_csv(report_path)
    
    # Verify exact column headers
    expected_cols = ["metric", "baseline", "candidate", "delta", "flag"]
    assert list(df.columns) == expected_cols, f"Columns mismatch: {list(df.columns)} vs {expected_cols}"
    
    # Verify exactly 4 rows
    assert len(df) == 4, f"Expected 4 rows, found {len(df)}"
    
    metrics = list(df["metric"])
    expected_metrics = ["Recall", "Correctness", "Groundedness", "P95_Latency"]
    assert metrics == expected_metrics, f"Metrics mismatch: {metrics} vs {expected_metrics}"
    
    # Verify mathematical flag logic (Requirement 7)
    for idx, row in df.iterrows():
        metric = row["metric"]
        baseline = float(row["baseline"])
        candidate = float(row["candidate"])
        delta = float(row["delta"])
        flag = str(row["flag"])
        
        # Check delta math
        assert abs(delta - (candidate - baseline)) < 0.01, f"Row {metric} delta calculation error: {delta} != {candidate} - {baseline}"
        
        # Check flag math logic
        expected_flag = determine_flag(metric, baseline, candidate)
        assert flag == expected_flag, f"Row {metric} flag mismatch: got '{flag}', expected '{expected_flag}'"

def test_demonstrated_regression_criteria():
    report_path = os.path.join("results", "regression_report.csv")
    df = pd.read_csv(report_path).set_index("metric")
    
    cor_row = df.loc["Correctness"]
    gro_row = df.loc["Groundedness"]
    lat_row = df.loc["P95_Latency"]
    
    # Correctness Candidate >= Correctness Baseline
    assert float(cor_row["candidate"]) >= float(cor_row["baseline"]), \
        f"Correctness candidate ({cor_row['candidate']}) must be >= baseline ({cor_row['baseline']})"
        
    # Groundedness Candidate < Groundedness Baseline OR P95_Latency Candidate > P95_Latency Baseline
    gro_regress = float(gro_row["candidate"]) < float(gro_row["baseline"])
    lat_regress = float(lat_row["candidate"]) > float(lat_row["baseline"])
    assert gro_regress or lat_regress, "Either Groundedness must regress or P95_Latency must regress"
    
    # Assert at least one REGRESS flag is present
    flags = list(df["flag"])
    assert "REGRESS" in flags, f"At least one REGRESS flag must be present in flags: {flags}"
