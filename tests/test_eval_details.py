import json
import os
import pytest

@pytest.mark.parametrize("file_name", ["eval_A_details.json", "eval_B_details.json"])
def test_eval_details_schema(file_name):
    eval_set_path = os.path.join("dataset", "eval_set.json")
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set_data = json.load(f)
        
    details_path = os.path.join("results", file_name)
    assert os.path.exists(details_path), f"File missing: {details_path}"
    
    with open(details_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert isinstance(data, list), f"{file_name} must be a JSON array"
    assert len(data) == len(eval_set_data), f"{file_name} length ({len(data)}) must match eval_set length ({len(eval_set_data)})"
    
    for idx, item in enumerate(data):
        assert "id" in item and isinstance(item["id"], str)
        assert "generated_answer" in item and isinstance(item["generated_answer"], str)
        assert "retrieved_context_titles" in item and isinstance(item["retrieved_context_titles"], list)
        assert "metrics" in item and isinstance(item["metrics"], dict)
        
        m = item["metrics"]
        assert "recall" in m and isinstance(m["recall"], (int, float)) and 0.0 <= m["recall"] <= 1.0
        assert "correctness" in m and isinstance(m["correctness"], int) and m["correctness"] in (0, 1)
        assert "groundedness" in m and isinstance(m["groundedness"], int) and m["groundedness"] in (0, 1)
        assert "latency_ms" in m and isinstance(m["latency_ms"], (int, float)) and m["latency_ms"] >= 0.0
