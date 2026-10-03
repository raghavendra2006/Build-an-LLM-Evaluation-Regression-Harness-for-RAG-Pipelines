import json
import os
import pytest

def test_eval_set_schema_and_size():
    eval_set_path = os.path.join("dataset", "eval_set.json")
    assert os.path.exists(eval_set_path), f"File missing: {eval_set_path}"
    
    with open(eval_set_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert isinstance(data, list), "eval_set.json must be a JSON array"
    assert 30 <= len(data) <= 50, f"eval_set.json length ({len(data)}) must be between 30 and 50"
    
    required_keys = {"id", "question", "ground_truth_answer", "ground_truth_context_titles"}
    for idx, item in enumerate(data):
        assert isinstance(item, dict), f"Item {idx} is not an object"
        missing = required_keys - set(item.keys())
        assert not missing, f"Item {idx} missing required keys: {missing}"
        assert isinstance(item["id"], str) and len(item["id"]) > 0
        assert isinstance(item["question"], str) and len(item["question"]) > 0
        assert isinstance(item["ground_truth_answer"], str)
        assert isinstance(item["ground_truth_context_titles"], list)
