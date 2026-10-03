import json
import os
import pytest

def test_eval_pins_schema():
    pins_path = os.path.join("config", "eval_pins.json")
    assert os.path.exists(pins_path), f"File missing: {pins_path}"
    
    with open(pins_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert isinstance(data, dict), "eval_pins.json must be a JSON object"
    
    assert "dataset_size" in data and isinstance(data["dataset_size"], (int, float))
    assert "llm_judge_model" in data and isinstance(data["llm_judge_model"], str)
    assert "pipeline_a_embedder" in data and isinstance(data["pipeline_a_embedder"], str)
    assert "pipeline_b_embedder" in data and isinstance(data["pipeline_b_embedder"], str)
