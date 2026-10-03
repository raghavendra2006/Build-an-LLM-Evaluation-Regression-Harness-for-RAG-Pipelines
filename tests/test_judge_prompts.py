import json
import os
import pytest

def test_judge_prompts_schema():
    prompts_path = os.path.join("prompts", "judge_prompts.json")
    assert os.path.exists(prompts_path), f"File missing: {prompts_path}"
    
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert isinstance(data, dict), "judge_prompts.json must be a JSON object"
    assert "correctness_prompt" in data and isinstance(data["correctness_prompt"], str)
    assert "groundedness_prompt" in data and isinstance(data["groundedness_prompt"], str)
    
    groundedness_str = data["groundedness_prompt"].lower()
    assert "score 0" in groundedness_str
    assert "context" in groundedness_str
