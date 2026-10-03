import json
import os
from typing import Dict, Any

EVAL_PINS_PATH = os.path.join("config", "eval_pins.json")
JUDGE_PROMPTS_PATH = os.path.join("prompts", "judge_prompts.json")

def load_eval_pins() -> Dict[str, Any]:
    if not os.path.exists(EVAL_PINS_PATH):
        raise FileNotFoundError(f"Config file not found at {EVAL_PINS_PATH}")
    with open(EVAL_PINS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_judge_prompts() -> Dict[str, str]:
    if not os.path.exists(JUDGE_PROMPTS_PATH):
        raise FileNotFoundError(f"Prompts file not found at {JUDGE_PROMPTS_PATH}")
    with open(JUDGE_PROMPTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
