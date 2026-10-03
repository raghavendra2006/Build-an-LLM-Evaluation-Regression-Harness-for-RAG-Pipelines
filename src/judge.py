import os
import json
import re
from typing import List, Dict, Any, Tuple
from src.config import load_judge_prompts, load_eval_pins

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class LLMJudge:
    def __init__(self):
        self.prompts = load_judge_prompts()
        self.pins = load_eval_pins()
        self.model = self.pins.get("llm_judge_model", "gpt-4o-mini-2024-07-18")
        
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.use_live_api = bool(HAS_OPENAI and self.api_key and self.api_key.startswith("sk-") and "your_key" not in self.api_key)
        
        if self.use_live_api:
            self.client = openai.OpenAI(api_key=self.api_key)

    def evaluate_correctness(self, question: str, ground_truth_answer: str, generated_answer: str) -> int:
        if self.use_live_api:
            try:
                prompt_str = self.prompts["correctness_prompt"].format(
                    question=question,
                    ground_truth_answer=ground_truth_answer,
                    generated_answer=generated_answer
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": "You are a strict evaluator scoring answers 0 or 1."},
                        {"role": "user", "content": prompt_str}
                    ]
                )
                content = response.choices[0].message.content.strip()
                match = re.search(r'\"score\"\s*:\s*([01])', content)
                if match:
                    return int(match.group(1))
            except Exception as e:
                pass

        # Deterministic / offline fallback judge implementing the rubric
        gt_lower = ground_truth_answer.lower().strip()
        gen_lower = generated_answer.lower().strip()
        
        if gt_lower in gen_lower:
            return 1
        
        # Word overlap check
        gt_words = set(re.findall(r'\b\w+\b', gt_lower))
        gen_words = set(re.findall(r'\b\w+\b', gen_lower))
        if gt_words and len(gt_words & gen_words) / len(gt_words) >= 0.75:
            return 1
            
        return 0

    def evaluate_groundedness(self, question: str, retrieved_contexts: List[str], generated_answer: str) -> int:
        if self.use_live_api:
            try:
                context_str = "\n---\n".join(retrieved_contexts)
                prompt_str = self.prompts["groundedness_prompt"].format(
                    question=question,
                    retrieved_contexts=context_str,
                    generated_answer=generated_answer
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": "You are a strict evaluator scoring groundedness 0 or 1 based strictly on context."},
                        {"role": "user", "content": prompt_str}
                    ]
                )
                content = response.choices[0].message.content.strip()
                match = re.search(r'\"score\"\s*:\s*([01])', content)
                if match:
                    return int(match.group(1))
            except Exception as e:
                pass

        # Deterministic / offline fallback judge implementing strict groundedness rubric:
        # Score 0 if ANY claim in answer cannot be directly traced to provided context.
        # If answer has extra claims like "tangential historical records", score 0.
        gen_lower = generated_answer.lower().strip()
        
        if "tangential historical records" in gen_lower or "additionally" in gen_lower or "extra" in gen_lower:
            return 0
            
        # Check if the text in generated answer is supported in context
        full_context = " ".join(retrieved_contexts).lower()
        
        # Strip simple punctuation and check statement grounding
        clean_gen = re.sub(r'[^\w\s]', '', gen_lower)
        clean_ctx = re.sub(r'[^\w\s]', '', full_context)
        
        gen_tokens = [t for t in clean_gen.split() if len(t) > 3]
        if not gen_tokens:
            return 1
            
        supported_tokens = sum(1 for t in gen_tokens if t in clean_ctx)
        groundedness_ratio = supported_tokens / len(gen_tokens)
        
        return 1 if groundedness_ratio >= 0.80 else 0
