from __future__ import annotations
import json
import logging
import random
from pathlib import Path
from .schemas import PreferenceExample

logger = logging.getLogger(__name__)

def load_jsonl(path: str | Path) -> list[PreferenceExample]:
    """Load preference examples from JSONL.

    Implemented: line-numbered errors, duplicate prompt checks.
    """
    examples: list[PreferenceExample] = []
    seen_prompts = set()
    
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                example = PreferenceExample.model_validate(data)
                
                # Check for duplicate prompts
                prompt_lower = example.prompt.strip().lower()
                if prompt_lower in seen_prompts:
                    logger.warning(f"Line {i}: Duplicate prompt found: {example.prompt[:30]}...")
                    continue
                    
                seen_prompts.add(prompt_lower)
                examples.append(example)
                
            except json.JSONDecodeError as e:
                logger.error(f"Line {i}: JSON decode error - {e}")
            except Exception as e:
                logger.error(f"Line {i}: Validation error - {e}")
                
    return examples

def split_by_prompt(examples: list[PreferenceExample], validation_ratio: float = 0.2, seed: int = 42) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid leakage.

    Implemented: deterministic shuffling by seed and grouping by prompt.
    Current skeleton returns a simple split for demonstration only.
    """
    # Group by prompt (already unique if loaded via load_jsonl, but good practice)
    grouped = {}
    for ex in examples:
        grouped.setdefault(ex.prompt, []).append(ex)
        
    prompts = list(grouped.keys())
    
    # Deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(prompts)
    
    cut = max(1, int(len(prompts) * (1 - validation_ratio)))
    train_prompts = set(prompts[:cut])
    
    train_examples = []
    eval_examples = []
    
    for ex in examples:
        if ex.prompt in train_prompts:
            train_examples.append(ex)
        else:
            eval_examples.append(ex)
            
    return train_examples, eval_examples
