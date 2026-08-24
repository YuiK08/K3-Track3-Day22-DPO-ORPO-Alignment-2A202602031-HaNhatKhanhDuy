from __future__ import annotations
import json
from pathlib import Path
from .schemas import PreferenceExample

def pairwise_accuracy(examples: list[PreferenceExample], chosen_scores: list[float], rejected_scores: list[float]) -> float:
    """Return fraction where chosen score is greater than rejected score."""
    if not examples:
        return 0.0
    if len(examples) != len(chosen_scores) or len(examples) != len(rejected_scores):
        raise ValueError("Lengths of examples and scores must match")
    wins = sum(c > r for c, r in zip(chosen_scores, rejected_scores, strict=False))
    ties = sum(c == r for c, r in zip(chosen_scores, rejected_scores, strict=False))
    return (wins + ties * 0.5) / len(examples)

def write_metrics(metrics: dict[str, float], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
