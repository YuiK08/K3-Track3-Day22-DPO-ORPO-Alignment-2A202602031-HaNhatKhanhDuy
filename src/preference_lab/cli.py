from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from .config import load_config
from .data import load_jsonl
from .evaluate import pairwise_accuracy, write_metrics

app = typer.Typer(help="Preference alignment lab CLI")


@app.command()
def validate(data: Path) -> None:
    examples = load_jsonl(data)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")


@app.command()
def evaluate(config: Path = typer.Option(..., "--config", help="Path to config file")) -> None:  # noqa: B008
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"]["train_data"])

    import numpy as np

    from .losses import dpo_loss, orpo_loss

    # Dummy deterministic scorer for CPU mode
    chosen_scores = [float(len(ex.chosen) % 10) * 0.1 for ex in examples]
    rejected_scores = [float(len(ex.rejected) % 10) * 0.1 for ex in examples]

    # Numpy Array cho hàm Loss
    policy_chosen_logps = np.array(chosen_scores)
    policy_rejected_logps = np.array(rejected_scores)
    ref_chosen_logps = policy_chosen_logps - 0.5
    ref_rejected_logps = policy_rejected_logps - 0.5
    sft_nll = -policy_chosen_logps

    # Tính toán kết quả DPO & ORPO Loss qua hàm đã code
    dpo_val = dpo_loss(
        policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1
    )
    orpo_val = orpo_loss(sft_nll, policy_chosen_logps, policy_rejected_logps, lambda_orpo=0.1)

    metrics = {
        "pairwise_accuracy": pairwise_accuracy(examples, chosen_scores, rejected_scores),
        "dpo_loss_mock": dpo_val,
        "orpo_loss_mock": orpo_val,
    }
    out = write_metrics(metrics, cfg["paths"]["output_dir"])
    print(f"[green]Wrote metrics to {out}[/green]")


if __name__ == "__main__":
    app()
