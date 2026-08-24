from __future__ import annotations
import numpy as np

def dpo_loss(policy_chosen_logps: np.ndarray, policy_rejected_logps: np.ndarray, ref_chosen_logps: np.ndarray, ref_rejected_logps: np.ndarray, beta: float) -> float:
    """Compute batch DPO loss from sequence log probabilities."""
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = policy_logratios - ref_logratios
    losses = np.logaddexp(0, -beta * logits)
    return float(np.mean(losses))

def orpo_loss(sft_nll: np.ndarray, chosen_logps: np.ndarray, rejected_logps: np.ndarray, lambda_orpo: float) -> float:
    """Compute a simplified ORPO-style objective."""
    log_odds_chosen = chosen_logps - np.log1p(-np.exp(np.clip(chosen_logps, -np.inf, -1e-7)))
    log_odds_rejected = rejected_logps - np.log1p(-np.exp(np.clip(rejected_logps, -np.inf, -1e-7)))
    log_odds_ratio = log_odds_chosen - log_odds_rejected
    odds_loss = np.mean(np.logaddexp(0, -log_odds_ratio))
    return float(np.mean(sft_nll) + lambda_orpo * odds_loss)
