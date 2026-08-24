from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2


class PreferenceTrainer:
    """Interface for DPO/ORPO training implementations."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self) -> None:
        """Train the policy."""
        import logging

        logging.getLogger(__name__).info(
            f"Mock training with {self.config.method} (beta={self.config.beta}, lambda={self.config.lambda_orpo})"
        )
