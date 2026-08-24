import numpy as np

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_todo() -> None:
    loss = dpo_loss(
        np.array([-0.5]), np.array([-1.5]), np.array([-0.6]), np.array([-1.0]), beta=0.1
    )
    assert isinstance(loss, float)


def test_orpo_loss_todo() -> None:
    loss = orpo_loss(np.array([1.0]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=0.1)
    assert isinstance(loss, float)
