"""Thin wrapper around the Casmopolitan PestControl benchmark."""

import numpy as np
import torch

from .pest import PestControl


class PestModel:
    """Instantiate and evaluate the PestControl synthetic benchmark."""

    def __init__(
        self,
        *,
        random_seed: int = 0,
        normalize: bool = False,
    ) -> None:
        self.random_seed = int(random_seed)
        self.normalize = bool(normalize)
        self.problem = PestControl(random_seed=self.random_seed, normalize=self.normalize)
        self.problem_name = "PestControl"
        self.nx = int(self.problem.dim)
        self.nf = 1
        self.ng = 0
        self.bounds = None

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate one PestControl design vector and return objective and optional constraints."""
        x = np.asarray(x).reshape(-1)
        if x.shape[0] != self.nx:
            raise ValueError(
                f"{self.problem_name} expects {self.nx} variables, got {x.shape[0]}."
            )

        x_tensor = torch.tensor(x, dtype=torch.long)
        value = self.problem.compute(x_tensor, normalize=False).detach().cpu().numpy().reshape(-1)
        return value.astype(float), None
