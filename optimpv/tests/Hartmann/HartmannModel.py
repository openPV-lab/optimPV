"""Thin wrapper around the BoTorch Hartmann benchmark."""

from __future__ import annotations

import numpy as np
import torch
from botorch.test_functions.synthetic import Hartmann


class HartmannModel:
    """Instantiate and evaluate the Hartmann synthetic benchmark."""

    def __init__(
        self,
        *,
        dim: int = 6,
        negate: bool = False,
    ) -> None:
        if dim not in (3, 4, 6):
            raise ValueError("Hartmann benchmark supports only dimensions 3, 4, or 6.")

        self.dim = int(dim)
        self.negate = bool(negate)
        self.problem = Hartmann(dim=self.dim, negate=self.negate)
        self.problem_name = f"Hartmann{self.dim}D"
        self.nx = self.dim # number of free parameters
        self.nf = 1 # number of objectives
        self.ng = 0 # constraints
        self.bounds = np.column_stack(
            [
                self.problem.bounds[0].detach().cpu().numpy(),
                self.problem.bounds[1].detach().cpu().numpy(),
            ]
        )

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate one Hartmann design vector and return objective and optional constraints."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.nx:
            raise ValueError(
                f"{self.problem_name} expects {self.nx} variables, got {x.shape[0]}."
            )

        x_tensor = torch.tensor(x, dtype=torch.double).unsqueeze(0)
        value = self.problem(x_tensor).detach().cpu().numpy().reshape(-1)
        return value.astype(float), None
