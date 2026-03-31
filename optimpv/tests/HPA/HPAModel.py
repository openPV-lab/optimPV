"""Thin wrapper around the vendored HPA benchmark problems."""

from __future__ import annotations

import numpy as np

from optimpv.tests.HPA.hpa import problem as hpa_problem


class HPAModel:
    """Instantiate and evaluate a selected HPA benchmark problem."""

    def __init__(
        self,
        problem_name: str = "HPA101",
        *,
        n_div: int = 4,
        level: int = 0,
        normalized: bool = True,
    ) -> None:
        if not hasattr(hpa_problem, problem_name):
            raise ValueError(f"Unknown HPA problem '{problem_name}'.")

        problem_cls = getattr(hpa_problem, problem_name)
        self.problem_name = problem_name
        self.problem = problem_cls(n_div=n_div, level=level, NORMALIZED=normalized)
        self.nx = int(self.problem.nx)
        self.nf = int(self.problem.nf)
        self.ng = int(self.problem.ng)
        self.normalized = bool(normalized)
        self.bounds = np.column_stack([self.problem.lbound, self.problem.ubound])

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate one design vector and return objective and optional constraints."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.nx:
            raise ValueError(
                f"{self.problem_name} expects {self.nx} variables, got {x.shape[0]}."
            )

        result = self.problem(x)
        if self.ng > 0:
            f, g = result
            return np.asarray(f, dtype=float).reshape(-1), np.asarray(g, dtype=float).reshape(-1)
        return np.asarray(result, dtype=float).reshape(-1), None
