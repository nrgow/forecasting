from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class PortfolioOptimizationConfig:
    """Configuration for prediction market portfolio optimization."""

    max_fraction_per_market: float
    max_total_fraction: float
    kelly_fraction: float = 1.0
    turnover_penalty: float = 0.0
    epsilon: float = 1e-9
    solver: str = "ECOS"


@dataclass(frozen=True)
class PortfolioOptimizationResult:
    """Result for prediction market portfolio optimization."""

    target_alloc: list[float]
    expected_log_growth: float


class PortfolioOptimizer:
    """CVXPY-based portfolio optimizer for binary prediction markets."""

    def __init__(self, config: PortfolioOptimizationConfig) -> None:
        """Initialize the optimizer with a configuration."""
        self.config = config

    def optimize(
        self,
        p_yes: list[float] | np.ndarray,
        q: list[float] | np.ndarray,
        current_alloc: list[float] | np.ndarray,
    ) -> PortfolioOptimizationResult:
        """Solve for optimal portfolio allocations."""
        start_time = time.time()
        p_yes = np.asarray(p_yes, dtype=float)
        q = np.asarray(q, dtype=float)
        current_alloc = np.asarray(current_alloc, dtype=float)

        if not (len(p_yes) == len(q) == len(current_alloc)):
            raise ValueError("Input arrays must have the same length.")

        n_markets = len(p_yes)
        logging.info("Portfolio optimization starting for %d markets", n_markets)

        p_no = 1 - p_yes
        f_yes = cp.Variable(n_markets, nonneg=True)
        f_no = cp.Variable(n_markets, nonneg=True)
        f_signed = f_yes - f_no

        yes_multiplier = 1 + cp.multiply(f_yes, (1 - p_yes) / p_yes) - f_no
        no_multiplier = 1 + cp.multiply(f_no, (1 - p_no) / p_no) - f_yes

        expected_log = cp.sum(
            cp.multiply(q, cp.log(yes_multiplier))
            + cp.multiply(1 - q, cp.log(no_multiplier))
        )

        turnover = cp.sum(cp.abs(f_signed - current_alloc))

        objective = cp.Maximize(
            self.config.kelly_fraction * expected_log
            - self.config.turnover_penalty * turnover
        )

        constraints = [
            f_yes <= self.config.max_fraction_per_market,
            f_no <= self.config.max_fraction_per_market,
            f_yes + f_no <= self.config.max_fraction_per_market,
            cp.sum(f_yes + f_no) <= self.config.max_total_fraction,
            yes_multiplier >= self.config.epsilon,
            no_multiplier >= self.config.epsilon,
        ]

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=self.config.solver)
        except cp.SolverError:
            problem.solve(solver="SCS")

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise ValueError(f"Optimization failed with status={problem.status}")

        elapsed = time.time() - start_time
        logging.info(
            "Portfolio optimization finished in %.2fs for %d markets",
            elapsed,
            n_markets,
        )

        return PortfolioOptimizationResult(
            target_alloc=f_signed.value.tolist(),
            expected_log_growth=float(expected_log.value),
        )
