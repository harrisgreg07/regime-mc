"""regime-mc — Markov regime-switching Monte Carlo for correlated assets.

A small, generic engine for simulating multi-asset return paths that switch
between market regimes (e.g. calm vs. crisis) via a Markov transition matrix.
Each regime has its own mean vector and covariance, draws are correlated
(Cholesky), and the step distribution can be normal or Student-t (fat tails).

This is the *technique* — textbook Markov regime-switching simulation — with
parameters you supply. It ships no calibrated scenarios and no market data.

References: Hamilton (1989) on regime-switching; Markov-modulated return models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Regime",
    "RegimeSwitchingModel",
    "to_equity_curve",
    "portfolio_returns",
    "path_stats",
]

__version__ = "0.1.0"


@dataclass
class Regime:
    """A market regime: a mean vector and covariance over the same assets."""

    name: str
    mean: np.ndarray  # shape (n_assets,)
    cov: np.ndarray   # shape (n_assets, n_assets), symmetric positive-definite

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float).ravel()
        self.cov = np.asarray(self.cov, dtype=float)
        n = self.mean.size
        if self.cov.shape != (n, n):
            raise ValueError(f"cov must be ({n}, {n}) to match mean of length {n}")

    @classmethod
    def from_vol_corr(cls, name: str, mean, vols, corr) -> "Regime":
        """Build a regime from per-asset volatilities and a correlation matrix."""
        vols = np.asarray(vols, dtype=float).ravel()
        corr = np.asarray(corr, dtype=float)
        cov = np.outer(vols, vols) * corr
        return cls(name=name, mean=np.asarray(mean, dtype=float), cov=cov)

    @property
    def n_assets(self) -> int:
        return self.mean.size


class RegimeSwitchingModel:
    """Simulate correlated multi-asset returns under Markov regime switching.

    Args:
        regimes: list of :class:`Regime`, all over the same number of assets.
        transition: ``(K, K)`` row-stochastic matrix; ``transition[i, j]`` is the
            probability of moving from regime ``i`` to regime ``j`` next step.
        dist: step distribution — ``"normal"`` or ``"t"`` (Student-t, fat tails).
        df: degrees of freedom for the t distribution (> 2). Draws are rescaled
            so the realized covariance still equals each regime's ``cov``.
        seed: optional seed for reproducibility.
    """

    def __init__(self, regimes, transition, dist="normal", df=5.0, seed=None):
        if len(regimes) == 0:
            raise ValueError("need at least one regime")
        n_assets = regimes[0].n_assets
        if any(r.n_assets != n_assets for r in regimes):
            raise ValueError("all regimes must cover the same number of assets")

        transition = np.asarray(transition, dtype=float)
        k = len(regimes)
        if transition.shape != (k, k):
            raise ValueError(f"transition must be ({k}, {k})")
        if np.any(transition < 0):
            raise ValueError("transition probabilities must be non-negative")
        if not np.allclose(transition.sum(axis=1), 1.0):
            raise ValueError("each row of transition must sum to 1")
        if dist not in ("normal", "t"):
            raise ValueError("dist must be 'normal' or 't'")
        if dist == "t" and df <= 2:
            raise ValueError("df must be > 2 for the t distribution")

        self.regimes = list(regimes)
        self.transition = transition
        self.dist = dist
        self.df = float(df)
        self.n_assets = n_assets
        self.n_regimes = k
        self.rng = np.random.default_rng(seed)

        self._means = np.vstack([r.mean for r in regimes])            # (K, n_assets)
        self._chol = [np.linalg.cholesky(r.cov) for r in regimes]     # list of (n, n)
        self._cdf = np.cumsum(transition, axis=1)                     # (K, K)

    def stationary_distribution(self) -> np.ndarray:
        """Long-run regime probabilities implied by the transition matrix."""
        p = self.transition
        k = self.n_regimes
        a = np.vstack([p.T - np.eye(k), np.ones(k)])
        b = np.append(np.zeros(k), 1.0)
        pi, *_ = np.linalg.lstsq(a, b, rcond=None)
        pi = np.clip(pi, 0.0, None)
        return pi / pi.sum()

    def simulate_regime_paths(self, steps: int, n_paths: int, start=None) -> np.ndarray:
        """Sample regime index paths, shape ``(n_paths, steps)``.

        Initial regime is drawn from the stationary distribution unless ``start``
        (a regime index) is given.
        """
        reg = np.empty((n_paths, steps), dtype=int)
        if start is None:
            cur = self.rng.choice(self.n_regimes, size=n_paths, p=self.stationary_distribution())
        else:
            cur = np.full(n_paths, int(start))
        for t in range(steps):
            reg[:, t] = cur
            u = self.rng.random(n_paths)
            cur = (u[:, None] < self._cdf[cur]).argmax(axis=1)
        return reg

    def simulate(self, steps: int, n_paths: int = 1, start=None, return_regimes=False):
        """Simulate return paths, shape ``(n_paths, steps, n_assets)``.

        Set ``return_regimes=True`` to also get the ``(n_paths, steps)`` regime
        indices.
        """
        reg = self.simulate_regime_paths(steps, n_paths, start)
        z = self.rng.standard_normal((n_paths, steps, self.n_assets))
        out = np.empty_like(z)
        for k in range(self.n_regimes):
            mask = reg == k
            m = int(mask.sum())
            if m == 0:
                continue
            draw = z[mask] @ self._chol[k].T  # covariance == regime cov
            if self.dist == "t":
                g = self.rng.chisquare(self.df, size=m) / self.df
                draw = draw / np.sqrt(g)[:, None] * np.sqrt((self.df - 2.0) / self.df)
            out[mask] = self._means[k] + draw
        if return_regimes:
            return out, reg
        return out


def to_equity_curve(returns, initial: float = 1.0) -> np.ndarray:
    """Compound simple returns along the last axis into a wealth curve."""
    r = np.asarray(returns, dtype=float)
    return initial * np.cumprod(1.0 + r, axis=-1)


def portfolio_returns(returns, weights) -> np.ndarray:
    """Collapse multi-asset returns ``(..., steps, n_assets)`` to a weighted
    portfolio return series ``(..., steps)``."""
    return np.asarray(returns, dtype=float) @ np.asarray(weights, dtype=float)


def path_stats(returns, levels=(0.05, 0.01)) -> dict:
    """Summary statistics over simulated single-stream returns ``(n_paths, steps)``.

    Returns per-step mean/vol, terminal-return mean/median, average and worst
    max-drawdown, and left-tail VaR/CVaR of the terminal return at each level
    (e.g. 0.05 -> the worst 5%).
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r[None, :]
    eq = to_equity_curve(r)
    terminal = eq[:, -1] - 1.0
    running_max = np.maximum.accumulate(eq, axis=1)
    max_dd = (eq / running_max - 1.0).min(axis=1)

    stats = {
        "mean_step": float(r.mean()),
        "vol_step": float(r.std()),
        "terminal_mean": float(terminal.mean()),
        "terminal_median": float(np.median(terminal)),
        "max_drawdown_mean": float(max_dd.mean()),
        "max_drawdown_worst": float(max_dd.min()),
    }
    for lvl in levels:
        var = float(np.quantile(terminal, lvl))
        tail = terminal[terminal <= var]
        cvar = float(tail.mean()) if tail.size else var
        pct = int(round(lvl * 100))
        stats[f"var_{pct}"] = var
        stats[f"cvar_{pct}"] = cvar
    return stats
