"""Two-regime (calm vs. crisis) simulation of an equity/bond book.

Shows regime clustering, the effect of a crisis correlation break, and how
regime switching produces fatter tails than a single-regime model — all with
synthetic parameters.
"""

import numpy as np

from regime_mc import Regime, RegimeSwitchingModel, portfolio_returns, path_stats

# Weekly parameters for two assets: [equity, bond].
calm = Regime.from_vol_corr(
    "calm",
    mean=[0.0012, 0.0002],
    vols=[0.020, 0.008],
    corr=[[1.0, -0.2], [-0.2, 1.0]],   # bonds hedge equities in calm markets
)
crisis = Regime.from_vol_corr(
    "crisis",
    mean=[-0.006, 0.0005],
    vols=[0.050, 0.015],
    corr=[[1.0, 0.5], [0.5, 1.0]],     # correlation break: the hedge weakens
)

# Calm is sticky (97%); crises are shorter but cluster (80% persistence).
transition = np.array([[0.97, 0.03], [0.20, 0.80]])

model = RegimeSwitchingModel([calm, crisis], transition, dist="t", df=5, seed=7)
print("Stationary regime mix [calm, crisis]:", np.round(model.stationary_distribution(), 3))

# One year of weekly returns, 10k paths, 60/40 book.
returns = model.simulate(steps=52, n_paths=10_000)
book = portfolio_returns(returns, weights=[0.6, 0.4])

stats = path_stats(book, levels=(0.05, 0.01))
print("\n60/40 book, 1 year (weekly), 10k paths:")
for k, v in stats.items():
    print(f"  {k:20s} {v:+.4f}")

# Compare tail risk to a single calm-only model (no regime switching).
calm_only = RegimeSwitchingModel([calm], [[1.0]], dist="t", df=5, seed=7)
book_calm = portfolio_returns(calm_only.simulate(52, 10_000), [0.6, 0.4])
print("\nWorst-1% terminal return (CVaR 1%):")
print(f"  regime-switching: {path_stats(book)['cvar_1']:+.4f}")
print(f"  calm-only:        {path_stats(book_calm)['cvar_1']:+.4f}")
