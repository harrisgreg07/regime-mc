# regime-mc

Markov **regime-switching Monte Carlo** for correlated multi-asset returns —
simulate paths that switch between regimes (e.g. calm vs. crisis) via a
transition matrix, with correlated draws and optional fat tails.

This is the generic *technique* with parameters you supply: no calibrated
scenarios, no market data.

```python
import numpy as np
from regime_mc import Regime, RegimeSwitchingModel, portfolio_returns, path_stats

calm   = Regime.from_vol_corr("calm",   [0.0012, 0.0002], [0.02, 0.008], [[1, -0.2], [-0.2, 1]])
crisis = Regime.from_vol_corr("crisis", [-0.006, 0.0005], [0.05, 0.015], [[1,  0.5], [ 0.5, 1]])

model = RegimeSwitchingModel([calm, crisis], [[0.97, 0.03], [0.20, 0.80]], dist="t", df=5, seed=7)

returns = model.simulate(steps=52, n_paths=10_000)     # (10000, 52, 2)
book    = portfolio_returns(returns, weights=[0.6, 0.4])
print(path_stats(book))                                # VaR/CVaR at 5% & 1%, drawdowns, ...
```

## What's in it

- **`Regime`** — mean vector + covariance over a set of assets (`from_vol_corr`
  builds it from vols + a correlation matrix).
- **`RegimeSwitchingModel`** — regimes + a Markov `transition` matrix:
  - `simulate(steps, n_paths)` → correlated return paths `(n_paths, steps, n_assets)`,
  - `simulate_regime_paths(...)` → the regime-index paths,
  - `stationary_distribution()` → long-run regime mix.
  - Step distribution is **normal** or **Student-t** (fat tails), with t rescaled
    so the realized covariance still equals each regime's `cov`.
- **Helpers** — `to_equity_curve`, `portfolio_returns`, and `path_stats`
  (per-step mean/vol, terminal mean/median, average & worst max-drawdown, and
  left-tail **VaR/CVaR** at any levels).

## Why it's useful

A single covariance matrix can't express "correlations converge and hedges fail
in a crisis." Regime switching can: give the crisis regime higher vols and a
higher cross-correlation, make it sticky, and the simulated tails get fatter and
more clustered — much closer to how markets actually behave. See
[`examples/calm_vs_crisis.py`](examples/calm_vs_crisis.py).

Pairs naturally with [kelly-sizing](https://github.com/harrisgreg07/kelly-sizing):
`regime-mc` generates a return sample, `kelly-sizing.growth_optimal_fraction`
sizes it from the full (fat-tailed) distribution.

## Install & test

```bash
pip install git+https://github.com/harrisgreg07/regime-mc.git
# dev:
pip install -e ".[test]" && pytest
python examples/calm_vs_crisis.py
```

## License

MIT © Greg Harris
