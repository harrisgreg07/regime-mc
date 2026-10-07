import numpy as np
import pytest

from regime_mc import (
    Regime,
    RegimeSwitchingModel,
    to_equity_curve,
    portfolio_returns,
    path_stats,
)


def _calm_crisis():
    calm = Regime.from_vol_corr("calm", [0.001, 0.0002], [0.02, 0.008], [[1, -0.2], [-0.2, 1]])
    crisis = Regime.from_vol_corr("crisis", [-0.005, 0.0005], [0.05, 0.015], [[1, 0.5], [0.5, 1]])
    transition = np.array([[0.97, 0.03], [0.20, 0.80]])
    return [calm, crisis], transition


def test_shapes():
    regimes, trans = _calm_crisis()
    m = RegimeSwitchingModel(regimes, trans, seed=0)
    out, reg = m.simulate(steps=50, n_paths=10, return_regimes=True)
    assert out.shape == (10, 50, 2)
    assert reg.shape == (10, 50)
    assert set(np.unique(reg)).issubset({0, 1})


def test_stationary_matches_empirical_frequency():
    regimes, trans = _calm_crisis()
    m = RegimeSwitchingModel(regimes, trans, seed=1)
    _, reg = m.simulate(steps=4000, n_paths=200, return_regimes=True)
    emp = np.bincount(reg.ravel(), minlength=2) / reg.size
    pi = m.stationary_distribution()
    assert np.allclose(emp, pi, atol=0.02)


def test_within_regime_covariance_recovered():
    # Single persistent regime -> realized correlation matches the target.
    target_corr = 0.5
    r = Regime.from_vol_corr("only", [0.0, 0.0], [0.02, 0.03], [[1, target_corr], [target_corr, 1]])
    m = RegimeSwitchingModel([r], [[1.0]], seed=2)
    out = m.simulate(steps=1, n_paths=200_000)[:, 0, :]
    corr = np.corrcoef(out.T)[0, 1]
    assert corr == pytest.approx(target_corr, abs=0.02)


def test_t_has_fatter_tails_than_normal_same_cov():
    cov = [[0.0004, 0.0], [0.0, 0.0004]]
    r = Regime("r", [0.0, 0.0], cov)
    normal = RegimeSwitchingModel([r], [[1.0]], dist="normal", seed=3)
    heavy = RegimeSwitchingModel([r], [[1.0]], dist="t", df=4, seed=3)
    xn = normal.simulate(1, 100_000)[:, 0, 0]
    xt = heavy.simulate(1, 100_000)[:, 0, 0]
    # similar variance...
    assert xt.std() == pytest.approx(xn.std(), rel=0.1)
    # ...but heavier tails (excess kurtosis).
    def kurt(x):
        z = (x - x.mean()) / x.std()
        return (z ** 4).mean()
    assert kurt(xt) > kurt(xn) + 1.0


def test_portfolio_and_stats():
    regimes, trans = _calm_crisis()
    m = RegimeSwitchingModel(regimes, trans, seed=4)
    out = m.simulate(steps=52, n_paths=5000)
    port = portfolio_returns(out, [0.6, 0.4])
    assert port.shape == (5000, 52)
    stats = path_stats(port, levels=(0.05, 0.01))
    # CVaR is at least as extreme as VaR, and both are losses here.
    assert stats["cvar_5"] <= stats["var_5"]
    assert stats["cvar_1"] <= stats["var_1"]
    assert stats["max_drawdown_worst"] <= stats["max_drawdown_mean"] <= 0


def test_equity_curve():
    eq = to_equity_curve(np.array([0.1, -0.5, 2.0]))
    assert eq[-1] == pytest.approx(1.1 * 0.5 * 3.0)


def test_transition_validation():
    r = Regime("r", [0.0], [[0.01]])
    with pytest.raises(ValueError):
        RegimeSwitchingModel([r, r], [[0.5, 0.5]], seed=0)  # wrong shape
    with pytest.raises(ValueError):
        RegimeSwitchingModel([r], [[0.9]], seed=0)          # row doesn't sum to 1
