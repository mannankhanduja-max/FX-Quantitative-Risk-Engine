"""
Tests for the no-information path generator.

The generator's whole value is that it contains NOTHING. If it drifts,
or if the control regime quietly clusters like the treatment, then any
conclusion drawn from it is worthless - so most of these tests are
about the absence of things rather than the presence of them.
"""
from __future__ import annotations

import numpy as np
import pytest

from fxrisk.research import null_paths as npaths


def logret(b):
    return np.log(b["Close"]).diff().dropna().to_numpy()


# ------------------------------------------------------------ no drift

@pytest.mark.parametrize("regime", ["constant", "garch"])
def test_log_drift_is_indistinguishable_from_zero(regime):
    b = npaths.garch_path(40_000, 1, regime=regime)
    r = logret(b)
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    assert abs(t) < 3.0, f"{regime} path drifts, t={t:+.2f}"


@pytest.mark.parametrize("regime", ["constant", "garch"])
def test_both_regimes_share_the_same_unconditional_scale(regime):
    # If the treatment were also more volatile, resolution and vol
    # would be confounded and the comparison would prove nothing.
    b = npaths.garch_path(40_000, 2, regime=regime)
    assert logret(b).std() == pytest.approx(npaths.SIGMA_5M, rel=0.15)


# ------------------------------------------- the control is a control

def test_constant_regime_does_not_cluster():
    b = npaths.garch_path(40_000, 3, regime="constant")
    assert abs(npaths.realised_clustering(b)) < 0.03


def test_garch_regime_does_cluster():
    b = npaths.garch_path(40_000, 3, regime="garch")
    assert npaths.realised_clustering(b) > 0.08


def test_the_two_regimes_are_actually_different():
    a = npaths.realised_clustering(npaths.garch_path(40_000, 4, regime="constant"))
    g = npaths.realised_clustering(npaths.garch_path(40_000, 4, regime="garch"))
    assert g > a + 0.05


# ------------------------------------------------------------ the bars

@pytest.mark.parametrize("regime", ["constant", "garch"])
def test_ohlc_is_internally_consistent(regime):
    b = npaths.garch_path(5_000, 5, regime=regime)
    assert (b["High"] >= b[["Open", "Close"]].max(axis=1) - 1e-12).all()
    assert (b["Low"] <= b[["Open", "Close"]].min(axis=1) + 1e-12).all()
    assert (b["High"] >= b["Low"]).all()
    assert (b[["Open", "High", "Low", "Close"]] > 0).all().all()


def test_bars_do_not_gap_against_the_previous_close():
    # A synthetic gap would create barrier touches the real pipeline
    # could never see on continuous data.
    b = npaths.garch_path(2_000, 6)
    assert np.allclose(b["Open"].to_numpy()[1:], b["Close"].to_numpy()[:-1])


def test_the_bar_range_is_wider_than_its_body():
    # Sub-steps exist so High/Low mean something; without them the
    # barrier walk would understate touches.
    b = npaths.garch_path(5_000, 7)
    body = (b["Close"] - b["Open"]).abs()
    rng = b["High"] - b["Low"]
    # Not every bar: with 10 sub-steps about a tenth are monotone runs
    # where the range IS the body, which is correct, not a defect.
    assert (rng > body).mean() > 0.85
    assert (rng / body.replace(0.0, float("nan"))).median() > 1.3


# ---------------------------------------------------- reproducibility

def test_same_seed_same_path():
    a = npaths.garch_path(500, 11)
    b = npaths.garch_path(500, 11)
    assert a.equals(b)


def test_different_seed_different_path():
    a = npaths.garch_path(500, 11)
    b = npaths.garch_path(500, 12)
    assert not np.allclose(a["Close"], b["Close"])


# --------------------------------------------------------- refusals

def test_an_unknown_regime_is_refused():
    with pytest.raises(ValueError, match="regime"):
        npaths.garch_path(100, 0, regime="stochastic")


def test_a_nonstationary_variance_is_refused():
    # alpha + beta >= 1 has no unconditional level, so the two regimes
    # would not be on the same scale and the control would be void.
    with pytest.raises(ValueError, match="stationary"):
        npaths.garch_path(100, 0, alpha=0.2, beta=0.85)


def test_a_degenerate_length_is_refused():
    with pytest.raises(ValueError):
        npaths.garch_path(1, 0)


def test_clustering_returns_nan_rather_than_lying_on_a_short_series():
    b = npaths.garch_path(3, 0)
    assert np.isnan(npaths.realised_clustering(b, lag=2))
