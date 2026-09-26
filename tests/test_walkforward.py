"""
Tests for fxrisk.research.walkforward.

A walk-forward harness has exactly one job: make it impossible for
the test window to influence the choice made on the training window.
If it fails at that it is worse than no harness at all, because it
produces in-sample numbers wearing an out-of-sample label. The first
four tests attack that and nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxrisk.research import walkforward as wf


def _trades(configs=("a", "b"), n=400, seed=0, start="2022-01-01"):
    rng = np.random.default_rng(seed)
    rows = []
    for c in configs:
        t = pd.date_range(start, periods=n, freq="12h", tz="America/New_York")
        rows.append(pd.DataFrame({"config": c, "entry_time": t,
                                  "unit_net_R": rng.normal(0, 1, n)}))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------
# The leak
# ------------------------------------------------------------

def test_a_fold_that_overlaps_is_refused_at_construction():
    ts = pd.Timestamp
    with pytest.raises(ValueError, match="leak"):
        wf.Fold(ts("2022-01-01"), ts("2022-07-01"),
                ts("2022-04-01"), ts("2022-10-01"))   # test starts mid-train


def test_selection_ignores_everything_after_train_end():
    """
    The decisive test. Config 'b' is made spectacular in the TEST
    window only. A correct harness must still choose 'a'.
    """
    t = _trades(("a", "b"), n=400, seed=1)
    f = wf.make_folds("2022-01-01", "2022-10-01", train_months=3,
                      test_months=3, tz="America/New_York")[0]
    t.loc[(t["config"] == "a") & (t["entry_time"] < f.train_end),
          "unit_net_R"] = 1.0
    t.loc[(t["config"] == "b") & (t["entry_time"] >= f.test_start),
          "unit_net_R"] = 99.0
    assert wf.select(t, f, wf.mean_net_r) == "a"


def test_reported_results_contain_only_test_segments():
    t = _trades(n=600, seed=2)
    folds = wf.make_folds("2022-01-01", "2023-01-01", train_months=3,
                          test_months=3, tz="America/New_York")
    res = wf.run(t, folds)
    o = res["oos"]
    if o.empty:
        pytest.skip("no out-of-sample trades on this fixture")
    for f in folds:
        inside = o[(o["fold_test_start"] == f.test_start)]
        assert (inside["entry_time"] >= f.test_start).all()
        assert (inside["entry_time"] < f.test_end).all()


def test_test_segments_do_not_overlap_each_other():
    """Concatenating overlapping segments double-counts trades."""
    t = _trades(n=800, seed=3)
    folds = wf.make_folds("2022-01-01", "2023-06-01", train_months=6,
                          test_months=3, tz="America/New_York")
    res = wf.run(t, folds)
    o = res["oos"]
    if o.empty:
        pytest.skip("no out-of-sample trades")
    assert not o.duplicated(subset=["config", "entry_time"]).any()


# ------------------------------------------------------------
# Fold construction
# ------------------------------------------------------------

def test_folds_are_adjacent_and_cover_the_span_once():
    folds = wf.make_folds("2021-09-01", "2026-09-01",
                          train_months=18, test_months=6)
    assert len(folds) == 7
    for a, b in zip(folds, folds[1:]):
        assert a.test_end == b.test_start


def test_no_fold_runs_past_the_end_of_the_data():
    end = pd.Timestamp("2024-01-01")
    for f in wf.make_folds("2022-01-01", end, train_months=6, test_months=3):
        assert f.test_end <= end


def test_fold_parameters_are_validated():
    for kw in ({"train_months": 0}, {"test_months": 0}, {"step_months": 0}):
        with pytest.raises(ValueError):
            wf.make_folds("2022-01-01", "2024-01-01", **kw)


# ------------------------------------------------------------
# Selection behaviour
# ------------------------------------------------------------

def test_a_training_window_too_thin_to_judge_is_refused():
    """20 trades cannot rank configurations; scoring them anyway is
    how a walk-forward starts choosing noise."""
    t = _trades(("a",), n=20, seed=4)
    assert wf.mean_net_r(t) == float("-inf")


def test_ties_break_deterministically_not_by_grid_order():
    t = _trades(("b", "a"), n=200, seed=5)
    f = wf.make_folds("2022-01-01", "2022-07-01", train_months=3,
                      test_months=3, tz="America/New_York")[0]
    t["unit_net_R"] = 1.0                       # exact tie
    assert wf.select(t, f, wf.mean_net_r) == "a"
    assert wf.select(t.iloc[::-1], f, wf.mean_net_r) == "a"


def test_selection_stability_is_reported():
    t = _trades(("a", "b"), n=900, seed=6)
    t.loc[t["config"] == "a", "unit_net_R"] += 0.5   # 'a' always better
    folds = wf.make_folds("2022-01-01", "2023-06-01", train_months=6,
                          test_months=3, tz="America/New_York")
    res = wf.run(t, folds)
    assert res["most_chosen"] == "a"
    assert res["selection_stability"] == pytest.approx(1.0)
    assert res["distinct_configs"] == 1


def test_the_fixed_comparison_is_scored_on_the_same_segments():
    """Without this the 'does selection help' question is unanswerable."""
    t = _trades(("a", "b"), n=900, seed=7)
    folds = wf.make_folds("2022-01-01", "2023-06-01", train_months=6,
                          test_months=3, tz="America/New_York")
    res = wf.run(t, folds, fixed_config="a")
    s = wf.summarise(res)
    if res["fixed_oos"].empty:
        pytest.skip("no fixed-config trades")
    assert "selection_edge" in s
    assert s["selection_edge"] == pytest.approx(
        s["mean_net_R"] - s["fixed_mean_net_R"])


def test_summarise_on_nothing_is_not_a_crash():
    assert wf.summarise({"oos": pd.DataFrame()})["trades"] == 0
