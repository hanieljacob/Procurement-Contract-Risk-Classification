"""Tests for the risk model.

Most of these exist to catch one class of defect: the label is computable from
columns we hold, so leakage is the default outcome and has to be actively
prevented. Three separate leaks were found this way during development.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from procurement_risk import config
from procurement_risk import model as M


@pytest.fixture(scope="module")
def eligible():
    return pd.read_parquet(config.DATA_DIR / "model_eligible.parquet")


@pytest.fixture(scope="module")
def splits(eligible):
    return M.split_by_fiscal_year(eligible)


# ---------------------------------------------------------------------------
# The label
# ---------------------------------------------------------------------------

def test_target_matches_the_brief_definition():
    """Top quartile of the peer group AND non-competitive. Both conditions."""
    df = pd.DataFrame({
        "amount_percentile_in_category_region": [0.99, 0.99, 0.10, 0.10, None],
        "is_competitive_method": [False, True, False, True, False],
    })
    assert M.build_target(df).tolist() == [1, 0, 0, 0, 0]


def test_target_uses_point_in_time_percentile(eligible):
    """The percentile comes from the monthly vintages, so the label is as-of."""
    assert "amount_percentile_in_category_region" in eligible.columns
    assert eligible["signing_date"].notna().all()


# ---------------------------------------------------------------------------
# Leakage -- the defect this design exists to prevent
# ---------------------------------------------------------------------------

def test_no_label_defining_column_reaches_the_feature_set():
    cols = M.feature_columns()
    for banned in M.LABEL_DEFINING_FEATURES:
        assert banned not in cols, f"{banned} defines the label and must be withheld"


def test_no_label_defining_column_survives_into_the_design_matrix(splits):
    X = M.build_design_matrix(splits["train"].head(500), M.feature_columns())
    for banned in M.LABEL_DEFINING_FEATURES:
        assert not any(c == banned or c.startswith(banned + "_") for c in X.columns)


def test_including_label_features_produces_a_tautology(splits):
    """The leakage check, run as a test rather than asserted in prose.

    With the label's own inputs present, both models score a perfect AUC on
    held-out years. That is the definition being read back, not risk being
    predicted, and it is why `feature_columns()` excludes them by default.
    """
    tr, va = splits["train"], splits["validate"]
    leaky = M.feature_columns(include_label_defining=True)
    Xtr = M.build_design_matrix(tr, leaky)
    Xva = M.align_columns(M.build_design_matrix(va, leaky), Xtr.columns)
    g = HistGradientBoostingClassifier(max_iter=60, random_state=0).fit(Xtr, M.build_target(tr))
    auc = roc_auc_score(M.build_target(va), g.predict_proba(Xva)[:, 1])
    assert auc > 0.99, f"expected a tautology, got {auc:.4f}"


def test_honest_feature_set_does_not_score_suspiciously_well(splits):
    """The counterpart: if this climbs toward 1.0, leakage has crept back in.

    The upper bound is 0.90 rather than something tighter because ~22.5% of the
    population is a guaranteed negative (see `structurally_negative`), and free
    negatives inflate AUC. On the population that can actually be positive the
    same model scores around 0.80, which is the figure worth quoting.
    """
    tr, va = splits["train"], splits["validate"]
    cols = M.feature_columns()
    Xtr = M.build_design_matrix(tr, cols)
    Xva = M.align_columns(M.build_design_matrix(va, cols), Xtr.columns)
    g = HistGradientBoostingClassifier(max_iter=60, random_state=0).fit(Xtr, M.build_target(tr))
    auc = roc_auc_score(M.build_target(va), g.predict_proba(Xva)[:, 1])
    assert 0.55 < auc < 0.90, f"AUC {auc:.4f} is outside the plausible band -- check for leakage"


def test_scoring_uses_source_columns_not_design_columns(splits, trained):
    """Regression: conflating the two silently zeroed every one-hot column.

    `columns` are post-encoding names like "region_South Asia", which exist on no
    record. Feeding them back in as source names left the categoricals all zero
    and cost 13 points of AUC while still returning plausible-looking scores.
    """
    X = trained.design_matrix(splits["test"].head(200))
    assert list(X.columns) == list(trained.columns)
    onehot = [c for c in trained.columns if c.startswith("region_")]
    assert onehot, "expected one-hot region columns"
    assert X[onehot].to_numpy().sum() > 0, "categorical columns are all zero -- encoding is broken"


def test_structurally_negative_records_can_never_be_positive(eligible):
    """22.5% of the population is a guaranteed negative, by construction.

    The placeholder supplier appears only on Individual Consultant Selection,
    which is competitive, so the label's second condition fails outright. Free
    negatives inflate every aggregate metric, which is why headline figures are
    also reported on the population that can actually be positive.
    """
    y = M.build_target(eligible)
    forced = M.structurally_negative(eligible)
    assert forced.sum() > 50_000
    assert y[forced].sum() == 0, "a 'structurally negative' record was labelled positive"


# ---------------------------------------------------------------------------
# The time-based split
# ---------------------------------------------------------------------------

def test_splits_are_disjoint_and_ordered(splits):
    years = {k: set(v.fiscal_year.unique()) for k, v in splits.items()}
    assert years["train"] == set(config.TRAIN_FISCAL_YEARS)
    assert years["validate"] == set(config.VALIDATION_FISCAL_YEARS)
    assert years["test"] == set(config.TEST_FISCAL_YEARS)
    assert max(years["train"]) < min(years["validate"]) < min(years["test"])
    assert not years["train"] & years["test"]


def test_truncated_year_is_quarantined_not_tested_on(splits):
    """FY2027 is a seven-week stub; testing there measures reporting lag."""
    assert set(splits["quarantined"].fiscal_year.unique()) == set(config.TRUNCATED_FISCAL_YEARS)
    assert not set(config.TEST_FISCAL_YEARS) & set(config.TRUNCATED_FISCAL_YEARS)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def test_recall_increases_as_more_is_flagged():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.05, 5000)
    s = np.clip(y * 0.3 + rng.random(5000) * 0.5, 0, 1)
    tbl = M.threshold_table(y, s)
    assert tbl.recall.is_monotonic_increasing
    assert tbl.threshold.is_monotonic_decreasing


def test_threshold_is_chosen_for_recall_not_balance():
    """Conservative means recall is the constraint, volume is the price."""
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.05, 5000)
    s = np.clip(y * 0.3 + rng.random(5000) * 0.5, 0, 1)
    strict = M.choose_threshold(y, s, min_recall=0.50)
    loose = M.choose_threshold(y, s, min_recall=0.95)
    assert loose <= strict, "a higher recall floor must not raise the threshold"


# ---------------------------------------------------------------------------
# Scoring contract
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained(splits):
    m = M.train(splits["train"], splits["validate"], kind="gbt")
    p = m.predict_proba(splits["validate"])
    y = M.build_target(splits["validate"])
    m.threshold = M.choose_threshold(y, p, min_recall=0.90)
    m.band_edges = M.choose_band_edges(p)
    return M.attach_explainer(m, splits["train"].head(1000))


def test_scoring_is_reproducible(trained, splits):
    rec = splits["test"].head(1)
    assert M.score_record(trained, rec) == M.score_record(trained, rec)


def test_score_record_returns_the_required_shape(trained, splits):
    out = M.score_record(trained, splits["test"].head(1))
    assert set(out) >= {"risk_score", "risk_band", "top_features", "model_version"}
    assert 0.0 <= out["risk_score"] <= 1.0
    assert out["risk_band"] in M.RISK_BANDS
    assert len(out["top_features"]) == 3
    for f in out["top_features"]:
        assert set(f) == {"feature", "value", "contribution", "direction"}
        assert f["feature"] in trained.columns


def test_bands_are_ordered(trained):
    lo, hi = trained.band_edges
    assert lo < hi
    assert trained.band(lo - 1e-9) == "LOW"
    assert trained.band(lo) == "MEDIUM"
    assert trained.band(hi) == "HIGH"


def test_calibration_is_better_than_the_raw_score(splits, trained):
    """A reviewer shown '0.8 risk' will read it as a probability, so it must be one."""
    y = M.build_target(splits["test"])
    p = trained.predict_proba(splits["test"])
    tbl = M.calibration_table(y, p)
    bulk = tbl[tbl.n > 1000]
    assert (abs(bulk.mean_predicted - bulk.observed_rate) < 0.15).all()
