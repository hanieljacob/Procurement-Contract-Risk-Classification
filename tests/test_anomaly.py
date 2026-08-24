"""Tests for the out-of-distribution check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from procurement_risk import anomaly as A
from procurement_risk import config
from procurement_risk import model as M
from procurement_risk.rules import Cohort


@pytest.fixture(scope="module")
def eligible():
    return pd.read_parquet(config.DATA_DIR / "model_eligible.parquet")


@pytest.fixture(scope="module")
def splits(eligible):
    return M.split_by_fiscal_year(eligible)


@pytest.fixture(scope="module")
def detector(splits):
    return A.fit_detector(splits["train"])


# ---------------------------------------------------------------------------
# Fitting discipline
# ---------------------------------------------------------------------------

def test_detector_is_fitted_on_training_years_only(splits, detector):
    """"Unusual relative to the training population" needs the population to
    predate what it judges -- otherwise a contract helps define the distribution
    it is then measured against."""
    flagged = detector.is_anomalous(splits["train"])
    assert abs(flagged.mean() - A.CONTAMINATION) < 0.005


def test_flag_rate_rises_on_later_years(splits, detector):
    """Drift is a finding, not a defect: an OOD check should notice the
    portfolio moving away from what it was fitted on."""
    train_rate = detector.is_anomalous(splits["train"]).mean()
    test_rate = detector.is_anomalous(splits["test"]).mean()
    assert test_rate > train_rate


def test_flagged_records_are_enriched_in_high_attention(splits, detector):
    """Not a requirement, but a sanity check: anomalies should not be noise."""
    te = splits["test"]
    y = M.build_target(te)
    flagged = detector.is_anomalous(te)
    assert y[flagged].mean() > y.mean() * 2


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

def test_every_flagged_record_gets_an_explanation(splits, detector):
    te = splits["test"]
    anoms = te[detector.is_anomalous(te)].head(300)
    for _, row in anoms.iterrows():
        sentence = detector.describe(row)
        assert sentence.endswith(".")
        assert 40 < len(sentence) < 400
        assert sentence[0].isupper()


def test_explanations_are_deterministic(splits, detector):
    row = splits["test"].iloc[0]
    assert detector.describe(row) == detector.describe(row)


def test_common_values_are_not_reported_as_extreme(detector):
    """Regression: searchsorted defaults to side='left', which counts values
    STRICTLY less than the input -- so a value shared by most of the population
    landed at percentile 0 and read as maximally unusual. That had the
    explanation announcing "a single-supplier award" (93% of contracts) as the
    most remarkable feature of nearly every flagged record.
    """
    extremity, _ = detector._extremity("consortium_size", 1.0)
    assert extremity < 0.2, f"the ordinary case scored extremity {extremity:.3f}"


def test_explanation_names_different_kinds_of_unusual(splits, detector):
    """At most one phrase per feature family, so a sentence is informative."""
    te = splits["test"]
    anoms = te[detector.is_anomalous(te)].head(200)
    for _, row in anoms.iterrows():
        s = detector.describe(row)
        # the two amount phrasings must never both appear
        assert not ("above the median for its category and region" in s
                    and "at the top of its peer group" in s)


def test_pipeline_internals_are_not_offered_as_reasons(detector):
    """`benchmark_support_n` is a property of our own reference table, not of
    the contract. It stays in the detector but must never reach a reviewer."""
    assert "benchmark_support_n" not in A._PHRASING


# ---------------------------------------------------------------------------
# The anomaly override
# ---------------------------------------------------------------------------

def test_anomalous_routine_becomes_high_attention():
    assert A.apply_anomaly_override(Cohort.ROUTINE, True) is Cohort.HIGH_ATTENTION


def test_override_never_downgrades_a_stronger_verdict():
    """It only ever moves a record up the precedence order."""
    assert A.apply_anomaly_override(Cohort.EXCEPTIONAL, True) is Cohort.EXCEPTIONAL
    assert A.apply_anomaly_override(Cohort.NOT_ELIGIBLE, True) is Cohort.NOT_ELIGIBLE


def test_override_is_inert_when_not_anomalous():
    for c in (Cohort.ROUTINE, Cohort.HIGH_ATTENTION, Cohort.EXCEPTIONAL):
        assert A.apply_anomaly_override(c, False) is c


# ---------------------------------------------------------------------------
# Per-record contract
# ---------------------------------------------------------------------------

def test_assess_returns_the_expected_shape(splits, detector):
    te = splits["test"]
    flagged_idx = np.flatnonzero(detector.is_anomalous(te))[0]
    out = A.assess(detector, te.iloc[[flagged_idx]])
    assert out["anomaly_flag"] is True
    assert isinstance(out["anomaly_score"], float)
    assert out["anomaly_explanation"].endswith(".")
    assert out["detector_version"] == A.DETECTOR_VERSION


def test_unflagged_record_carries_no_explanation(splits, detector):
    te = splits["test"]
    ok_idx = np.flatnonzero(~detector.is_anomalous(te))[0]
    out = A.assess(detector, te.iloc[[ok_idx]])
    assert out["anomaly_flag"] is False
    assert out["anomaly_explanation"] is None


def test_detector_is_reproducible(splits):
    a = A.fit_detector(splits["train"], random_state=0)
    b = A.fit_detector(splits["train"], random_state=0)
    sample = splits["test"].head(500)
    assert np.array_equal(a.is_anomalous(sample), b.is_anomalous(sample))
