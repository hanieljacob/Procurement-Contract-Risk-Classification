"""Tests for final cohort assignment and the audit record.

The two properties that matter: the same input must always give the same
output, and nothing may reach ROUTINE by accident.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone

import pandas as pd
import pytest

from procurement_risk import anomaly as A
from procurement_risk import config
from procurement_risk import model as M
from procurement_risk.cleaning import clean_frame
from procurement_risk.cohort import (
    AUDIT_SCHEMA_VERSION,
    PipelineArtefacts,
    ReasonCode,
    classify_contract,
    describe_reason,
)
from procurement_risk.features import build_reference_stats
from procurement_risk.loading import load_raw
from procurement_risk.rules import Cohort

TS = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)

REQUIRED_KEYS = {
    "cohort", "risk_score", "risk_band", "anomaly_flag", "reason_codes",
    "data_quality_flag", "model_version", "feature_snapshot",
    "classification_timestamp",
}


@pytest.fixture(scope="module")
def artefacts():
    clean = clean_frame(load_raw())
    stats = build_reference_stats(clean)
    elig = pd.read_parquet(config.DATA_DIR / "model_eligible.parquet")
    sp = M.split_by_fiscal_year(elig)
    gbt = M.train(sp["train"], sp["validate"], kind="gbt")
    yva = M.build_target(sp["validate"])
    p = gbt.predict_proba(sp["validate"])
    gbt.threshold = M.choose_threshold(yva, p, 0.90)
    gbt.band_edges = M.choose_band_edges(p)
    gbt = M.attach_explainer(gbt, sp["train"].head(500))
    return PipelineArtefacts(stats=stats, model=gbt, detector=A.fit_detector(sp["train"]))


@pytest.fixture(scope="module")
def raw_records():
    return load_raw().head(40).to_dict("records")


# ---------------------------------------------------------------------------
# The required output schema
# ---------------------------------------------------------------------------

def test_output_contains_every_required_field(artefacts, raw_records):
    out = classify_contract(raw_records[1], artefacts, TS)
    assert REQUIRED_KEYS <= set(out)


def test_cohort_is_always_one_of_the_four(artefacts, raw_records):
    valid = {c.name for c in Cohort}
    for rec in raw_records:
        assert classify_contract(rec, artefacts, TS)["cohort"] in valid


def test_risk_band_is_low_medium_or_high_when_scored(artefacts, raw_records):
    for rec in raw_records:
        out = classify_contract(rec, artefacts, TS)
        if out["risk_score"] is not None:
            assert out["risk_band"] in ("LOW", "MEDIUM", "HIGH")


def test_every_reason_code_has_a_written_meaning(artefacts, raw_records):
    for rec in raw_records:
        for code in classify_contract(rec, artefacts, TS)["reason_codes"]:
            meaning = describe_reason(code)
            assert meaning and len(meaning) > 10, code


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_input_gives_identical_output(artefacts, raw_records):
    for rec in raw_records[:10]:
        assert classify_contract(rec, artefacts, TS) == classify_contract(rec, artefacts, TS)


def test_timestamp_is_injected_not_read_from_a_clock(artefacts, raw_records):
    """A function that calls datetime.now() cannot be tested for reproducibility."""
    other = datetime(2020, 1, 1, tzinfo=timezone.utc)
    a = classify_contract(raw_records[1], artefacts, TS)
    b = classify_contract(raw_records[1], artefacts, other)
    assert a["classification_timestamp"] != b["classification_timestamp"]
    assert {k: v for k, v in a.items() if k != "classification_timestamp"} == \
           {k: v for k, v in b.items() if k != "classification_timestamp"}


def test_naive_timestamp_is_rejected(artefacts, raw_records):
    with pytest.raises(ValueError):
        classify_contract(raw_records[1], artefacts, datetime(2026, 8, 23, 12, 0))


def test_input_record_is_not_mutated(artefacts, raw_records):
    rec = copy.deepcopy(raw_records[1])
    before = copy.deepcopy(rec)
    classify_contract(rec, artefacts, TS)
    assert rec == before


def test_audit_record_carries_every_artefact_version(artefacts, raw_records):
    """Recording the model version while the benchmark table changed underneath
    would make a decision look reproducible without being so."""
    v = classify_contract(raw_records[1], artefacts, TS)["artefact_versions"]
    assert set(v) == {"audit_schema", "reference_stats", "model", "detector", "pipeline"}
    assert v["audit_schema"] == AUDIT_SCHEMA_VERSION


def test_snapshot_can_reconstruct_the_decision(artefacts, raw_records):
    """Enough to re-check the arithmetic AND identify which contract it was."""
    snap = classify_contract(raw_records[1], artefacts, TS)["feature_snapshot"]
    assert snap["features"]["amount_usd"] is not None
    assert snap["normalized_inputs"]["supplier_key"] is not None
    assert snap["normalized_inputs"]["benchmark_median"] is not None


# ---------------------------------------------------------------------------
# Safe default -- nothing reaches ROUTINE by accident
# ---------------------------------------------------------------------------

class _BrokenModel:
    """A model artefact that fails the way a real one might in production."""
    version = "broken"
    threshold = 0.5

    def predict_proba(self, df):
        raise RuntimeError("model service unavailable")


def test_unavailable_model_lands_in_high_attention_not_routine(artefacts, raw_records):
    """An unavailable model result must default to HIGH_ATTENTION."""
    broken = PipelineArtefacts(stats=artefacts.stats, model=_BrokenModel(),
                               detector=artefacts.detector)
    seen, flagged_unavailable = set(), 0
    for rec in raw_records:
        out = classify_contract(rec, broken, TS)
        seen.add(out["cohort"])
        assert out["cohort"] != "ROUTINE", "a broken model produced a ROUTINE verdict"
        if ReasonCode.MODEL_UNAVAILABLE.value in out["reason_codes"]:
            flagged_unavailable += 1
    assert "HIGH_ATTENTION" in seen
    assert flagged_unavailable > 0, "the failure was never recorded in an audit record"


def test_missing_model_artefact_lands_in_high_attention(artefacts, raw_records):
    none_model = PipelineArtefacts(stats=artefacts.stats, model=None,
                                   detector=artefacts.detector)
    deferred = 0
    for rec in raw_records:
        out = classify_contract(rec, none_model, TS)
        assert out["cohort"] != "ROUTINE"
        # Records the RULES already decided never reach the model, so they
        # correctly carry no MODEL_UNAVAILABLE code -- only deferred ones do.
        if ReasonCode.MODEL_UNAVAILABLE.value in out["reason_codes"]:
            deferred += 1
            assert out["cohort"] == "HIGH_ATTENTION"
    assert deferred > 0, "no record reached the model stage in this sample"


def test_broken_detector_does_not_produce_routine(artefacts, raw_records):
    class _BrokenDetector:
        version = "broken"
        def is_anomalous(self, df): raise RuntimeError("detector unavailable")
    broken = PipelineArtefacts(stats=artefacts.stats, model=artefacts.model,
                               detector=_BrokenDetector())
    for rec in raw_records:
        assert classify_contract(rec, broken, TS)["cohort"] != "ROUTINE"


def test_record_with_fatal_data_quality_is_not_eligible(artefacts, raw_records):
    rec = dict(raw_records[1]); rec["Supplier Contract Amount (USD)"] = None
    out = classify_contract(rec, artefacts, TS)
    assert out["cohort"] == "NOT_ELIGIBLE"
    assert out["data_quality_flag"] is True
    assert out["risk_score"] is None, "a failed record must not carry a score"


def test_no_record_is_left_unclassified(artefacts, raw_records):
    for rec in raw_records:
        assert classify_contract(rec, artefacts, TS)["cohort"]


# ---------------------------------------------------------------------------
# Reason codes state what is known, and never more
# ---------------------------------------------------------------------------

def test_unknown_supplier_history_is_reported_as_unknown(artefacts):
    """"No prior contracts" would be a false statement about a placeholder supplier."""
    raw = load_raw()
    placeholder = raw[raw["supplier_name_raw"] == "INDIVIDUAL CONSULTANT"].iloc[0].to_dict()
    codes = classify_contract(placeholder, artefacts, TS)["reason_codes"]
    assert ReasonCode.SUPPLIER_HISTORY_UNAVAILABLE.value in codes
    assert ReasonCode.SUPPLIER_HAS_NO_PRIOR_CONTRACTS.value not in codes
    assert ReasonCode.SUPPLIER_HAS_PRIOR_CONTRACTS.value not in codes


def test_no_reason_code_claims_prior_contracts_were_clean():
    """An obvious code to emit would be SUPPLIER_HAS_PRIOR_CLEAN_CONTRACTS.

    Nothing in this extract establishes that any contract was clean -- there are
    no findings, disputes or cancellations, only that contracts existed. Putting
    "clean" in an audit record would assert something the evidence cannot support.
    """
    assert not any("CLEAN" in c.value for c in ReasonCode)
    assert "no outcome data exists" in \
        __import__("procurement_risk.cohort", fromlist=["x"]).REASON_DESCRIPTIONS[
            ReasonCode.SUPPLIER_HAS_PRIOR_CONTRACTS]


def test_routine_records_carry_affirmative_reasons(artefacts):
    """A reviewer told only what did NOT fire learns nothing."""
    raw = load_raw().head(400).to_dict("records")
    routine = [classify_contract(r, artefacts, TS) for r in raw]
    routine = [o for o in routine if o["cohort"] == "ROUTINE"]
    assert routine, "expected at least one routine record in the sample"
    affirmative = {ReasonCode.AMOUNT_WITHIN_CATEGORY_RANGE.value,
                   ReasonCode.COMPETITIVE_PROCUREMENT_METHOD.value,
                   ReasonCode.WITHIN_TRAINING_DISTRIBUTION.value,
                   ReasonCode.ESTABLISHED_PROJECT.value,
                   ReasonCode.DOMESTIC_SUPPLIER.value}
    for out in routine:
        assert set(out["reason_codes"]) & affirmative
