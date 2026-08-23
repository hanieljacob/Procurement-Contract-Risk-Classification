"""Tests for the deterministic rule engine."""

from __future__ import annotations

import pandas as pd
import pytest

from procurement_risk import config
from procurement_risk.cleaning import clean_frame
from procurement_risk.features import build_reference_stats, engineer_features
from procurement_risk.loading import load_raw
from procurement_risk.pipeline import EnrichedRecord, validate_and_enrich
from procurement_risk.quality import DataQualityFlag as F
from procurement_risk.rules import (
    EXCEPTIONAL_RULES,
    Cohort,
    _kleene_and,
    apply_rules,
    rule_catalogue,
)


def _enriched(**features) -> EnrichedRecord:
    """A scoreable record whose features can be set directly."""
    base = {
        "amount_usd": 50_000.0,
        "amount_percentile_in_category_region": 0.5,
        "amount_vs_category_region_median": 1.0,
        "is_competitive_method": True,
        "is_first_contract_in_project": False,
        "supplier_in_secrecy_jurisdiction": False,
        "supplier_is_domestic": True,
        "days_into_fiscal_year": 100,
    }
    base.update(features)
    return EnrichedRecord(ok=True, features=base, data_quality_flags=[], normalized={})


# ---------------------------------------------------------------------------
# Three-valued logic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a,b,expected",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
        (False, None, False),   # the one that matters
        (None, False, False),   # ...in both orders
        (True, None, None),
        (None, True, None),
        (None, None, None),
    ],
)
def test_kleene_and_truth_table(a, b, expected):
    """False AND unknown is False, not unknown.

    Poisoning a conjunction with any unknown operand sent 9,666 records to a
    human because a country code was missing -- on a rule that could not have
    fired anyway. Conservatism means resolving genuine ambiguity upward, not
    manufacturing ambiguity the data has already settled.
    """
    assert _kleene_and(a, b) is expected


def test_definitively_false_operand_does_not_defer_to_a_human():
    """A supplier demonstrably not offshore cannot trigger the offshore rule."""
    outcome = apply_rules(_enriched(
        supplier_in_secrecy_jurisdiction=False,
        supplier_is_domestic=None,  # unknown, and irrelevant here
    ))
    assert outcome.proceeds_to_model
    assert "OFFSHORE_SUPPLIER_FOREIGN_TO_BORROWER" not in outcome.undecidable


# ---------------------------------------------------------------------------
# Cohort precedence
# ---------------------------------------------------------------------------

def test_cohort_precedence_is_ordered():
    assert Cohort.NOT_ELIGIBLE < Cohort.EXCEPTIONAL < Cohort.HIGH_ATTENTION < Cohort.ROUTINE
    assert min(Cohort.ROUTINE, Cohort.EXCEPTIONAL) is Cohort.EXCEPTIONAL
    assert str(Cohort.HIGH_ATTENTION) == "HIGH_ATTENTION"


def test_unscoreable_record_is_not_eligible_regardless_of_other_rules():
    """A failed record never reaches the exception rules; it has no features."""
    rec = EnrichedRecord(
        ok=False, features=None,
        data_quality_flags=[F.AMOUNT_NON_POSITIVE.value, F.SUPPLIER_MISSING.value],
        normalized={},
    )
    outcome = apply_rules(rec)
    assert outcome.cohort is Cohort.NOT_ELIGIBLE
    assert set(outcome.reason_codes) == {"AMOUNT_NON_POSITIVE", "SUPPLIER_MISSING"}


# ---------------------------------------------------------------------------
# Each rule fires when it should, and stays silent otherwise
# ---------------------------------------------------------------------------

def test_clean_record_triggers_nothing_and_proceeds_to_model():
    outcome = apply_rules(_enriched())
    assert outcome.proceeds_to_model
    assert outcome.cohort is None
    assert outcome.triggered == []


@pytest.mark.parametrize(
    "rule_id,features",
    [
        ("AMOUNT_EXTREME_FOR_PEER_GROUP",
         {"amount_percentile_in_category_region": 0.995}),
        ("NON_COMPETITIVE_HIGH_VALUE",
         {"is_competitive_method": False, "amount_usd": 5_000_000.0}),
        ("OFFSHORE_SUPPLIER_FOREIGN_TO_BORROWER",
         {"supplier_in_secrecy_jurisdiction": True, "supplier_is_domestic": False}),
        ("FIRST_CONTRACT_IN_PROJECT_HIGH_VALUE",
         {"is_first_contract_in_project": True,
          "amount_vs_category_region_median": 50.0}),
    ],
)
def test_each_exceptional_rule_fires(rule_id, features):
    outcome = apply_rules(_enriched(**features))
    assert outcome.cohort is Cohort.EXCEPTIONAL
    assert rule_id in outcome.triggered
    assert rule_id in outcome.reason_codes


def test_non_competitive_below_threshold_does_not_fire():
    """It is the combination with scale that matters, not the method alone."""
    outcome = apply_rules(_enriched(is_competitive_method=False, amount_usd=1_000.0))
    assert outcome.proceeds_to_model


def test_offshore_but_domestic_does_not_fire():
    """Several secrecy jurisdictions are borrowers; a domestic award is ordinary."""
    outcome = apply_rules(_enriched(
        supplier_in_secrecy_jurisdiction=True, supplier_is_domestic=True))
    assert outcome.proceeds_to_model


# ---------------------------------------------------------------------------
# Safe default
# ---------------------------------------------------------------------------

def test_unevaluable_rule_defaults_to_high_attention_never_routine():
    """A record with no benchmark carries an unknown where a control expected
    an answer, so it goes to a person rather than to the model."""
    outcome = apply_rules(_enriched(
        amount_percentile_in_category_region=None,
        amount_vs_category_region_median=None,
    ))
    assert outcome.cohort is Cohort.HIGH_ATTENTION
    assert not outcome.proceeds_to_model
    assert "AMOUNT_EXTREME_FOR_PEER_GROUP" in outcome.undecidable


def test_a_fired_rule_outranks_an_unevaluable_one():
    """Knowing a control fired beats not knowing whether another one did."""
    outcome = apply_rules(_enriched(
        is_competitive_method=False, amount_usd=5_000_000.0,
        amount_percentile_in_category_region=None,
    ))
    assert outcome.cohort is Cohort.EXCEPTIONAL


# ---------------------------------------------------------------------------
# Documentation contract
# ---------------------------------------------------------------------------

def test_every_rule_is_fully_documented():
    """The brief requires condition, threshold and why-it-is-a-rule for each."""
    for entry in rule_catalogue():
        assert entry["threshold"], f"{entry['rule_id']} has no stated threshold"
        for field in ("condition", "rationale", "why_hard_rule"):
            assert len(entry[field]) > 30, f"{entry['rule_id']}.{field} is not an explanation"


def test_rule_ids_are_unique():
    ids = [r.id for r in EXCEPTIONAL_RULES]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Integration against the real extract
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scored():
    clean = clean_frame(load_raw())
    stats = build_reference_stats(clean)
    fe = engineer_features(clean, stats)
    outcomes = [apply_rules(validate_and_enrich(r, stats)) for r in fe.to_dict("records")]
    return fe, outcomes


def test_every_record_receives_exactly_one_outcome(scored):
    """The pipeline must never leave a record unclassified."""
    fe, outcomes = scored
    assert len(outcomes) == len(fe)
    for o in outcomes:
        assert o.cohort is not None or o.proceeds_to_model


def test_not_eligible_reconciles_to_the_documented_count(scored):
    _, outcomes = scored
    assert sum(o.cohort is Cohort.NOT_ELIGIBLE for o in outcomes) == 13


def test_exceptional_volume_stays_reviewable(scored):
    """A control routing a fifth of the portfolio to senior review is not a
    control. If this drifts outside the band the thresholds need revisiting."""
    fe, outcomes = scored
    share = sum(o.cohort is Cohort.EXCEPTIONAL for o in outcomes) / len(fe)
    assert 0.01 <= share <= 0.03, f"EXCEPTIONAL is {share:.2%} of the portfolio"


def test_fiscal_year_window_rule_never_fires_on_this_extract(scored):
    """Proving the documented claim rather than asserting it.

    The publisher derives Fiscal Year from the signing date, so no record can
    fall outside its own window. The rule is retained as a guard for unvalidated
    upstream data and reported as a control with no coverage here.
    """
    _, outcomes = scored
    fired = sum("SIGNED_OUTSIDE_FISCAL_YEAR_WINDOW" in o.triggered for o in outcomes)
    assert fired == 0
    inactive = [r for r in EXCEPTIONAL_RULES if not r.active]
    assert [r.id for r in inactive] == ["SIGNED_OUTSIDE_FISCAL_YEAR_WINDOW"]


def test_records_without_a_benchmark_never_reach_the_model(scored):
    """A record with no peer history is decided by a person, not by a model.

    Which cohort it lands in still depends on the rules: one that fires an
    exception despite the missing benchmark is EXCEPTIONAL, since a control that
    definitively fired outranks one we could not evaluate. What must never happen
    is the record proceeding to the model as though nothing were unknown.
    """
    fe, outcomes = scored
    no_benchmark = fe["benchmark_median"].isna().to_numpy()
    decided = {Cohort.NOT_ELIGIBLE, Cohort.EXCEPTIONAL, Cohort.HIGH_ATTENTION}
    checked = 0
    for row_is_warmup, o in zip(no_benchmark, outcomes):
        if row_is_warmup:
            checked += 1
            assert not o.proceeds_to_model, "warm-up record reached the model"
            assert o.cohort in decided
    assert checked > 2_000, "expected the FY2020 warm-up population"


def test_rule_engine_is_deterministic(scored):
    fe, outcomes = scored
    clean = clean_frame(load_raw())
    stats = build_reference_stats(clean)
    sample = fe.head(500).to_dict("records")
    again = [apply_rules(validate_and_enrich(r, stats)) for r in sample]
    for a, b in zip(outcomes[:500], again):
        assert a.cohort == b.cohort and a.triggered == b.triggered
