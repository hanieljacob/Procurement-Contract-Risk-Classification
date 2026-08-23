"""Tests for record validation and feature preparation.

Two layers:
  * unit tests on a small synthetic frame -- fast, and they pin the exact edge
    cases the design is built around (missing values, placeholders, ties);
  * integration tests against the real extract -- they assert the reconciliation
    numbers quoted in the documentation, so the narrative cannot silently go
    stale as the code changes.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from procurement_risk import config
from procurement_risk.cleaning import (
    _is_missing,
    normalize_country_code,
    classify_method,
    clean_frame,
    contract_grain,
    days_into_fiscal_year,
    fiscal_year_of,
    is_regional_borrower,
    normalize_supplier_name,
    primary_global_practice,
    resolve_supplier_key,
)
from procurement_risk.features import (
    FEATURE_COLUMNS,
    PointInTimeCounter,
    build_reference_stats,
    engineer_features,
)
from procurement_risk.loading import load_raw
from procurement_risk.pipeline import validate_and_enrich
from procurement_risk.quality import DataQualityFlag as F, SEVERITY, DESCRIPTIONS, is_fatal


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

def _row(**kw):
    base = dict(
        as_of_date="08/22/2026",
        fiscal_year_published=2021,
        region="South Asia",
        borrower_country="India",
        borrower_country_code="IN",
        project_id="P0001",
        project_name="Test Project",
        global_practice_raw="Transportation",
        procurement_category="Goods",
        procurement_method_raw="Request for Bids",
        contract_id=1,
        contract_description="d",
        borrower_reference="r",
        signing_date_raw="01/15/2021",
        supplier_id="S1",
        supplier_name_raw="ACME TRADING CO LTD",
        supplier_country="India",
        supplier_country_code="IN",
        amount_usd=100_000.0,
        review_type="Prior",
        signed_calendar_year=2021,
    )
    base.update(kw)
    return base


@pytest.fixture(scope="module")
def synthetic():
    rows = []
    for i in range(60):  # enough to clear MIN_GROUP_SUPPORT
        rows.append(_row(contract_id=100 + i, amount_usd=10_000.0 * (i + 1),
                         supplier_name_raw=f"SUPPLIER {i}", supplier_id=f"S{i}"))
    return clean_frame(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def synthetic_stats(synthetic):
    return build_reference_stats(synthetic, median_fiscal_years=(2021,))


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

def test_method_taxonomy_covers_every_observed_value():
    """Every method in the extract maps; an unknown one must NOT be guessed."""
    df = load_raw()
    observed = set(df["procurement_method_raw"].dropna().unique())
    unmapped = {m for m in observed if config.method_lookup_key(m) not in config.METHOD_CLASS}
    assert unmapped == set(), f"unmapped procurement methods: {unmapped}"


def test_double_space_method_still_maps():
    """Regression: the source spells CQS with a double space.

    Whitespace-collapsing cleaners previously broke the exact-string lookup and
    wrongly failed 15,956 records (5.5%) as 'unmapped method'.
    """
    assert classify_method("Consultant Qualification  Selection")[0] == config.COMPETITIVE
    assert classify_method("Consultant Qualification Selection")[0] == config.COMPETITIVE
    assert classify_method("  direct   selection ")[0] == config.NON_COMPETITIVE


def test_unknown_method_is_flagged_not_guessed():
    cls, flags = classify_method("Interpretive Dance Selection")
    assert cls is None and F.PROCUREMENT_METHOD_UNMAPPED in flags


def test_quality_vocabulary_is_complete():
    for flag in F:
        assert flag in SEVERITY and flag in DESCRIPTIONS


# ---------------------------------------------------------------------------
# Missing values must never become zeros
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, float("nan"), pd.NA, "", "   "])
def test_all_missing_disguises_are_recognised(value):
    """None, NaN and pd.NA all arrive depending on the caller; all must match."""
    assert _is_missing(value) is True


def test_placeholder_supplier_is_not_an_entity():
    key, flags = resolve_supplier_key("INDIVIDUAL CONSULTANT")
    assert key is None
    assert F.SUPPLIER_UNIDENTIFIABLE in flags


def test_placeholder_supplier_prior_count_is_none_not_zero(synthetic_stats):
    """The distinction the whole pipeline turns on: unknown != no history."""
    rec = _row(supplier_name_raw="INDIVIDUAL CONSULTANT",
               procurement_method_raw="Individual Consultant Selection",
               procurement_category="Consultant Services")
    res = validate_and_enrich(rec, synthetic_stats)
    assert res.ok
    assert res.features["supplier_prior_contract_count"] is None
    assert res.features["supplier_is_known"] is False
    assert F.SUPPLIER_UNIDENTIFIABLE.value in res.data_quality_flags


def test_regional_borrower_domesticity_is_none_not_false(synthetic_stats):
    """A DRC-Angola programme has no 'domestic supplier' answer."""
    assert is_regional_borrower("DRC - Angola", None) is True
    assert is_regional_borrower("Kenya", "KE") is False
    rec = _row(borrower_country="Western Balkans", borrower_country_code=None)
    res = validate_and_enrich(rec, synthetic_stats)
    assert res.features["supplier_is_domestic"] is None
    assert F.BORROWER_IS_REGIONAL.value in res.data_quality_flags


def test_missing_practice_degrades_feature_without_failing_record(synthetic_stats):
    res = validate_and_enrich(_row(global_practice_raw=None), synthetic_stats)
    assert res.ok is True
    assert res.features["amount_vs_practice_median"] is None
    assert F.GLOBAL_PRACTICE_MISSING.value in res.data_quality_flags


# ---------------------------------------------------------------------------
# Required fields fail rather than impute
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "override,expected",
    [
        ({"amount_usd": 0.0}, F.AMOUNT_NON_POSITIVE),
        ({"amount_usd": -5.0}, F.AMOUNT_NON_POSITIVE),
        ({"amount_usd": None}, F.AMOUNT_MISSING),
        ({"amount_usd": "not a number"}, F.AMOUNT_UNPARSEABLE),
        ({"supplier_name_raw": None}, F.SUPPLIER_MISSING),
        ({"borrower_country": None}, F.BORROWER_COUNTRY_MISSING),
        ({"procurement_category": None}, F.PROCUREMENT_CATEGORY_MISSING),
        ({"signing_date_raw": None}, F.SIGNING_DATE_MISSING),
        ({"signing_date_raw": "2021-01-15"}, F.SIGNING_DATE_UNPARSEABLE),
        ({"procurement_method_raw": None}, F.PROCUREMENT_METHOD_MISSING),
        ({"procurement_method_raw": "Vibes"}, F.PROCUREMENT_METHOD_UNMAPPED),
    ],
)
def test_required_field_problem_fails_record(override, expected, synthetic_stats):
    res = validate_and_enrich(_row(**override), synthetic_stats)
    assert res.ok is False
    assert res.features is None, "a failed record must not carry features"
    assert expected.value in res.data_quality_flags
    assert is_fatal(res.data_quality_flags)


# ---------------------------------------------------------------------------
# Point-in-time correctness
# ---------------------------------------------------------------------------

def test_counter_excludes_same_day_and_later():
    keys = pd.Series(["A", "A", "A", "B"])
    dates = pd.Series(pd.to_datetime(["2021-01-01", "2021-06-01", "2021-06-01", "2021-03-01"]))
    c = PointInTimeCounter(keys, dates)
    assert c.count_before("A", date(2020, 12, 31)) == 0
    assert c.count_before("A", date(2021, 6, 1)) == 1, "same-day contracts are not prior"
    assert c.count_before("A", date(2021, 6, 2)) == 3
    assert c.count_before("B", date(2022, 1, 1)) == 1


def test_counter_returns_none_for_unresolvable_key():
    c = PointInTimeCounter(pd.Series(["A"]), pd.Series(pd.to_datetime(["2021-01-01"])))
    assert c.count_before(None, date(2021, 5, 1)) is None
    assert c.count_before(pd.NA, date(2021, 5, 1)) is None
    assert c.count_before("A", None) is None
    assert c.count_before("NEVER SEEN", date(2021, 5, 1)) == 0, "known key, genuine zero"


def test_no_future_leakage_in_supplier_history(synthetic_stats):
    """A supplier's earliest contract can never have prior contracts."""
    c = synthetic_stats.supplier_history
    for key in list(c._index)[:20]:
        earliest = c._earliest[c._index[key]]
        assert c.count_before(key, date.fromordinal(earliest)) == 0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_input_yields_identical_output(synthetic_stats):
    rec = _row()
    a = validate_and_enrich(rec, synthetic_stats)
    b = validate_and_enrich(rec, synthetic_stats)
    assert a.features == b.features
    assert a.data_quality_flags == b.data_quality_flags
    assert a.normalized == b.normalized


def test_contract_grain_is_order_independent(synthetic):
    a = contract_grain(synthetic).set_index("contract_id")["supplier_name"].sort_index()
    b = contract_grain(synthetic.sample(frac=1, random_state=3)) \
        .set_index("contract_id")["supplier_name"].sort_index()
    pd.testing.assert_series_equal(a, b)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def test_country_codes_are_canonicalised_in_both_paths(synthetic_stats):
    """Domesticity is an equality test, so casing must not decide it.

    The published extract is clean on this field, but validate_and_enrich takes
    raw upstream records where " ke" and "KE" are the same country. Without
    canonicalisation a domestic supplier would be reported as foreign.
    """
    assert normalize_country_code(" ke ") == "KE"
    assert normalize_country_code(None) is None
    messy = _row(borrower_country_code=" in ", supplier_country_code="In")
    res = validate_and_enrich(messy, synthetic_stats)
    assert res.features["supplier_is_domestic"] is True


def test_suffix_stripping_never_empties_a_real_name():
    """Regression: 4 real suppliers are made entirely of legal-suffix tokens.

    Greedy stripping erased "CIE SARL" / "CIA SARL" to nothing and they were
    then failed as SUPPLIER_MISSING -- a cleaning step manufacturing a defect
    that did not exist in the source.
    """
    assert normalize_supplier_name("CIE SARL") == "CIE"
    assert normalize_supplier_name("CIA SARL") == "CIA"
    assert normalize_supplier_name("SA") == "SA"
    assert resolve_supplier_key("CIE SARL")[0] == "CIE"


def test_supplier_name_normalisation_resolves_legal_suffixes():
    assert normalize_supplier_name("Acme Trading Co., Ltd.") == "ACME TRADING"
    assert normalize_supplier_name("ACME TRADING LIMITED") == "ACME TRADING"
    assert normalize_supplier_name("  acme   trading ") == "ACME TRADING"


def test_primary_practice_and_multi_flag():
    assert primary_global_practice("Public Admin;Education;Health") == ("Public Admin", True)
    assert primary_global_practice("Transportation") == ("Transportation", False)
    assert primary_global_practice(None) == (None, False)


def test_fiscal_year_boundaries():
    assert fiscal_year_of(date(2020, 6, 30)) == 2020
    assert fiscal_year_of(date(2020, 7, 1)) == 2021
    assert days_into_fiscal_year(date(2020, 7, 1)) == 0
    assert days_into_fiscal_year(date(2021, 6, 30)) == 364


# ---------------------------------------------------------------------------
# Integration: the numbers quoted in the write-up
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real():
    clean = clean_frame(load_raw())
    stats = build_reference_stats(clean)
    return clean, stats, engineer_features(clean, stats)


def test_extract_shape_and_no_unmapped_methods(real):
    clean, _, fe = real
    assert len(clean) == 288_237
    assert fe["method_class"].isna().sum() == 0


def test_placeholder_supplier_share(real):
    clean, _, _ = real
    assert int(clean["supplier_key"].isna().sum()) == 63_609
    assert int((clean["supplier_name"] == "INDIVIDUAL CONSULTANT").sum()) == 63_603


def test_published_fiscal_year_always_matches_derived(real):
    """The FY window is derived by the publisher, so no record falls outside it.

    Documented consequence: the rule engine's 'signed outside the fiscal year
    window' check cannot fire on this extract. It is retained as an input guard
    for unvalidated data, not presented as an active control.
    """
    clean, _, _ = real
    mismatch = (clean["fiscal_year"] != clean["fiscal_year_published"].astype("Int64")).sum()
    assert mismatch == 0
    assert clean["days_into_fiscal_year"].between(0, 365).all()


def test_consortium_double_counting_is_material(real):
    clean, _, _ = real
    grain = contract_grain(clean)
    assert len(grain) == 276_417
    assert clean["amount_usd"].sum() > grain["amount_usd"].sum() * 1.10


def test_scalar_and_batch_feature_paths_agree(real):
    """The per-record path must reproduce the batch table exactly."""
    _, stats, fe = real
    sample = fe.sample(500, random_state=42)

    def norm(v):
        if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return round(float(v), 6)
        return v

    compared = 0
    for _, row in sample.iterrows():
        res = validate_and_enrich(row.to_dict(), stats)
        assert res.ok, "no row in this sample should fail validation"
        compared += 1
        for col in FEATURE_COLUMNS:
            assert norm(res.features[col]) == norm(row[col]), f"{col} disagrees"
    assert compared == 500
