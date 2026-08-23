"""The pipeline entry point: one raw record in, one validated enriched record out.

Contract with the downstream stages:

  * A required field that is missing or unusable FAILS the record (`ok=False`,
    `features=None`). The rule engine turns that into NOT_ELIGIBLE. We never substitute
    a plausible value for a required field -- a silently imputed amount would
    be indistinguishable from a real one in the audit record six months later.
  * An optional field that is missing DEGRADES exactly one feature to None and
    raises a flag. The record survives. Downstream must treat None as
    "unknown", never as zero -- the difference between "this supplier has no
    track record" and "we cannot tell whether this supplier has a track
    record" is the distinction this pipeline is built to preserve.
  * The function is pure: it reads no clock, no globals and no files. Its only
    inputs are the record and the frozen ReferenceStats. This is what makes the
    audit record reproducible for a fixed pipeline version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import config
from .cleaning import (
    _is_missing,
    classify_method,
    days_into_fiscal_year,
    fiscal_year_of,
    is_regional_borrower,
    normalize_country_code,
    normalize_text,
    parse_signing_date,
    primary_global_practice,
    resolve_supplier_key,
)
from .features import ReferenceStats, _safe_ratio
from .quality import DataQualityFlag as F
from .quality import Severity, SEVERITY, is_fatal

# Accept either the published column names or our internal snake_case names.
_ALIASES: dict[str, tuple[str, ...]] = {
    "region": ("Region", "region"),
    "borrower_country": ("Borrower Country / Economy", "borrower_country"),
    "borrower_country_code": ("Borrower Country / Economy Code", "borrower_country_code"),
    "project_id": ("Project ID", "project_id"),
    "global_practice_raw": ("Project Global Practice", "global_practice_raw"),
    "procurement_category": ("Procurement Category", "procurement_category"),
    "procurement_method_raw": ("Procurement Method", "procurement_method_raw"),
    "contract_id": ("WB Contract Number", "contract_id"),
    "signing_date_raw": ("Contract Signing Date", "signing_date_raw"),
    "supplier_name_raw": ("Supplier", "supplier_name_raw"),
    "supplier_country_code": ("Supplier Country / Economy Code", "supplier_country_code"),
    "supplier_country": ("Supplier Country / Economy", "supplier_country"),
    "amount_usd": ("Supplier Contract Amount (USD)", "amount_usd"),
    "review_type": ("Review type", "review_type"),
    "fiscal_year_published": ("Fiscal Year", "fiscal_year_published"),
    "consortium_size": ("consortium_size",),
}


def _get(record: Mapping[str, Any], name: str):
    for alias in _ALIASES[name]:
        if alias in record:
            value = record[alias]
            return None if _is_missing(value) else value
    return None


@dataclass
class EnrichedRecord:
    """Result of validating and enriching one raw contract record."""

    ok: bool
    features: dict[str, Any] | None
    data_quality_flags: list[str]
    normalized: dict[str, Any] = field(default_factory=dict)

    @property
    def fatal_flags(self) -> list[str]:
        return [f for f in self.data_quality_flags if SEVERITY[F(f)] is Severity.FATAL]

    @property
    def degraded_flags(self) -> list[str]:
        return [f for f in self.data_quality_flags if SEVERITY[F(f)] is Severity.DEGRADED]


def _coerce_amount(value) -> tuple[float | None, list[F]]:
    if _is_missing(value):
        return None, [F.AMOUNT_MISSING]
    try:
        amount = float(str(value).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None, [F.AMOUNT_UNPARSEABLE]
    if np.isnan(amount):
        return None, [F.AMOUNT_MISSING]
    if amount <= 0:
        # Kept as a value AND flagged: the rule engine reports the actual amount
        # in the exception reason, not just that something was wrong with it.
        return amount, [F.AMOUNT_NON_POSITIVE]
    return amount, []


def validate_and_enrich(
    record: Mapping[str, Any],
    stats: ReferenceStats,
) -> EnrichedRecord:
    """Validate one raw record and enrich it with model-ready features."""
    flags: list[F] = []
    norm: dict[str, Any] = {}

    # ---- required fields -------------------------------------------------
    supplier_key, supplier_flags = resolve_supplier_key(_get(record, "supplier_name_raw"))
    flags.extend(supplier_flags)
    norm["supplier_key"] = supplier_key
    norm["supplier_name_raw"] = normalize_text(_get(record, "supplier_name_raw"))

    borrower_country = normalize_text(_get(record, "borrower_country"))
    if borrower_country is None:
        flags.append(F.BORROWER_COUNTRY_MISSING)
    norm["borrower_country"] = borrower_country

    category = normalize_text(_get(record, "procurement_category"))
    if category is None:
        flags.append(F.PROCUREMENT_CATEGORY_MISSING)
    norm["procurement_category"] = category

    amount, amount_flags = _coerce_amount(_get(record, "amount_usd"))
    flags.extend(amount_flags)
    norm["amount_usd"] = amount

    signing_date, date_flags = parse_signing_date(_get(record, "signing_date_raw"))
    flags.extend(date_flags)
    norm["signing_date"] = signing_date.isoformat() if signing_date else None

    method_class, method_flags = classify_method(_get(record, "procurement_method_raw"))
    flags.extend(method_flags)
    norm["procurement_method"] = normalize_text(_get(record, "procurement_method_raw"))
    norm["method_class"] = method_class

    region = normalize_text(_get(record, "region"))
    norm["region"] = region

    # ---- fail fast on required fields -----------------------------------
    # Enriching a record whose amount or date is unusable would produce
    # features that look real. Better to stop and say so.
    if is_fatal([f.value for f in flags]):
        return EnrichedRecord(
            ok=False,
            features=None,
            data_quality_flags=sorted({f.value for f in flags}),
            normalized=norm,
        )

    # ---- fiscal calendar -------------------------------------------------
    fiscal_year = fiscal_year_of(signing_date)
    days_into_fy = days_into_fiscal_year(signing_date)
    published_fy = _get(record, "fiscal_year_published")
    if published_fy is not None and int(published_fy) != fiscal_year:
        # NOTICE, not fatal: our derivation is the authority, but a mismatch is
        # worth recording because it means the source is internally inconsistent.
        flags.append(F.FISCAL_YEAR_DISAGREEMENT)
    if not 0 <= days_into_fy <= 366:
        flags.append(F.SIGNING_DATE_OUTSIDE_FY_WINDOW)
    if signing_date > date.fromisoformat(config.DATA_AS_OF_DATE):
        flags.append(F.SIGNING_DATE_AFTER_EXTRACT)
    norm["fiscal_year"] = fiscal_year

    # ---- supplier domesticity: tri-state --------------------------------
    borrower_code = normalize_country_code(_get(record, "borrower_country_code"))
    supplier_code = normalize_country_code(_get(record, "supplier_country_code"))
    regional = is_regional_borrower(borrower_country, borrower_code)
    if regional:
        flags.append(F.BORROWER_IS_REGIONAL)
    if supplier_code is None:
        flags.append(F.SUPPLIER_COUNTRY_MISSING)
    if regional or borrower_code is None or supplier_code is None:
        supplier_is_domestic = None  # undefined, NOT False
    else:
        supplier_is_domestic = supplier_code == borrower_code

    # ---- benchmarks ------------------------------------------------------
    median, support_n, bench_flags = stats.benchmark_median(category, region)
    flags.extend(bench_flags)
    practice, multi_practice = primary_global_practice(_get(record, "global_practice_raw"))
    if multi_practice:
        flags.append(F.MULTI_PRACTICE_PROJECT)
    practice_median, _, practice_flags = stats.practice_benchmark(practice)
    flags.extend(practice_flags)

    # ---- point-in-time history ------------------------------------------
    project_id = normalize_text(_get(record, "project_id"))
    prior_supplier = stats.supplier_history.count_before(supplier_key, signing_date)
    project_seq = stats.project_history.count_before(project_id, signing_date)
    if project_id is None:
        flags.append(F.PROJECT_HISTORY_UNAVAILABLE)

    review_type = normalize_text(_get(record, "review_type"))
    if review_type is None:
        flags.append(F.REVIEW_TYPE_MISSING)

    consortium_size = _get(record, "consortium_size")
    consortium_size = int(consortium_size) if consortium_size is not None else 1
    if consortium_size > 1:
        flags.append(F.CONSORTIUM_MEMBER_ROW)

    features = {
        "amount_usd": amount,
        "log_amount": float(np.log10(amount)),
        "amount_vs_category_region_median": _safe_ratio(amount, median),
        "amount_vs_practice_median": _safe_ratio(amount, practice_median),
        "amount_percentile_in_category_region": stats.amount_percentile(
            amount, category, region
        ),
        "benchmark_support_n": support_n,
        "supplier_is_domestic": supplier_is_domestic,
        "is_first_contract_in_project": (
            None if project_seq is None else project_seq == 0
        ),
        "project_contract_sequence": project_seq,
        "supplier_prior_contract_count": prior_supplier,
        "supplier_is_known": supplier_key is not None,
        "days_into_fiscal_year": days_into_fy,
        "fy_quarter": min(days_into_fy // 91 + 1, 4),
        "is_competitive_method": (
            None if method_class is None else method_class == config.COMPETITIVE
        ),
        "consortium_size": consortium_size,
    }

    norm.update(
        {
            "global_practice": practice,
            "project_id": project_id,
            "review_type": review_type,
            "benchmark_median": median,
            "practice_median": practice_median,
            "reference_version": stats.version,
            "reference_median_fiscal_years": list(stats.median_fiscal_years),
        }
    )

    return EnrichedRecord(
        ok=True,
        features=features,
        data_quality_flags=sorted({f.value for f in flags}),
        normalized=norm,
    )
