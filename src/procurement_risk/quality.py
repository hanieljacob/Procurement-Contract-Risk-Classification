"""Data-quality flag vocabulary.

One vocabulary shared by every stage of the pipeline. Feature preparation raises
these flags, the rule engine turns the FATAL ones into NOT_ELIGIBLE, and the
audit record carries them as reason codes. Keeping them in one enum means a code
read out of an audit record six months from now still resolves to a documented
meaning.

Severity contract:
  FATAL    -> the record cannot be responsibly assessed. Fails the record.
  DEGRADED -> one feature is unavailable. The record survives; that feature is
              None and downstream must treat None as "unknown", never as zero.
  NOTICE   -> worth recording for audit, but nothing is wrong with the record.
"""

from __future__ import annotations

from enum import Enum


class Severity(str, Enum):
    FATAL = "FATAL"
    DEGRADED = "DEGRADED"
    NOTICE = "NOTICE"


class DataQualityFlag(str, Enum):
    """Each member's value is the audit-record reason code."""

    # ---- FATAL: required field missing or unusable -----------------------
    SUPPLIER_MISSING = "SUPPLIER_MISSING"
    BORROWER_COUNTRY_MISSING = "BORROWER_COUNTRY_MISSING"
    PROCUREMENT_CATEGORY_MISSING = "PROCUREMENT_CATEGORY_MISSING"
    AMOUNT_MISSING = "AMOUNT_MISSING"
    AMOUNT_NON_POSITIVE = "AMOUNT_NON_POSITIVE"
    AMOUNT_UNPARSEABLE = "AMOUNT_UNPARSEABLE"
    SIGNING_DATE_MISSING = "SIGNING_DATE_MISSING"
    SIGNING_DATE_UNPARSEABLE = "SIGNING_DATE_UNPARSEABLE"
    PROCUREMENT_METHOD_MISSING = "PROCUREMENT_METHOD_MISSING"
    PROCUREMENT_METHOD_UNMAPPED = "PROCUREMENT_METHOD_UNMAPPED"

    # ---- DEGRADED: a feature cannot be computed -------------------------
    SUPPLIER_UNIDENTIFIABLE = "SUPPLIER_UNIDENTIFIABLE"
    SUPPLIER_COUNTRY_MISSING = "SUPPLIER_COUNTRY_MISSING"
    BORROWER_IS_REGIONAL = "BORROWER_IS_REGIONAL"
    GLOBAL_PRACTICE_MISSING = "GLOBAL_PRACTICE_MISSING"
    THIN_REFERENCE_GROUP = "THIN_REFERENCE_GROUP"
    REFERENCE_MEDIAN_UNAVAILABLE = "REFERENCE_MEDIAN_UNAVAILABLE"
    PROJECT_HISTORY_UNAVAILABLE = "PROJECT_HISTORY_UNAVAILABLE"

    # ---- NOTICE: recorded, not a defect ----------------------------------
    FISCAL_YEAR_DISAGREEMENT = "FISCAL_YEAR_DISAGREEMENT"
    SIGNING_DATE_OUTSIDE_FY_WINDOW = "SIGNING_DATE_OUTSIDE_FY_WINDOW"
    SIGNING_DATE_AFTER_EXTRACT = "SIGNING_DATE_AFTER_EXTRACT"
    CONSORTIUM_MEMBER_ROW = "CONSORTIUM_MEMBER_ROW"
    MULTI_PRACTICE_PROJECT = "MULTI_PRACTICE_PROJECT"
    REVIEW_TYPE_MISSING = "REVIEW_TYPE_MISSING"


SEVERITY: dict[DataQualityFlag, Severity] = {
    DataQualityFlag.SUPPLIER_MISSING: Severity.FATAL,
    DataQualityFlag.BORROWER_COUNTRY_MISSING: Severity.FATAL,
    DataQualityFlag.PROCUREMENT_CATEGORY_MISSING: Severity.FATAL,
    DataQualityFlag.AMOUNT_MISSING: Severity.FATAL,
    DataQualityFlag.AMOUNT_NON_POSITIVE: Severity.FATAL,
    DataQualityFlag.AMOUNT_UNPARSEABLE: Severity.FATAL,
    DataQualityFlag.SIGNING_DATE_MISSING: Severity.FATAL,
    DataQualityFlag.SIGNING_DATE_UNPARSEABLE: Severity.FATAL,
    DataQualityFlag.PROCUREMENT_METHOD_MISSING: Severity.FATAL,
    DataQualityFlag.PROCUREMENT_METHOD_UNMAPPED: Severity.FATAL,
    DataQualityFlag.SUPPLIER_UNIDENTIFIABLE: Severity.DEGRADED,
    DataQualityFlag.SUPPLIER_COUNTRY_MISSING: Severity.DEGRADED,
    DataQualityFlag.BORROWER_IS_REGIONAL: Severity.DEGRADED,
    DataQualityFlag.GLOBAL_PRACTICE_MISSING: Severity.DEGRADED,
    DataQualityFlag.THIN_REFERENCE_GROUP: Severity.DEGRADED,
    DataQualityFlag.REFERENCE_MEDIAN_UNAVAILABLE: Severity.DEGRADED,
    DataQualityFlag.PROJECT_HISTORY_UNAVAILABLE: Severity.DEGRADED,
    DataQualityFlag.FISCAL_YEAR_DISAGREEMENT: Severity.NOTICE,
    DataQualityFlag.SIGNING_DATE_OUTSIDE_FY_WINDOW: Severity.NOTICE,
    DataQualityFlag.SIGNING_DATE_AFTER_EXTRACT: Severity.NOTICE,
    DataQualityFlag.CONSORTIUM_MEMBER_ROW: Severity.NOTICE,
    DataQualityFlag.MULTI_PRACTICE_PROJECT: Severity.NOTICE,
    DataQualityFlag.REVIEW_TYPE_MISSING: Severity.NOTICE,
}

# Human-readable meaning, surfaced in the audit record and the notebook register.
DESCRIPTIONS: dict[DataQualityFlag, str] = {
    DataQualityFlag.SUPPLIER_MISSING: "Supplier name is absent; the counterparty is unknown.",
    DataQualityFlag.BORROWER_COUNTRY_MISSING: "Borrower country is absent.",
    DataQualityFlag.PROCUREMENT_CATEGORY_MISSING: "Procurement category is absent; no peer group can be formed.",
    DataQualityFlag.AMOUNT_MISSING: "Contract amount is absent.",
    DataQualityFlag.AMOUNT_NON_POSITIVE: "Contract amount is zero or negative; not an assessable award value.",
    DataQualityFlag.AMOUNT_UNPARSEABLE: "Contract amount could not be read as a number.",
    DataQualityFlag.SIGNING_DATE_MISSING: "Signing date is absent; the record cannot be placed in time.",
    DataQualityFlag.SIGNING_DATE_UNPARSEABLE: "Signing date did not match the expected MM/DD/YYYY format.",
    DataQualityFlag.PROCUREMENT_METHOD_MISSING: "Procurement method is absent.",
    DataQualityFlag.PROCUREMENT_METHOD_UNMAPPED: "Procurement method is not in the documented taxonomy; competitiveness is unknown.",
    DataQualityFlag.SUPPLIER_UNIDENTIFIABLE: "Supplier field is a placeholder, not a named entity; supplier history cannot be computed.",
    DataQualityFlag.SUPPLIER_COUNTRY_MISSING: "Supplier country is absent; domesticity cannot be determined.",
    DataQualityFlag.BORROWER_IS_REGIONAL: "Borrower is a multi-country or regional programme; supplier domesticity is undefined.",
    DataQualityFlag.GLOBAL_PRACTICE_MISSING: "Project global practice is absent; the practice benchmark is unavailable.",
    DataQualityFlag.THIN_REFERENCE_GROUP: "Peer group had too few observations; a broader fallback benchmark was used.",
    DataQualityFlag.REFERENCE_MEDIAN_UNAVAILABLE: "No usable benchmark median exists for this record.",
    DataQualityFlag.PROJECT_HISTORY_UNAVAILABLE: "Project was not seen during reference fitting; project-position features are unknown.",
    DataQualityFlag.FISCAL_YEAR_DISAGREEMENT: "Published fiscal year disagrees with the year derived from the signing date.",
    DataQualityFlag.SIGNING_DATE_OUTSIDE_FY_WINDOW: "Signing date falls outside its published fiscal year window.",
    DataQualityFlag.SIGNING_DATE_AFTER_EXTRACT: "Signing date is later than the extract's as-of date.",
    DataQualityFlag.CONSORTIUM_MEMBER_ROW: "One of several supplier rows sharing a contract number (joint venture).",
    DataQualityFlag.MULTI_PRACTICE_PROJECT: "Project spans several global practices; the first listed was used as primary.",
    DataQualityFlag.REVIEW_TYPE_MISSING: "Prior/post review type is absent.",
}

FATAL_FLAGS: frozenset[DataQualityFlag] = frozenset(
    f for f, s in SEVERITY.items() if s is Severity.FATAL
)


def is_fatal(flags) -> bool:
    """True when any flag in `flags` prevents assessment."""
    return any(DataQualityFlag(f) in FATAL_FLAGS for f in flags)


def describe(flag) -> str:
    return DESCRIPTIONS[DataQualityFlag(flag)]
