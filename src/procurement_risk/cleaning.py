"""Field-level cleaning and entity resolution.

Design rule throughout: cleaning never invents a value. Where a field is
missing or untrustworthy we return None and raise a DataQualityFlag, so the
uncertainty stays visible all the way into the audit record. The one
thing we do change is *representation* -- casing, whitespace, legal suffixes --
because that is normalisation, not imputation.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime

import numpy as np
import pandas as pd

from . import config
from .quality import DataQualityFlag as F

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s&]")
_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in config.LEGAL_SUFFIXES) + r")\b\.?\s*$"
)


# ---------------------------------------------------------------------------
# Scalar normalisers
# ---------------------------------------------------------------------------

def _is_missing(value) -> bool:
    """True for None, NaN, pd.NA and empty strings.

    Needed because a value's "missingness" arrives in several disguises: None
    from a dict record, float('nan') from a pandas .map, and pd.NA from an
    Arrow-backed string column. Every cleaner routes through here so that all
    three take the same path instead of one of them crashing.
    """
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()


def normalize_text(value) -> str | None:
    """Trim, collapse internal whitespace, strip control chars. Missing-safe."""
    if _is_missing(value):
        return None
    s = str(value)
    s = unicodedata.normalize("NFKC", s)
    s = _WS.sub(" ", s).strip()
    return s or None


def normalize_country_code(value) -> str | None:
    """Canonicalise an ISO country code: trimmed, upper-cased, or None.

    The published extract happens to be clean on this field, but
    `validate_and_enrich` is meant to accept raw records from upstream systems
    where " ke" and "KE" are the same country. Domesticity is decided by an
    equality test between two codes, so any casing or whitespace difference
    would silently report a domestic supplier as foreign.
    """
    s = normalize_text(value)
    return s.upper() if s is not None else None


def normalize_supplier_name(value) -> str | None:
    """Uppercase, de-punctuate, and strip trailing legal suffixes.

    "Acme Trading Co., Ltd." and "ACME TRADING CO LTD" both resolve to
    "ACME TRADING". This is deliberately conservative: we only strip suffixes
    from the end, never from inside a name, to avoid mangling firms whose name
    genuinely contains such a token.
    """
    s = normalize_text(value)
    if s is None:
        return None
    s = s.upper()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    # Strip trailing legal suffixes, but NEVER strip a name down to nothing.
    # Four real suppliers in the extract ("CIE SARL", "CIA SARL") consist
    # entirely of tokens that are also legal suffixes. Stripping greedily
    # erased them and they were then wrongly failed as SUPPLIER_MISSING -- a
    # cleaning step manufacturing a data-quality defect that did not exist.
    # The last non-empty form wins.
    prev = None
    while prev != s:  # e.g. "... CO LTD" needs two passes
        prev = s
        stripped = _SUFFIX_RE.sub("", s).strip()
        if not stripped:
            break
        s = stripped
    return s or None


def is_placeholder_supplier(normalized_name: str | None) -> bool:
    """True when the supplier field is a category label, not an entity.

    "INDIVIDUAL CONSULTANT" accounts for 63,603 rows (22.1%) across 53,905
    Supplier IDs. Treating it as a vendor would give a fifth of the dataset the
    contract history of the single most prolific supplier in the portfolio.
    """
    if _is_missing(normalized_name):
        return True
    return str(normalized_name).upper().strip() in config.SUPPLIER_PLACEHOLDERS


def resolve_supplier_key(raw_name) -> tuple[str | None, list[F]]:
    """Return (stable entity key, flags).

    We key on the normalised *name*, not Supplier ID. The extract carries
    183,980 Supplier IDs for 121,914 names, and 53,905 of those IDs belong to
    the single "INDIVIDUAL CONSULTANT" placeholder -- so the ID behaves like a
    per-award reference number, not a durable vendor identifier. The name is
    the better entity key here, and we say so rather than trusting the field
    that merely looks like a primary key.
    """
    flags: list[F] = []
    normalized = normalize_supplier_name(raw_name)
    if normalized is None:
        return None, [F.SUPPLIER_MISSING]
    if is_placeholder_supplier(normalized):
        return None, [F.SUPPLIER_UNIDENTIFIABLE]
    return normalized, flags


def primary_global_practice(value) -> tuple[str | None, bool]:
    """Return (primary practice, spans_multiple).

    48.2% of records list several practices separated by ';' (240 raw
    combinations over 16 atomic practices). We take the first listed as
    primary. Simplification, documented: the field has no ordering guarantee,
    so "first" is a convention, not a claim about which practice dominates the
    contract. The alternative -- exploding to one row per practice -- would
    break the one-row-per-award grain the whole pipeline depends on.
    """
    s = normalize_text(value)
    if s is None:
        return None, False
    parts = [p.strip() for p in s.split(";") if p.strip()]
    if not parts:
        return None, False
    return parts[0], len(parts) > 1


def is_regional_borrower(country_name, country_code) -> bool:
    """True when the 'borrower country' is a multi-country or regional programme.

    Matters because "is the supplier domestic?" has no answer for a
    DRC-Angola or Western Balkans programme -- the honest value is None, not
    False. A missing ISO code is the trigger; the name markers disambiguate a
    genuine regional programme from a mere encoding gap (Cote d'Ivoire).
    """
    if normalize_text(country_code) is not None:
        return False
    name = (normalize_text(country_name) or "").upper()
    return any(marker in name for marker in config.REGIONAL_BORROWER_MARKERS)


def parse_signing_date(value) -> tuple[date | None, list[F]]:
    """Strict MM/DD/YYYY. An unreadable date fails the record, never defaults."""
    s = normalize_text(value)
    if s is None:
        return None, [F.SIGNING_DATE_MISSING]
    try:
        return datetime.strptime(s, config.SIGNING_DATE_FORMAT).date(), []
    except ValueError:
        return None, [F.SIGNING_DATE_UNPARSEABLE]


def fiscal_year_of(d: date) -> int:
    """World Bank FY N runs 1 Jul (N-1) .. 30 Jun (N)."""
    return d.year + 1 if d.month >= config.FY_START_MONTH else d.year


def fiscal_year_start(fy: int) -> date:
    return date(fy - 1, config.FY_START_MONTH, config.FY_START_DAY)


def days_into_fiscal_year(d: date) -> int:
    return (d - fiscal_year_start(fiscal_year_of(d))).days


def classify_method(value) -> tuple[str | None, list[F]]:
    """Map a procurement method to COMPETITIVE / NON_COMPETITIVE.

    Exhaustive lookup, never a keyword guess: an unrecognised method is a
    NOT_ELIGIBLE condition in the rule engine, and a heuristic that always returns an
    answer would quietly disable that control.
    """
    s = normalize_text(value)
    if s is None:
        return None, [F.PROCUREMENT_METHOD_MISSING]
    cls = config.METHOD_CLASS.get(config.method_lookup_key(s))
    if cls is None:
        return None, [F.PROCUREMENT_METHOD_UNMAPPED]
    return cls, []


# ---------------------------------------------------------------------------
# Frame-level cleaning
# ---------------------------------------------------------------------------

def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised equivalent of the scalar cleaners, for the whole extract.

    Adds cleaned columns alongside the raw ones (raw columns are kept so the
    notebook can show before/after). Row-level quality flags are attached by
    `quality_flags_frame`, not here, to keep the two concerns separable.
    """
    out = df.copy()

    for col in ("region", "borrower_country", "supplier_country", "project_id",
                "procurement_category", "review_type", "supplier_id"):
        out[col] = out[col].map(normalize_text)

    for col in ("borrower_country_code", "supplier_country_code"):
        out[col] = out[col].map(normalize_country_code)

    out["supplier_name"] = out["supplier_name_raw"].map(normalize_supplier_name)
    out["supplier_is_placeholder"] = out["supplier_name"].map(is_placeholder_supplier)
    out["supplier_key"] = out["supplier_name"].where(~out["supplier_is_placeholder"])

    practice = out["global_practice_raw"].map(primary_global_practice)
    out["global_practice"] = practice.map(lambda t: t[0])
    out["global_practice_multi"] = practice.map(lambda t: t[1])

    out["borrower_is_regional"] = [
        is_regional_borrower(n, c)
        for n, c in zip(out["borrower_country"], out["borrower_country_code"])
    ]

    out["signing_date"] = pd.to_datetime(
        out["signing_date_raw"], format=config.SIGNING_DATE_FORMAT, errors="coerce"
    )
    ok = out["signing_date"].notna()
    out["fiscal_year"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[ok, "fiscal_year"] = np.where(
        out.loc[ok, "signing_date"].dt.month >= config.FY_START_MONTH,
        out.loc[ok, "signing_date"].dt.year + 1,
        out.loc[ok, "signing_date"].dt.year,
    )
    fy = out["fiscal_year"]
    fy_start = pd.to_datetime(
        dict(year=fy.astype("float") - 1, month=config.FY_START_MONTH,
             day=config.FY_START_DAY), errors="coerce"
    )
    out["fiscal_year_start"] = fy_start
    out["days_into_fiscal_year"] = (out["signing_date"] - fy_start).dt.days
    out["fy_quarter"] = (out["days_into_fiscal_year"] // 91 + 1).clip(upper=4)

    out["method_class"] = (
        out["procurement_method_raw"]
        .map(lambda v: config.METHOD_CLASS.get(config.method_lookup_key(v))
             if not _is_missing(v) else None)
    )
    out["is_competitive_method"] = out["method_class"].map(
        {config.COMPETITIVE: True, config.NON_COMPETITIVE: False}
    ).astype("boolean")

    # Supplier domesticity: tri-state. None where undecidable.
    same = (out["supplier_country_code"].notna()
            & out["borrower_country_code"].notna()
            & (out["supplier_country_code"] == out["borrower_country_code"]))
    undecidable = (out["borrower_is_regional"]
                   | out["borrower_country_code"].isna()
                   | out["supplier_country_code"].isna())
    out["supplier_is_domestic"] = pd.Series(same, index=out.index, dtype="boolean")
    out.loc[undecidable, "supplier_is_domestic"] = pd.NA

    # Tri-state: None when the supplier country is unknown, because "we cannot
    # tell where this supplier is registered" is not the same claim as "this
    # supplier is registered somewhere transparent".
    out["supplier_in_secrecy_jurisdiction"] = pd.Series(
        out["supplier_country"].isin(config.SECRECY_JURISDICTIONS), index=out.index
    ).astype("boolean")
    out.loc[out["supplier_country"].isna(), "supplier_in_secrecy_jurisdiction"] = pd.NA

    # Consortium / joint-venture structure. 8,584 contract numbers carry more
    # than one supplier row and most repeat the FULL contract amount on each
    # row, so any statistic computed on raw rows double-counts money.
    sizes = out.groupby("contract_id")["contract_id"].transform("size")
    out["consortium_size"] = sizes.astype("int32")
    out["is_consortium_member"] = sizes > 1

    return out


def contract_grain(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per contract for population statistics.

    Deterministic: within a contract number we keep the alphabetically first
    supplier name so repeated runs pick the same representative row. Used for
    every reference median and history count -- reporting still happens at row
    grain, but the *statistics* must not count a three-way joint venture as
    three contracts worth USD 3x.
    """
    ordered = df.sort_values(
        ["contract_id", "supplier_name", "supplier_id"], na_position="last", kind="stable"
    )
    return ordered.drop_duplicates("contract_id", keep="first")
