"""Descriptive summary and the data-quality register.

The register is built by running the *same* `validate_and_enrich` that serves
single records in production over every row of the extract (~33 us/record,
~16 s for the full file). A separate vectorised flag path would have been faster, but it
would also have been a second implementation free to drift from the first --
and the register's whole purpose is to describe what the pipeline actually
does, not what a parallel implementation thinks it does.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from . import config
from .features import ReferenceStats
from .pipeline import validate_and_enrich
from .quality import DESCRIPTIONS, SEVERITY, DataQualityFlag as F, Severity


def run_quality_audit(clean_df: pd.DataFrame, stats: ReferenceStats) -> pd.DataFrame:
    """Apply the record-level validator to every row; return one row per record."""
    rows = []
    for rec in clean_df.to_dict("records"):
        res = validate_and_enrich(rec, stats)
        rows.append(
            {
                "contract_id": rec.get("contract_id"),
                "ok": res.ok,
                "n_flags": len(res.data_quality_flags),
                "flags": tuple(res.data_quality_flags),
                "fatal": tuple(res.fatal_flags),
            }
        )
    return pd.DataFrame(rows, index=clean_df.index)


def quality_register(audit: pd.DataFrame, total_rows: int | None = None) -> pd.DataFrame:
    """issue -> rows affected -> % -> severity -> meaning."""
    total = total_rows or len(audit)
    counts = Counter(f for flags in audit["flags"] for f in flags)
    records = [
        {
            "flag": flag.value,
            "severity": SEVERITY[flag].value,
            "rows": counts.get(flag.value, 0),
            "pct": round(counts.get(flag.value, 0) / total * 100, 3),
            "meaning": DESCRIPTIONS[flag],
        }
        for flag in F
    ]
    reg = pd.DataFrame(records)
    order = {Severity.FATAL.value: 0, Severity.DEGRADED.value: 1, Severity.NOTICE.value: 2}
    return reg.sort_values(
        ["severity", "rows"], key=lambda s: s.map(order) if s.name == "severity" else -s
    ).reset_index(drop=True)


def amount_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Contract-amount percentiles overall and by procurement category."""
    qs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    amt = df["amount_usd"]
    rows = {"All categories": amt.describe(percentiles=qs)}
    for cat, sub in df.groupby("procurement_category")["amount_usd"]:
        rows[cat] = sub.describe(percentiles=qs)
    out = pd.DataFrame(rows).T
    return out.drop(columns=["std"]).round(0)


def category_region_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Row counts by category x region, with row and column totals."""
    tab = pd.crosstab(df["region"], df["procurement_category"], margins=True, margins_name="Total")
    return tab


def fiscal_year_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-FY volume, value and prior-review share, with a completeness verdict.

    The completeness column is the point of this table. FY2027 holds 1,170 rows
    against a ~43,000 norm because the extract was frozen seven weeks into it,
    and its prior-review share is inflated accordingly. Reading that row as a
    signal rather than as an artefact would poison the model's test split.
    """
    g = df.groupby("fiscal_year")
    out = pd.DataFrame(
        {
            "contracts": g.size(),
            "total_usd_bn": (g["amount_usd"].sum() / 1e9).round(2),
            "median_usd": g["amount_usd"].median().round(0),
            "prior_review_pct": (
                g["review_type"].apply(lambda s: (s == "Prior").mean()) * 100
            ).round(1),
        }
    )
    typical = out.loc[out.index.isin(config.TRAIN_FISCAL_YEARS), "contracts"].median()
    out["vs_typical"] = (out["contracts"] / typical).round(2)
    out["completeness"] = np.where(
        out.index.isin(config.TRUNCATED_FISCAL_YEARS),
        "TRUNCATED - extract frozen mid-year",
        np.where(out["vs_typical"] < 0.85, "partial - reporting lag", "complete"),
    )
    return out


def region_coverage(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("region")
    return pd.DataFrame(
        {
            "contracts": g.size(),
            "share_pct": (g.size() / len(df) * 100).round(1),
            "total_usd_bn": (g["amount_usd"].sum() / 1e9).round(2),
            "median_usd": g["amount_usd"].median().round(0),
            "non_competitive_pct": (
                g["is_competitive_method"].apply(lambda s: (s == False).mean()) * 100
            ).round(1),
        }
    ).sort_values("contracts", ascending=False)


def method_taxonomy_table() -> pd.DataFrame:
    """The documented competitive/non-competitive mapping with its reasoning."""
    return pd.DataFrame(
        [
            {"method": m, "class": cls, "rationale": why}
            for m, (cls, why) in config.PROCUREMENT_METHOD_TAXONOMY.items()
        ]
    ).sort_values(["class", "method"]).reset_index(drop=True)


def consortium_impact(row_grain: pd.DataFrame, contract_grain_df: pd.DataFrame) -> dict:
    """Quantify the double-counting avoided by moving to contract grain."""
    return {
        "supplier_rows": len(row_grain),
        "distinct_contracts": len(contract_grain_df),
        "rows_in_multi_supplier_contracts": int(row_grain["is_consortium_member"].sum()),
        "multi_supplier_contracts": int(
            (row_grain.groupby("contract_id").size() > 1).sum()
        ),
        "largest_consortium": int(row_grain["consortium_size"].max()),
        "row_grain_total_usd_bn": round(row_grain["amount_usd"].sum() / 1e9, 1),
        "contract_grain_total_usd_bn": round(contract_grain_df["amount_usd"].sum() / 1e9, 1),
    }


def benchmark_drift(clean_df: pd.DataFrame) -> pd.DataFrame:
    """How far the frozen FY2020-22 medians drift from later years.

    Freezing the benchmark on the training window buys reproducibility and
    removes look-ahead bias, but it costs accuracy as the portfolio moves. This
    table quantifies that cost rather than leaving it as an assertion, so the
    decision to refresh the artefact can be made on evidence.
    """
    usable = clean_df[clean_df["amount_usd"] > 0]
    base = (
        usable[usable["fiscal_year"].isin(config.TRAIN_FISCAL_YEARS)]
        .groupby("procurement_category")["amount_usd"].median()
    )
    rows = []
    for fy, sub in usable.groupby("fiscal_year"):
        med = sub.groupby("procurement_category")["amount_usd"].median()
        drift = ((med / base - 1) * 100).round(1)
        rows.append(drift.rename(fy))
    return pd.DataFrame(rows).rename_axis("fiscal_year")
