"""Reference statistics (fit) and feature engineering (transform).

Every population statistic in this module is **date-filtered**: it reads only
what was signed before the record being scored. That is the single rule, and it
applies uniformly to benchmarks and to history counts alike.

It was not always so. Benchmark medians were originally fitted over a fixed
window (FY2020-22) and frozen, on the reasoning that a real review system
publishes a benchmark table on a schedule rather than recomputing it per
request. That is true of deployment, but it is wrong for scoring history: a
contract signed in July 2019 was being divided by a median containing contracts
signed up to three years *after* it. Measured, the frozen median ran +3.3%
against the true as-of value in FY2020 and -10.6% by FY2026 -- look-ahead at one
end of the timeline and staleness at the other, moving up to 2.6% of records
across a threshold.

The fix is a **vintage table**: for each peer group and each month, the median of
everything signed strictly before that month. A benchmark may now draw on every
year in the extract precisely because it can only ever read the past -- the same
argument that always justified the history counts. So the two classes of
statistic collapse into one, and the train/test split now matters only to the
model, never to feature construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import config
from .cleaning import _is_missing, contract_grain
from .quality import DataQualityFlag as F

_ORDINAL_SPACE = 10_000_000  # > any date.toordinal(); keeps composites unique

# Quantile sketch resolution. 101 points gives whole-percentile precision, which
# is finer than any threshold we set, at 2,416 cells x 101 floats -- trivial.
_SKETCH = np.linspace(0.0, 1.0, 101)


def month_key(d: date) -> int:
    """Months since year 0. A cheap, hashable, orderable vintage key."""
    return d.year * 12 + (d.month - 1)


def month_key_series(dates: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dates)
    return (dt.dt.year * 12 + (dt.dt.month - 1)).astype("Int64")


class PointInTimeCounter:
    """Counts prior events for a key, strictly before a query date.

    Implemented as one flat sorted int64 array of composite
    ``group_id * 10_000_000 + date_ordinal`` values. Because group ids are
    contiguous and ordered, a single global ``np.searchsorted`` respects group
    boundaries, which makes the batch path a vectorised one-liner while the
    scalar path is the same searchsorted on the same array. Batch and single
    record therefore cannot drift apart -- they are literally the same lookup,
    which is what the "same input, same output" guarantee needs.
    """

    def __init__(self, keys: pd.Series, dates: pd.Series):
        mask = keys.notna() & dates.notna()
        k = keys[mask].astype(str).to_numpy()
        d = pd.to_datetime(dates[mask]).map(lambda x: x.toordinal()).to_numpy(dtype=np.int64)
        uniq, group_ids = np.unique(k, return_inverse=True)
        composite = group_ids.astype(np.int64) * _ORDINAL_SPACE + d
        composite.sort()
        self._index = {key: i for i, key in enumerate(uniq)}
        self._composite = composite
        order = np.argsort(group_ids, kind="stable")
        self._earliest = (
            pd.DataFrame({"g": group_ids[order], "d": d[order]})
            .groupby("g")["d"].min().to_dict()
        )

    def __len__(self) -> int:
        return len(self._index)

    def count_before(self, key, query_date: date | None) -> int | None:
        """Events for `key` strictly before `query_date`.

        None means "unknowable" -- an unresolved key or an unusable date -- and
        must never be read as zero by a caller.
        """
        if _is_missing(key) or query_date is None:
            return None
        gid = self._index.get(str(key))
        if gid is None:
            return 0  # key resolved, simply never seen before: a genuine zero
        lo = gid * _ORDINAL_SPACE
        left = np.searchsorted(self._composite, lo, side="left")
        pos = np.searchsorted(self._composite, lo + query_date.toordinal(), side="left")
        return int(pos - left)

    def count_before_batch(self, keys: pd.Series, dates: pd.Series) -> pd.Series:
        """Vectorised equivalent of `count_before` over a whole frame."""
        out = pd.Series(pd.NA, index=keys.index, dtype="Int64")
        mask = keys.notna() & dates.notna()
        if not mask.any():
            return out
        k = keys[mask].astype(str).to_numpy()
        gid = np.array([self._index.get(x, -1) for x in k], dtype=np.int64)
        d = pd.to_datetime(dates[mask]).map(lambda x: x.toordinal()).to_numpy(dtype=np.int64)
        lo = gid * _ORDINAL_SPACE
        left = np.searchsorted(self._composite, lo, side="left")
        pos = np.searchsorted(self._composite, lo + d, side="left")
        out.loc[mask] = np.where(gid < 0, 0, pos - left).astype(np.int64)
        return out

    def earliest(self, key) -> int | None:
        if _is_missing(key):
            return None
        gid = self._index.get(str(key))
        return self._earliest.get(gid) if gid is not None else None


@dataclass(frozen=True)
class Vintage:
    """A benchmark as it stood at the start of one month."""

    median: float
    support_n: int
    quantiles: np.ndarray  # 101 points; percentile lookup by searchsorted

    def percentile_of(self, amount: float) -> float:
        return float(np.searchsorted(self.quantiles, float(amount), side="right") / len(self.quantiles))


def _build_vintages(frame: pd.DataFrame, keys: list[str]) -> dict[tuple, Vintage]:
    """Median + quantile sketch of everything signed strictly before each month.

    "Strictly before the month" rather than "strictly before the day" is
    deliberate: a benchmark table is published periodically, and a monthly
    vintage is both how a real control function would consume one and a
    conservative reading of point-in-time -- a contract signed on the 15th is
    never compared against anything signed in its own month.
    """
    out: dict[tuple, Vintage] = {}
    group_cols = keys + ["_month"]
    for key, sub in frame.groupby(keys, observed=True, dropna=True):
        key = key if isinstance(key, tuple) else (key,)
        sub = sub.sort_values("_month")
        amounts = sub["amount_usd"].to_numpy()
        months = sub["_month"].to_numpy()
        # One pass: the prefix of `amounts` before each distinct month.
        distinct = np.unique(months)
        for m in distinct:
            prior = amounts[months < m]
            if len(prior) < config.MIN_GROUP_SUPPORT:
                continue
            prior = np.sort(prior)
            out[key + (int(m),)] = Vintage(
                median=float(np.median(prior)),
                support_n=int(len(prior)),
                quantiles=np.quantile(prior, _SKETCH),
            )
    return out


@dataclass
class ReferenceStats:
    """Versioned reference artefact. Fitted once, then read-only.

    Every member is date-keyed, so applying it to a record can only ever consult
    information that predates that record.
    """

    version: str
    vintage_months: tuple[int, int]
    category_region: dict[tuple, Vintage] = field(repr=False)
    category: dict[tuple, Vintage] = field(repr=False)
    overall: dict[tuple, Vintage] = field(repr=False)
    practice: dict[tuple, Vintage] = field(repr=False)
    supplier_history: PointInTimeCounter = field(repr=False)
    project_history: PointInTimeCounter = field(repr=False)

    # -- benchmark lookup with a documented fallback ladder -----------------
    def _vintage(self, category, region, as_of: date | None) -> tuple[Vintage | None, list[F]]:
        if as_of is None or _is_missing(category):
            return None, [F.REFERENCE_MEDIAN_UNAVAILABLE]
        m = month_key(as_of)
        if not _is_missing(region):
            cell = self.category_region.get((category, region, m))
            if cell is not None:
                return cell, []
        # Widen rather than pretend: a peer group with too little history yet is
        # not a benchmark, so we climb to a coarser one and say that we did.
        cell = self.category.get((category, m))
        if cell is not None:
            return cell, [F.THIN_REFERENCE_GROUP]
        cell = self.overall.get((m,))
        if cell is not None:
            return cell, [F.THIN_REFERENCE_GROUP]
        # Warm-up: too early in the extract for any benchmark to exist yet.
        return None, [F.REFERENCE_MEDIAN_UNAVAILABLE]

    def benchmark_median(self, category, region, as_of: date | None) -> tuple[float | None, int, list[F]]:
        cell, flags = self._vintage(category, region, as_of)
        return (None, 0, flags) if cell is None else (cell.median, cell.support_n, flags)

    def amount_percentile(self, amount, category, region, as_of: date | None) -> float | None:
        """Where this amount sat within its peer group as of that month, 0-1."""
        if _is_missing(amount):
            return None
        cell, _ = self._vintage(category, region, as_of)
        return None if cell is None else cell.percentile_of(amount)

    def practice_benchmark(self, practice, as_of: date | None) -> tuple[float | None, int, list[F]]:
        # _is_missing, not `is None`: a missing practice arrives as None from a
        # dict record but as NaN from a DataFrame column. Checking only for None
        # let NaN fall through to the fallback, manufacturing a benchmark for a
        # field we do not actually have.
        if _is_missing(practice):
            return None, 0, [F.GLOBAL_PRACTICE_MISSING]
        if as_of is None:
            return None, 0, [F.REFERENCE_MEDIAN_UNAVAILABLE]
        m = month_key(as_of)
        cell = self.practice.get((practice, m))
        if cell is not None:
            return cell.median, cell.support_n, []
        cell = self.overall.get((m,))
        if cell is not None:
            return cell.median, cell.support_n, [F.THIN_REFERENCE_GROUP]
        return None, 0, [F.REFERENCE_MEDIAN_UNAVAILABLE]


def _safe_ratio(amount, median) -> float | None:
    """Amount / median, guarded against a zero or absurdly small denominator."""
    if _is_missing(amount) or _is_missing(median):
        return None
    if median < config.MIN_MEDIAN_DENOMINATOR:
        return None
    return float(amount) / float(median)


def build_reference_stats(
    clean_df: pd.DataFrame,
    version: str = config.PIPELINE_VERSION,
) -> ReferenceStats:
    """Fit the reference artefact.

    All statistics are computed at CONTRACT grain, not row grain. 8,584
    contracts are split across several supplier rows, most repeating the full
    contract amount on each -- counting rows would let a three-way joint
    venture push its amount into the median three times.

    There is no fitting window. Each vintage cell reads only what preceded it,
    so spanning every year adds history without adding look-ahead.
    """
    grain = contract_grain(clean_df)
    usable = grain[
        grain["amount_usd"].notna()
        & (grain["amount_usd"] > 0)
        & grain["signing_date"].notna()
    ].copy()
    if usable.empty:
        raise ValueError("No usable contracts to fit reference statistics from")

    usable["_month"] = month_key_series(usable["signing_date"]).astype(int)
    usable["_all"] = "ALL"

    months = (int(usable["_month"].min()), int(usable["_month"].max()))

    return ReferenceStats(
        version=version,
        vintage_months=months,
        category_region=_build_vintages(usable, ["procurement_category", "region"]),
        category=_build_vintages(usable, ["procurement_category"]),
        overall=_build_vintages(usable, ["_all"]),
        practice=_build_vintages(usable, ["global_practice"]),
        supplier_history=PointInTimeCounter(usable["supplier_key"], usable["signing_date"]),
        project_history=PointInTimeCounter(usable["project_id"], usable["signing_date"]),
    )


# ---------------------------------------------------------------------------
# Batch transform
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: tuple[str, ...] = (
    "amount_usd",
    "log_amount",
    "amount_vs_category_region_median",
    "amount_vs_practice_median",
    "amount_percentile_in_category_region",
    "benchmark_support_n",
    "supplier_is_domestic",
    "is_first_contract_in_project",
    "project_contract_sequence",
    "supplier_prior_contract_count",
    "supplier_is_known",
    "days_into_fiscal_year",
    "fy_quarter",
    "is_competitive_method",
    "consortium_size",
)


def engineer_features(clean_df: pd.DataFrame, stats: ReferenceStats) -> pd.DataFrame:
    """Add the engineered feature columns to a cleaned frame.

    Pure with respect to `stats`: the same cleaned frame and the same artefact
    give an identical result every time. Benchmark lookups go through the same
    `ReferenceStats` methods the single-record path calls, so the two cannot
    disagree.
    """
    out = clean_df.copy()
    as_of = [d.date() if pd.notna(d) else None for d in out["signing_date"]]

    bench = [
        stats.benchmark_median(c, r, d)
        for c, r, d in zip(out["procurement_category"], out["region"], as_of)
    ]
    out["benchmark_median"] = [b[0] for b in bench]
    out["benchmark_support_n"] = [b[1] for b in bench]
    out["benchmark_is_thin"] = [F.THIN_REFERENCE_GROUP in b[2] for b in bench]
    out["amount_vs_category_region_median"] = [
        _safe_ratio(a, m) for a, m in zip(out["amount_usd"], out["benchmark_median"])
    ]

    prac = [stats.practice_benchmark(p, d) for p, d in zip(out["global_practice"], as_of)]
    out["practice_median"] = [p[0] for p in prac]
    out["amount_vs_practice_median"] = [
        _safe_ratio(a, m) for a, m in zip(out["amount_usd"], out["practice_median"])
    ]

    out["amount_percentile_in_category_region"] = [
        stats.amount_percentile(a, c, r, d)
        for a, c, r, d in zip(
            out["amount_usd"], out["procurement_category"], out["region"], as_of
        )
    ]

    out["log_amount"] = np.log10(out["amount_usd"].where(out["amount_usd"] > 0))

    out["supplier_prior_contract_count"] = stats.supplier_history.count_before_batch(
        out["supplier_key"], out["signing_date"]
    )
    out["supplier_is_known"] = out["supplier_key"].notna()

    out["project_contract_sequence"] = stats.project_history.count_before_batch(
        out["project_id"], out["signing_date"]
    )
    # "First" means strictly: nothing in this project was signed BEFORE this
    # contract. Contracts sharing the project's earliest signing date all
    # qualify. That is deliberate -- the extract records a date, not a time, so
    # within a single day there is no defensible ordering, and inventing one via
    # row order would make the feature depend on file layout. Reviewer-facing
    # meaning: "no prior contract was observable in this project when this one
    # was signed."
    out["is_first_contract_in_project"] = (
        out["project_contract_sequence"] == 0
    ).astype("boolean")
    out.loc[out["project_contract_sequence"].isna(), "is_first_contract_in_project"] = pd.NA

    return out
