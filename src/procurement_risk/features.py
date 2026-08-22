"""Reference statistics (fit) and feature engineering (transform).

The single most important decision in the pipeline lives here: the split between
what is *fitted* and what is *computed per record*.

Two kinds of population statistic feed the feature set, and they have different
leakage properties:

  Benchmark medians ("amount relative to the median for this category and
  region") have no date filter. A median taken over the whole extract is
  contaminated by contracts signed after the record being assessed. These are
  therefore fitted on the TRAINING FISCAL YEARS ONLY and frozen into a
  versioned artefact. That mirrors how a real review system works -- a
  benchmark table is refreshed on a schedule, not recomputed per request -- and
  it is what makes the pipeline's reproducibility guarantee achievable.

  History counts ("prior contracts from this supplier", "first contract in this
  project") are queried WITH a date filter: count only what was signed strictly
  before this record. Restricting the store to training years would not reduce
  leakage, it would simply make a FY2025 record wrongly look like a first-time
  supplier. So the history store spans every year available, and correctness
  comes from the date predicate, not from the fitting window.

Getting these two backwards -- freezing history and floating medians -- is the
easy mistake, and it silently destroys both the model and the audit trail.
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
        # First occurrence per group, for "is this the first ever?" questions.
        order = np.argsort(group_ids, kind="stable")
        self._earliest = (
            pd.DataFrame({"g": group_ids[order], "d": d[order]})
            .groupby("g")["d"].min().to_dict()
        )

    def __len__(self) -> int:
        return len(self._index)

    def count_before(self, key, query_date: date | None) -> int | None:
        """Number of stored events for `key` strictly before `query_date`.

        None means "unknowable" -- an unresolved key or an unusable date --
        and must never be read as zero by a caller.
        """
        if _is_missing(key) or query_date is None:
            return None
        gid = self._index.get(str(key))
        if gid is None:
            return 0  # key resolved, simply never seen before: a genuine zero
        lo = gid * _ORDINAL_SPACE
        target = lo + query_date.toordinal()
        left = np.searchsorted(self._composite, lo, side="left")
        pos = np.searchsorted(self._composite, target, side="left")
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
        counts = np.where(gid < 0, 0, pos - left)
        out.loc[mask] = counts.astype(np.int64)
        return out

    def earliest(self, key) -> int | None:
        if _is_missing(key):
            return None
        gid = self._index.get(str(key))
        return self._earliest.get(gid) if gid is not None else None


@dataclass
class ReferenceStats:
    """Frozen, versioned benchmark artefact. Fitted once, then read-only."""

    version: str
    median_fiscal_years: tuple[int, ...]
    category_region_median: dict[tuple[str, str], float]
    category_region_count: dict[tuple[str, str], int]
    category_median: dict[str, float]
    category_count: dict[str, int]
    practice_median: dict[str, float]
    practice_count: dict[str, int]
    global_median: float
    category_region_quantiles: dict[tuple[str, str], np.ndarray]
    supplier_history: PointInTimeCounter = field(repr=False)
    project_history: PointInTimeCounter = field(repr=False)

    # -- benchmark lookup with a documented fallback ladder -----------------
    def benchmark_median(self, category, region) -> tuple[float | None, int, list[F]]:
        """Median contract amount for a peer group, with fallback.

        Category x Region cells range from n=1 (Works x Other) to n=25,773. A
        median over one observation is not a benchmark, so below
        MIN_GROUP_SUPPORT we widen the peer group rather than pretend. The
        support count is returned alongside so a downstream stage can discount
        a thin cell instead of trusting it blindly.
        """
        flags: list[F] = []
        if _is_missing(category) or _is_missing(region):
            return None, 0, [F.REFERENCE_MEDIAN_UNAVAILABLE]
        key = (category, region)
        n = self.category_region_count.get(key, 0)
        if n >= config.MIN_GROUP_SUPPORT:
            return self.category_region_median[key], n, flags
        flags.append(F.THIN_REFERENCE_GROUP)
        n_cat = self.category_count.get(category, 0)
        if n_cat >= config.MIN_GROUP_SUPPORT:
            return self.category_median[category], n_cat, flags
        if self.global_median and self.global_median > 0:
            return self.global_median, sum(self.category_count.values()), flags
        return None, 0, flags + [F.REFERENCE_MEDIAN_UNAVAILABLE]

    def practice_benchmark(self, practice) -> tuple[float | None, int, list[F]]:
        # _is_missing, not `is None`: a missing practice arrives as None from a
        # dict record but as NaN from a DataFrame column. Checking only for None
        # let NaN fall through to the global-median fallback, which silently
        # manufactured a benchmark for a field we do not actually have.
        if _is_missing(practice):
            return None, 0, [F.GLOBAL_PRACTICE_MISSING]
        n = self.practice_count.get(practice, 0)
        if n >= config.MIN_GROUP_SUPPORT:
            return self.practice_median[practice], n, []
        if self.global_median and self.global_median > 0:
            return self.global_median, sum(self.practice_count.values()), [F.THIN_REFERENCE_GROUP]
        return None, 0, [F.REFERENCE_MEDIAN_UNAVAILABLE]

    def amount_percentile(self, amount, category, region) -> float | None:
        """Where this amount sits within its peer group, 0-1."""
        if _is_missing(category) or _is_missing(region) or _is_missing(amount):
            return None
        q = self.category_region_quantiles.get((category, region))
        if q is None:
            return None
        return float(np.searchsorted(q, float(amount), side="right") / len(q))


def _safe_ratio(amount, median) -> float | None:
    """Amount / median, guarded against a zero or absurdly small denominator."""
    if _is_missing(amount) or _is_missing(median):
        return None
    if median < config.MIN_MEDIAN_DENOMINATOR:
        return None
    return float(amount) / float(median)


def build_reference_stats(
    clean_df: pd.DataFrame,
    median_fiscal_years: tuple[int, ...] = config.TRAIN_FISCAL_YEARS,
    version: str = config.PIPELINE_VERSION,
) -> ReferenceStats:
    """Fit the benchmark artefact.

    All statistics are computed at CONTRACT grain, not row grain. 8,584
    contracts are split across several supplier rows, most repeating the full
    contract amount on each -- counting rows would let a three-way joint
    venture push its amount into the median three times.
    """
    grain = contract_grain(clean_df)
    usable = grain[grain["amount_usd"].notna() & (grain["amount_usd"] > 0)]

    median_pool = usable[usable["fiscal_year"].isin(median_fiscal_years)]
    if median_pool.empty:
        raise ValueError(f"No usable contracts in fiscal years {median_fiscal_years}")

    cr = median_pool.groupby(["procurement_category", "region"])["amount_usd"]
    cat = median_pool.groupby("procurement_category")["amount_usd"]
    prac = median_pool.groupby("global_practice")["amount_usd"]

    quantiles = {
        k: np.sort(v.to_numpy())
        for k, v in median_pool.groupby(["procurement_category", "region"])["amount_usd"]
        if len(v) >= config.MIN_GROUP_SUPPORT
    }

    # History spans ALL years by design -- see module docstring. The date
    # predicate at query time, not the fitting window, is what prevents leakage.
    supplier_history = PointInTimeCounter(usable["supplier_key"], usable["signing_date"])
    project_history = PointInTimeCounter(usable["project_id"], usable["signing_date"])

    return ReferenceStats(
        version=version,
        median_fiscal_years=tuple(median_fiscal_years),
        category_region_median=cr.median().to_dict(),
        category_region_count=cr.size().to_dict(),
        category_median=cat.median().to_dict(),
        category_count=cat.size().to_dict(),
        practice_median=prac.median().to_dict(),
        practice_count=prac.size().to_dict(),
        global_median=float(median_pool["amount_usd"].median()),
        category_region_quantiles=quantiles,
        supplier_history=supplier_history,
        project_history=project_history,
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

    Pure with respect to `stats`: given the same cleaned frame and the same
    frozen artefact, the output is identical every time.
    """
    out = clean_df.copy()

    # --- benchmark ratios ------------------------------------------------
    bench = [
        stats.benchmark_median(c, r)
        for c, r in zip(out["procurement_category"], out["region"])
    ]
    out["benchmark_median"] = [b[0] for b in bench]
    out["benchmark_support_n"] = [b[1] for b in bench]
    out["benchmark_is_thin"] = [F.THIN_REFERENCE_GROUP in b[2] for b in bench]
    out["amount_vs_category_region_median"] = [
        _safe_ratio(a, m) for a, m in zip(out["amount_usd"], out["benchmark_median"])
    ]

    prac = [stats.practice_benchmark(p) for p in out["global_practice"]]
    out["practice_median"] = [p[0] for p in prac]
    out["amount_vs_practice_median"] = [
        _safe_ratio(a, m) for a, m in zip(out["amount_usd"], out["practice_median"])
    ]

    out["amount_percentile_in_category_region"] = [
        stats.amount_percentile(a, c, r)
        for a, c, r in zip(out["amount_usd"], out["procurement_category"], out["region"])
    ]

    out["log_amount"] = np.log10(out["amount_usd"].where(out["amount_usd"] > 0))

    # --- point-in-time history ------------------------------------------
    out["supplier_prior_contract_count"] = stats.supplier_history.count_before_batch(
        out["supplier_key"], out["signing_date"]
    )
    out["supplier_is_known"] = out["supplier_key"].notna()

    out["project_contract_sequence"] = stats.project_history.count_before_batch(
        out["project_id"], out["signing_date"]
    )
    # "First" means strictly: nothing in this project was signed BEFORE this
    # contract. Contracts sharing the project's earliest signing date all
    # qualify (6,010 rows over 3,104 projects). That is deliberate -- the
    # extract records a date, not a time, so within a single day there is no
    # defensible ordering, and inventing one via row order would make the
    # feature depend on file layout. Reviewer-facing meaning: "no prior
    # contract was observable in this project when this one was signed."
    out["is_first_contract_in_project"] = (
        out["project_contract_sequence"] == 0
    ).astype("boolean")
    out.loc[out["project_contract_sequence"].isna(), "is_first_contract_in_project"] = pd.NA

    return out
