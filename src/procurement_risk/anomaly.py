"""Out-of-distribution check: contracts unlike anything in the training population.

The risk model answers "does this look like the contracts we defined as high
attention?". This answers a different question: "does this look like *anything*
we have seen before?". A contract can score low on the first and still fail the
second -- an unfamiliar shape the model has no basis to judge -- and the brief is
explicit that such a record becomes HIGH_ATTENTION rather than ROUTINE.

**The Part 3 leakage discipline deliberately does not apply here.** That existed
because the label was computable from two of its own inputs, so a supervised
model trained on them learned a tautology. This detector is unsupervised: it
never sees the label, so there is no target to leak into. It therefore uses the
full feature set including amount and method -- which is not merely permitted but
necessary, since "unusually high amount relative to the category median, using a
non-competitive method" is precisely the kind of finding it exists to surface.

Explanations are **templated and deterministic**, not generated. A procurement
decision has to be reproducible and auditable: the same contract must yield the
same sentence today and at an audit two years from now, and that sentence has to
be traceable to the feature values that produced it. A generated sentence that
varied between runs would break the audit record for no gain in accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config
from .rules import Cohort

DETECTOR_VERSION = "v1.0"

# Share of the training population treated as out-of-distribution. Calibrated on
# reviewable volume, as everywhere else: these records are promoted to
# HIGH_ATTENTION, so the flag has to stay rare enough to mean something. At 1% on
# the training years the rate rises to ~3% on the test years, and that rise is a
# finding rather than a defect -- the portfolio moves away from what the detector
# was fitted on, which is exactly what an out-of-distribution check should show.
CONTAMINATION = 0.01

# Sentinel for missing values inside the forest. IsolationForest cannot take NaN,
# and a far-out-of-range constant keeps "unknown" separable from any real value
# instead of blending it into the distribution -- consistent with the rest of the
# pipeline, where unknown is never quietly imputed.
_MISSING = -999.0

NUMERIC_FEATURES: tuple[str, ...] = (
    "log_amount",
    "amount_vs_category_region_median",
    "amount_percentile_in_category_region",
    "amount_vs_practice_median",
    "supplier_prior_contract_count",
    "project_contract_sequence",
    "days_into_fiscal_year",
    "consortium_size",
    "benchmark_support_n",
)

BOOLEAN_FEATURES: tuple[str, ...] = (
    "is_competitive_method",
    "supplier_is_domestic",
    "is_first_contract_in_project",
    "supplier_in_secrecy_jurisdiction",
)

# How each feature is described to a reviewer. (high phrasing, low phrasing).
_PHRASING: dict[str, tuple[str, str]] = {
    "log_amount": ("an unusually large contract amount", "an unusually small contract amount"),
    "amount_vs_category_region_median": (
        "an amount far above the median for its category and region",
        "an amount far below the median for its category and region"),
    "amount_percentile_in_category_region": (
        "an amount at the top of its peer group", "an amount at the bottom of its peer group"),
    "amount_vs_practice_median": (
        "an amount far above the median for its global practice",
        "an amount far below the median for its global practice"),
    "supplier_prior_contract_count": (
        "a supplier with an unusually long contract history",
        "a supplier with little or no prior contract history"),
    "project_contract_sequence": (
        "an unusually late position in a long-running project",
        "an unusually early position in its project"),
    "days_into_fiscal_year": (
        "signing unusually late in the fiscal year", "signing unusually early in the fiscal year"),
    # Only the high side is a finding: a single-supplier award is the ordinary
    # case, so there is no "low" phrasing to offer.
    "consortium_size": ("an unusually large joint venture", ""),
}

_BOOLEAN_PHRASING: dict[tuple[str, bool], str] = {
    ("is_competitive_method", False): "a non-competitive procurement method",
    ("is_competitive_method", True): "a competitive procurement method",
    ("supplier_is_domestic", False): "a supplier foreign to the borrower country",
    ("supplier_is_domestic", True): "a domestic supplier",
    ("is_first_contract_in_project", True): "being the first contract in its project",
    ("is_first_contract_in_project", False): "following earlier contracts in its project",
    ("supplier_in_secrecy_jurisdiction", True):
        "a supplier registered in a jurisdiction with minimal ownership transparency",
    ("supplier_in_secrecy_jurisdiction", False): "a supplier in a transparent jurisdiction",
}


# Features grouped by what they say about a contract. The explanation takes at
# most one phrase per family, so a sentence names two *different* kinds of
# unusual rather than saying "a large amount, combined with a large amount".
_FAMILY: dict[str, str] = {
    "log_amount": "amount",
    "amount_vs_category_region_median": "amount",
    "amount_percentile_in_category_region": "amount",
    "amount_vs_practice_median": "amount",
    "supplier_prior_contract_count": "supplier",
    "supplier_is_domestic": "supplier",
    "supplier_in_secrecy_jurisdiction": "supplier",
    "project_contract_sequence": "project",
    "is_first_contract_in_project": "project",
    "days_into_fiscal_year": "timing",
    "is_competitive_method": "method",
    "consortium_size": "structure",
}


def _to_float(series: pd.Series) -> pd.Series:
    if str(series.dtype) in ("boolean", "bool"):
        return series.astype("float64")
    return pd.to_numeric(series, errors="coerce").astype("float64")


@dataclass
class AnomalyDetector:
    """Fitted detector plus the reference distribution its explanations need."""

    forest: IsolationForest = field(repr=False)
    columns: list[str] = field(repr=False)
    reference: dict[str, np.ndarray] = field(repr=False)   # sorted training values
    rates: dict[str, dict[float, float]] = field(repr=False)  # boolean value frequencies
    version: str = DETECTOR_VERSION
    contamination: float = CONTAMINATION

    # -- scoring ---------------------------------------------------------
    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in self.columns:
            out[c] = _to_float(df[c]) if c in df.columns else np.nan
        return out.fillna(_MISSING)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Higher means more anomalous. Sign-flipped so it reads intuitively."""
        return -self.forest.score_samples(self._matrix(df))

    def is_anomalous(self, df: pd.DataFrame) -> np.ndarray:
        return self.forest.predict(self._matrix(df)) == -1

    # -- explanation -----------------------------------------------------
    def _extremity(self, feature: str, value) -> tuple[float, str] | None:
        """How far into the training distribution's tail this value sits, 0-1."""
        if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
            return None
        v = float(value)
        if feature in self.rates:
            rate = self.rates[feature].get(v)
            if rate is None:
                return 1.0, "unseen"
            return 1.0 - rate, "rare"
        ref = self.reference.get(feature)
        if ref is None or len(ref) == 0:
            return None
        # Midpoint of the tied range, not its lower edge. `searchsorted` defaults
        # to side="left", which counts values STRICTLY less than v -- so a value
        # shared by most of the population lands at percentile 0 and reads as
        # maximally extreme. consortium_size == 1 is the ordinary case for 93% of
        # contracts, and that defect had the explanation announcing "a
        # single-supplier award" as the most unusual thing about nearly every
        # flagged record.
        lo = float(np.searchsorted(ref, v, side="left"))
        hi = float(np.searchsorted(ref, v, side="right"))
        pct = ((lo + hi) / 2.0) / len(ref)
        return abs(pct - 0.5) * 2.0, ("high" if pct >= 0.5 else "low")

    def describe(self, record: pd.Series | dict, top_k: int = 2) -> str:
        """One sentence naming what makes this contract unusual.

        Deterministic and template-driven: the same record always yields the same
        sentence, which is what an audit record requires.
        """
        rec = record if isinstance(record, dict) else record.to_dict()
        scored: list[tuple[float, str]] = []
        for feature in self.columns:
            got = self._extremity(feature, rec.get(feature))
            if got is None:
                continue
            extremity, direction = got
            if feature in self.rates:
                value = bool(rec.get(feature))
                phrase = _BOOLEAN_PHRASING.get((feature, value))
                # A common value is not a finding; only flag the rare side.
                if phrase is None or extremity < 0.5:
                    continue
            else:
                pair = _PHRASING.get(feature)
                if pair is None or extremity < 0.80:
                    continue
                phrase = pair[0] if direction == "high" else pair[1]
                if not phrase:      # no phrasing offered for this direction
                    continue
            scored.append((extremity, phrase, _FAMILY.get(feature, feature)))

        if not scored:
            return ("This contract was flagged as unusual by the overall pattern of its features "
                    "rather than by any single one standing out.")
        scored.sort(key=lambda t: -t[0])
        phrases, seen = [], set()
        for _, phrase, family in scored:
            if family in seen:
                continue
            seen.add(family)
            phrases.append(phrase)
            if len(phrases) == top_k:
                break
        if len(phrases) == 1:
            return f"This contract is unusual for its training population: it has {phrases[0]}."
        return (f"This contract is unusual for its training population: it has {phrases[0]}, "
                f"combined with {phrases[1]}.")


def fit_detector(
    train_df: pd.DataFrame,
    contamination: float = CONTAMINATION,
    random_state: int = 0,
) -> AnomalyDetector:
    """Fit on the TRAINING years only.

    The brief says to train on model-assessment-eligible records; restricting
    further to the training fiscal years keeps the same discipline as everything
    else in the pipeline. "Unusual relative to the training population" is only
    meaningful if the training population predates what is being scored --
    otherwise a contract helps define the distribution it is then judged against.
    """
    columns = list(NUMERIC_FEATURES + BOOLEAN_FEATURES)
    columns = [c for c in columns if c in train_df.columns]

    matrix = pd.DataFrame(index=train_df.index)
    for c in columns:
        matrix[c] = _to_float(train_df[c])

    forest = IsolationForest(
        n_estimators=200, contamination=contamination,
        random_state=random_state, n_jobs=-1,
    ).fit(matrix.fillna(_MISSING))

    reference = {
        c: np.sort(matrix[c].dropna().to_numpy())
        for c in columns if c in NUMERIC_FEATURES
    }
    rates = {
        c: matrix[c].value_counts(normalize=True, dropna=True).to_dict()
        for c in columns if c in BOOLEAN_FEATURES
    }
    return AnomalyDetector(
        forest=forest, columns=columns, reference=reference,
        rates=rates, contamination=contamination,
    )


def assess(detector: AnomalyDetector, record: pd.DataFrame) -> dict:
    """Anomaly verdict and explanation for one record."""
    flagged = bool(detector.is_anomalous(record)[0])
    return {
        "anomaly_flag": flagged,
        "anomaly_score": round(float(detector.score(record)[0]), 5),
        "anomaly_explanation": detector.describe(record.iloc[0]) if flagged else None,
        "detector_version": detector.version,
    }


def apply_anomaly_override(cohort: "Cohort", anomaly_flag: bool) -> "Cohort":
    """A low-scoring but anomalous contract becomes HIGH_ATTENTION, never ROUTINE.

    The brief requires this, and the reasoning is worth keeping in view: a low
    risk score on an out-of-distribution record is not evidence of low risk. It
    means the model was asked about something unlike anything it was trained on,
    and a confident answer there is worth less than no answer at all. The
    override only ever moves a record *up*: it never rescues one the rules
    already stopped, and never downgrades EXCEPTIONAL or NOT_ELIGIBLE.
    """
    if not anomaly_flag:
        return cohort
    return min(cohort, Cohort.HIGH_ATTENTION)
