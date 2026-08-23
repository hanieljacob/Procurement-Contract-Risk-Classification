"""Risk model: scores the records the rule engine defers.

**The label is synthetic, and that governs everything here.** This extract
contains no realised outcome -- no fraud, dispute or cancellation anywhere -- so
there is nothing to predict in the ordinary sense. A ground-truth label is
therefore *defined* by rule: a contract is high attention when its amount sits in
the top quartile of its category-and-region peer group and its method is
non-competitive.

Two consequences follow, and both belong in any honest reading of the numbers:

1. **The label is computable from two columns we already hold**, so a model given
   those columns scores a perfect AUC of 1.0000. That is target leakage, not
   skill, and the standard fix is to withhold them. `LABEL_DEFINING_FEATURES`
   names them, `feature_columns()` excludes them, and a test asserts they never
   reach the training frame.

2. **Any score measures agreement with a definition, never with reality.** The
   definition encodes an assumption -- that large and non-competitive means risky
   -- which nothing in this data verifies. A model that agrees with it perfectly
   has learned the assumption, not the risk.

What the model is actually for, given that: the rule engine already applies the
deterministic part. The model contributes a *graded contextual score* over the
records that clear it -- useful for ranking the routine population for sampling,
and for records where the peer-group percentile is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

MODEL_VERSION = "v1.0"

# The brief's definition, in one place so it cannot drift.
TARGET_QUANTILE = 0.75

# Columns the label is built from. Withheld from training: including them makes
# the model a tautology (AUC 1.0000 on held-out years).
#
# `amount_vs_category_region_median` has to go as well even though the label does
# not name it -- within a peer group it is monotone in the percentile, so it
# reconstructs the quartile test almost exactly. `amount_usd` and `log_amount`
# rebuild it at one further remove, once the peer group is known.
LABEL_DEFINING_FEATURES: tuple[str, ...] = (
    "amount_percentile_in_category_region",
    "is_competitive_method",
    "amount_vs_category_region_median",
    "amount_usd",
    "log_amount",
)

# `amount_vs_practice_median` is deliberately absent. It is not named by the
# label, but it is the amount measured against a different peer grouping, and
# empirically it correlates 0.83 with the category-and-region percentile and
# scores AUC 0.81 on its own -- the amount in disguise. Any amount-relative-to-a-
# benchmark feature reconstructs the label's first condition.
NUMERIC_FEATURES: tuple[str, ...] = (
    "supplier_prior_contract_count",
    "project_contract_sequence",
    "days_into_fiscal_year",
    "fy_quarter",
    "consortium_size",
    "benchmark_support_n",
)

# `supplier_is_known` is deliberately absent -- see `structurally_negative`.
BOOLEAN_FEATURES: tuple[str, ...] = (
    "supplier_is_domestic",
    "is_first_contract_in_project",
    "supplier_in_secrecy_jurisdiction",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "region",
    "procurement_category",
    "review_type",
)

RISK_BANDS = ("LOW", "MEDIUM", "HIGH")


def build_target(df: pd.DataFrame) -> pd.Series:
    """The brief's label: top-quartile amount for its peer group AND non-competitive.

    The percentile comes from the monthly benchmark vintages, so "top quartile"
    means top quartile *as of the month the contract was signed* -- consistent
    with every other statistic in the pipeline.
    """
    return (
        (df["amount_percentile_in_category_region"] > TARGET_QUANTILE)
        & (df["is_competitive_method"] == False)  # noqa: E712 - pandas needs ==
    ).fillna(False).astype(int)


def structurally_negative(df: pd.DataFrame) -> pd.Series:
    """Records that CANNOT satisfy the label, whatever else is true of them.

    The placeholder supplier ("INDIVIDUAL CONSULTANT") appears only on Individual
    Consultant Selection, which is a competitive method -- so the label's second
    condition fails by construction and these records are guaranteed negatives.
    That is 62,992 contracts, 22.5% of the model-eligible population.

    This is not something feature selection can fix: it is a property of how the
    label was defined meeting a property of the data. Dropping the
    `supplier_is_known` flag does not remove the signal, because
    `supplier_prior_contract_count` is missing on exactly the same rows and a
    tree can split on missingness.

    It matters for *reporting*, not for training. Free negatives inflate every
    aggregate metric, so headline figures are quoted on the population that can
    actually be positive, with the full-population figure shown alongside.
    """
    return ~df["supplier_is_known"].astype("boolean").fillna(False)


def feature_columns(include_label_defining: bool = False) -> list[str]:
    """Training columns. The default deliberately excludes the label's own inputs."""
    cols = list(NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES)
    if include_label_defining:
        cols += list(LABEL_DEFINING_FEATURES)
    return cols


def build_design_matrix(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Numeric design matrix: booleans to float, categoricals one-hot.

    Missing stays missing where the estimator can use it. `HistGradientBoosting`
    handles NaN natively, which matters here -- an unknown supplier history is a
    real state, and filling it with a number would undo the tri-state discipline
    the rest of the pipeline maintains.
    """
    out = pd.DataFrame(index=df.index)
    for c in columns:
        if c not in df.columns:
            continue
        if c in CATEGORICAL_FEATURES:
            dummies = pd.get_dummies(df[c].astype("string"), prefix=c, dtype=float)
            out = out.join(dummies)
        else:
            out[c] = pd.to_numeric(df[c].astype("float64"), errors="coerce")
    return out


def align_columns(X: pd.DataFrame, reference: Sequence[str]) -> pd.DataFrame:
    """Reindex to the training columns; unseen categories become all-zero."""
    return X.reindex(columns=list(reference), fill_value=0.0)


def split_by_fiscal_year(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Time-based split. FY2027 is quarantined, not tested on -- see config."""
    return {
        "train": df[df["fiscal_year"].isin(config.TRAIN_FISCAL_YEARS)],
        "validate": df[df["fiscal_year"].isin(config.VALIDATION_FISCAL_YEARS)],
        "test": df[df["fiscal_year"].isin(config.TEST_FISCAL_YEARS)],
        "quarantined": df[df["fiscal_year"].isin(config.TRUNCATED_FISCAL_YEARS)],
    }


@dataclass
class TrainedModel:
    """A fitted, calibrated model plus everything needed to score reproducibly."""

    name: str
    estimator: Any
    # Two distinct column lists, and conflating them silently zeroed every
    # one-hot column at scoring time: `source_columns` are the feature names read
    # off a record, `columns` are the design-matrix names after one-hot encoding.
    # A design name like "region_South Asia" is not a column on any record.
    source_columns: list[str] = field(repr=False)
    columns: list[str] = field(repr=False)
    threshold: float = 0.5
    band_edges: tuple[float, float] = (0.33, 0.66)
    version: str = MODEL_VERSION
    explainer: Any = field(default=None, repr=False)
    background: np.ndarray | None = field(default=None, repr=False)

    def design_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        return align_columns(build_design_matrix(df, self.source_columns), self.columns)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self.design_matrix(df))[:, 1]

    def band(self, score: float) -> str:
        lo, hi = self.band_edges
        return RISK_BANDS[0] if score < lo else (RISK_BANDS[1] if score < hi else RISK_BANDS[2])


def train(
    train_df: pd.DataFrame,
    validate_df: pd.DataFrame,
    kind: str = "gbt",
    include_label_defining: bool = False,
) -> TrainedModel:
    """Fit on the training years, calibrate probabilities on the validation year.

    Calibrating on a held-out year is the honest use of a validation split: a raw
    tree score is not a probability, and a reviewer shown "0.8 risk" will read it
    as one. The test years are never touched here.
    """
    cols = feature_columns(include_label_defining)
    Xtr = build_design_matrix(train_df, cols)
    ytr = build_target(train_df)
    Xva = align_columns(build_design_matrix(validate_df, cols), Xtr.columns)
    yva = build_target(validate_df)

    if kind == "logistic":
        base = Pipeline([
            # The linear model cannot take NaN, so impute for it alone -- and
            # accompany each imputed column with a missingness indicator so
            # "unknown" stays visible to the model rather than being erased.
            ("impute", _MedianImputerWithIndicator()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
    elif kind == "gbt":
        base = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=0,
        )
    else:
        raise ValueError(f"unknown model kind: {kind}")

    base.fit(Xtr, ytr)
    # FrozenEstimator keeps the fit from the training years intact while
    # isotonic calibration is learned on the validation year -- sklearn 1.6
    # replaced the old cv="prefit" with this. The test years are untouched.
    # Platt (sigmoid) rather than isotonic. Isotonic calibrates marginally better
    # here (2.8pp vs 5.3pp max deviation) but collapses 51,783 distinct scores
    # into 107 steps, which flattens the threshold curve into unusable plateaus
    # and creates ties that depress measured ranking quality. A risk score has to
    # rank as well as calibrate, so the smooth monotone fit wins.
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(Xva, yva)

    return TrainedModel(name=kind, estimator=calibrated,
                        source_columns=list(cols), columns=list(Xtr.columns))


class _MedianImputerWithIndicator:
    """Median impute, plus a 0/1 column recording what was missing.

    Filling a missing value silently would contradict the rest of the pipeline,
    where unknown is never zero. The indicator keeps the fact of the gap in the
    feature set, so the model can use "we did not know this" as information.
    """

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.medians_ = X.median(numeric_only=True)
        self.missing_ = [c for c in X.columns if X[c].isna().any()]
        return self

    def transform(self, X):
        X = X.reindex(columns=self.columns_)
        out = X.copy()
        for c in self.missing_:
            out[f"{c}__missing"] = X[c].isna().astype(float)
        return out.fillna(self.medians_).fillna(0.0)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


# ---------------------------------------------------------------------------
# Evaluation -- the three measures the brief names
# ---------------------------------------------------------------------------

def precision_at_top_decile(y_true, scores) -> float:
    """Of the 10% of contracts we score riskiest, what share really are?"""
    n = max(1, int(len(scores) * 0.10))
    top = np.argsort(scores)[::-1][:n]
    return float(np.asarray(y_true)[top].mean())


def share_flagged_in_target(y_true, scores, threshold: float) -> float:
    """Of everything flagged at this threshold, what share is high attention?"""
    flagged = np.asarray(scores) >= threshold
    return float(np.asarray(y_true)[flagged].mean()) if flagged.any() else float("nan")


def calibration_table(y_true, scores, bins: int = 10) -> pd.DataFrame:
    """Reliability: predicted probability against observed frequency."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(scores)})
    df["bucket"] = pd.cut(df.p, np.linspace(0, 1, bins + 1), include_lowest=True)
    g = df.groupby("bucket", observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "mean_predicted": g.p.mean().round(4),
        "observed_rate": g.y.mean().round(4),
    }).reset_index()


def evaluate(model: TrainedModel, df: pd.DataFrame) -> dict:
    y = build_target(df)
    p = model.predict_proba(df)
    return {
        "n": len(df),
        "base_rate": round(float(y.mean()), 4),
        "auc": round(float(roc_auc_score(y, p)), 4) if y.nunique() > 1 else float("nan"),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "precision_at_top_decile": round(precision_at_top_decile(y, p), 4),
        "share_flagged_in_target": round(share_flagged_in_target(y, p, model.threshold), 4),
        "flagged_share_of_population": round(float((p >= model.threshold).mean()), 4),
    }


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def threshold_table(y_true, scores) -> pd.DataFrame:
    """Precision, recall and reviewer volume across candidate cut-offs.

    Cut-offs are taken at *score quantiles* rather than on a fixed 0.05 grid.
    With a 3.2% base rate the calibrated scores cluster near zero, so an evenly
    spaced grid never reaches useful recall and silently collapses to its floor.
    Working in flagged-volume terms also matches the question a review function
    actually asks: "if I can look at 5% of contracts, what do I catch?"
    """
    y = np.asarray(y_true); s_ = np.asarray(scores)
    rows = []
    for frac in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20,
                 0.30, 0.40, 0.50, 0.60, 0.70):
        t = float(np.quantile(s_, 1 - frac))
        flag = s_ >= t
        tp = int((flag & (y == 1)).sum())
        rows.append({
            "flag_rate_target": frac,
            "threshold": round(t, 6),
            "flagged": int(flag.sum()),
            "flagged_%": round(float(flag.mean()) * 100, 2),
            "recall": round(tp / max(1, int((y == 1).sum())), 4),
            "precision": round(tp / max(1, int(flag.sum())), 4),
        })
    return pd.DataFrame(rows)


def choose_threshold(y_true, scores, min_recall: float = 0.90) -> float:
    """Smallest flagged volume that still catches `min_recall` of positives.

    Deliberately asymmetric, as the brief requires: missing a genuinely
    high-attention contract costs more than over-flagging a routine one. So
    recall is the *constraint* and reviewer volume is the price -- we take the
    tightest cut-off that still clears the recall floor, rather than maximising
    a balanced score that would trade recall away for precision.

    If no cut-off reaches the floor, the loosest available is returned and the
    shortfall is visible in `threshold_table` rather than hidden.
    """
    tbl = threshold_table(y_true, scores)
    ok = tbl[tbl.recall >= min_recall]
    return float(ok.threshold.max()) if len(ok) else float(tbl.threshold.min())


def choose_band_edges(scores) -> tuple[float, float]:
    """Band cuts from the validation distribution, never from test."""
    return (float(np.quantile(scores, 0.70)), float(np.quantile(scores, 0.90)))


# ---------------------------------------------------------------------------
# Per-record explanation
# ---------------------------------------------------------------------------

def attach_explainer(model: TrainedModel, background: pd.DataFrame) -> TrainedModel:
    """Fit a SHAP explainer against the uncalibrated estimator.

    Calibration is a monotone transform of the score, so contributions computed
    on the underlying estimator keep their ranking and sign -- which is all the
    "top three contributing features" claim depends on.
    """
    import shap

    X = model.design_matrix(background)
    inner = model.estimator.calibrated_classifiers_[0].estimator
    inner = getattr(inner, "estimator", inner)   # unwrap FrozenEstimator
    try:
        model.explainer = shap.TreeExplainer(inner)
    except Exception:
        model.explainer = shap.LinearExplainer(inner, X.to_numpy())
    model.background = X.head(200).to_numpy()
    return model


def top_contributing_features(model: TrainedModel, record: pd.DataFrame, k: int = 3) -> list[dict]:
    """The k features moving this record's score furthest, with direction."""
    X = model.design_matrix(record)
    if model.explainer is None:
        return []
    values = np.asarray(model.explainer.shap_values(X))
    if values.ndim == 3:          # (rows, features, classes)
        values = values[..., -1]
    values = values.reshape(len(X), -1)[0]
    order = np.argsort(np.abs(values))[::-1][:k]
    return [
        {
            "feature": model.columns[i],
            "value": None if pd.isna(X.iloc[0, i]) else float(X.iloc[0, i]),
            "contribution": round(float(values[i]), 5),
            "direction": "increases risk" if values[i] > 0 else "decreases risk",
        }
        for i in order
    ]


def score_record(model: TrainedModel, record: pd.DataFrame) -> dict:
    """Risk score, band and top-three contributors for one record."""
    score = float(model.predict_proba(record)[0])
    return {
        "risk_score": round(score, 4),
        "risk_band": model.band(score),
        "top_features": top_contributing_features(model, record),
        "model_version": model.version,
        "threshold": model.threshold,
    }
