"""Final cohort assignment and the audit record.

One entry point -- `classify_contract` -- takes a raw record and returns the
complete structured decision. Everything upstream feeds into it and nothing
downstream reinterprets it.

Two properties govern the design.

**Reproducibility.** The same record must return the same output for a fixed set
of artefact versions, today and at an audit years from now. That is why the
classification timestamp is *injected* rather than read from a clock: a function
that calls `datetime.now()` cannot be tested for reproducibility, and an audit
record that cannot be regenerated is not an audit record. Every stage below is
already pure with respect to its artefacts, and this is the last place that
could have broken it.

**Safe default.** The brief is explicit: a record with missing data, an
unavailable model result, or conflicting rule outputs defaults to HIGH_ATTENTION
rather than ROUTINE. `ROUTINE` is a positive claim -- "we looked, and this is
ordinary" -- and it is only ever reached when every stage completed and none of
them objected. Any failure anywhere resolves upward, with a reason code saying
what failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

import pandas as pd

from . import config
from .anomaly import AnomalyDetector, apply_anomaly_override
from .features import ReferenceStats
from .model import TrainedModel, build_design_matrix, score_record
from .pipeline import EnrichedRecord, validate_and_enrich
from .quality import DataQualityFlag, Severity, SEVERITY
from .rules import Cohort, RuleOutcome, apply_rules

AUDIT_SCHEMA_VERSION = "v1.0"


class ReasonCode(str, Enum):
    """Narrative reason codes for the audit record.

    These sit alongside the two vocabularies that already exist -- data-quality
    flags from `quality.py` and rule ids from `rules.py` -- and cover what those
    two cannot: the *affirmative* reasons a record was found ordinary.

    That asymmetry matters. A reviewer told only what did NOT fire learns
    nothing; the brief's own example output is a routine contract carrying
    positive codes, and a cohort assignment that cannot say why something is
    routine is not auditable in the direction that matters most, since ROUTINE is
    the cohort that receives the least human attention.
    """

    # -- affirmative: why this record looks ordinary ----------------------
    AMOUNT_WITHIN_CATEGORY_RANGE = "AMOUNT_WITHIN_CATEGORY_RANGE"
    COMPETITIVE_PROCUREMENT_METHOD = "COMPETITIVE_PROCUREMENT_METHOD"
    SUPPLIER_HAS_PRIOR_CONTRACTS = "SUPPLIER_HAS_PRIOR_CONTRACTS"
    ESTABLISHED_PROJECT = "ESTABLISHED_PROJECT"
    DOMESTIC_SUPPLIER = "DOMESTIC_SUPPLIER"
    WITHIN_TRAINING_DISTRIBUTION = "WITHIN_TRAINING_DISTRIBUTION"

    # -- adverse: why this record drew attention --------------------------
    AMOUNT_ABOVE_CATEGORY_RANGE = "AMOUNT_ABOVE_CATEGORY_RANGE"
    NON_COMPETITIVE_PROCUREMENT_METHOD = "NON_COMPETITIVE_PROCUREMENT_METHOD"
    SUPPLIER_HAS_NO_PRIOR_CONTRACTS = "SUPPLIER_HAS_NO_PRIOR_CONTRACTS"
    FIRST_CONTRACT_IN_PROJECT = "FIRST_CONTRACT_IN_PROJECT"
    FOREIGN_SUPPLIER = "FOREIGN_SUPPLIER"
    ANOMALOUS_RELATIVE_TO_TRAINING = "ANOMALOUS_RELATIVE_TO_TRAINING"

    # -- unknown: stated, never assumed either way ------------------------
    SUPPLIER_HISTORY_UNAVAILABLE = "SUPPLIER_HISTORY_UNAVAILABLE"
    PEER_BENCHMARK_UNAVAILABLE = "PEER_BENCHMARK_UNAVAILABLE"

    # -- model verdict ----------------------------------------------------
    MODEL_SCORE_BELOW_THRESHOLD = "MODEL_SCORE_BELOW_THRESHOLD"
    MODEL_SCORE_ABOVE_THRESHOLD = "MODEL_SCORE_ABOVE_THRESHOLD"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


# NOTE ON THE BRIEF'S EXAMPLE. Its sample output lists
# "SUPPLIER_HAS_PRIOR_CLEAN_CONTRACTS". We emit SUPPLIER_HAS_PRIOR_CONTRACTS and
# drop the word "clean" deliberately: nothing in this dataset establishes that
# any prior contract was clean. There are no findings, disputes, cancellations or
# audit outcomes anywhere in the extract -- only that contracts existed. Asserting
# "clean" in an audit record would put a claim on file that the evidence cannot
# support, which is exactly the sort of thing an auditor is there to catch.
REASON_DESCRIPTIONS: dict[ReasonCode, str] = {
    ReasonCode.AMOUNT_WITHIN_CATEGORY_RANGE:
        "Amount is within the usual range for its procurement category and region.",
    ReasonCode.COMPETITIVE_PROCUREMENT_METHOD:
        "Awarded through a method that tested price or quality against more than one bidder.",
    ReasonCode.SUPPLIER_HAS_PRIOR_CONTRACTS:
        "Supplier has previously been awarded contracts in this portfolio. This records "
        "familiarity only -- no outcome data exists to establish that those contracts were clean.",
    ReasonCode.ESTABLISHED_PROJECT:
        "Contracts were already signed under this project before this one.",
    ReasonCode.DOMESTIC_SUPPLIER:
        "Supplier is registered in the borrower country.",
    ReasonCode.WITHIN_TRAINING_DISTRIBUTION:
        "Resembles the population the models were fitted on.",
    ReasonCode.AMOUNT_ABOVE_CATEGORY_RANGE:
        "Amount is high relative to the median for its procurement category and region.",
    ReasonCode.NON_COMPETITIVE_PROCUREMENT_METHOD:
        "Awarded without a competitive process.",
    ReasonCode.SUPPLIER_HAS_NO_PRIOR_CONTRACTS:
        "No earlier contract for this supplier had been signed before this one. Note the assessment anchor is the signing date; see config.ASSESSMENT_ANCHOR.",
    ReasonCode.FIRST_CONTRACT_IN_PROJECT:
        "No earlier contract in this project had been signed before this one.",
    ReasonCode.FOREIGN_SUPPLIER:
        "Supplier is registered outside the borrower country.",
    ReasonCode.ANOMALOUS_RELATIVE_TO_TRAINING:
        "Unlike the population the models were fitted on; see the anomaly explanation.",
    ReasonCode.SUPPLIER_HISTORY_UNAVAILABLE:
        "Supplier could not be identified, so contract history is unknown -- not zero.",
    ReasonCode.PEER_BENCHMARK_UNAVAILABLE:
        "No peer benchmark existed when this contract was signed, so amount could not be compared.",
    ReasonCode.MODEL_SCORE_BELOW_THRESHOLD:
        "Risk score fell below the conservative review threshold.",
    ReasonCode.MODEL_SCORE_ABOVE_THRESHOLD:
        "Risk score met or exceeded the conservative review threshold.",
    ReasonCode.MODEL_UNAVAILABLE:
        "The risk model could not score this record; routed for human review by default.",
}

# Amount ratio above which the affirmative "within range" code is withheld.
# Well below the EXCEPTIONAL rule's threshold -- this is narrative, not a control.
_WITHIN_RANGE_RATIO = 5.0


@dataclass(frozen=True)
class PipelineArtefacts:
    """Everything needed to classify a record, versioned together.

    Bundled rather than passed separately because a decision is only
    reproducible against a *set* of artefacts. Recording the model version while
    silently swapping the benchmark table underneath would make the audit record
    look reproducible while not being so.
    """

    stats: ReferenceStats
    model: TrainedModel | None = None
    detector: AnomalyDetector | None = None
    schema_version: str = AUDIT_SCHEMA_VERSION

    def versions(self) -> dict[str, str | None]:
        return {
            "audit_schema": self.schema_version,
            "reference_stats": self.stats.version,
            "model": self.model.version if self.model else None,
            "detector": self.detector.version if self.detector else None,
            "pipeline": config.PIPELINE_VERSION,
        }


def _narrative_codes(features: Mapping[str, Any]) -> list[ReasonCode]:
    """Affirmative and adverse codes describing the record itself.

    Every branch is tri-state: a feature that is unknown produces an *unknown*
    code, never the reassuring one. Saying "supplier has no prior contracts" when
    the supplier could not be identified would be a false statement on the record.
    """
    codes: list[ReasonCode] = []

    ratio = features.get("amount_vs_category_region_median")
    if ratio is None:
        codes.append(ReasonCode.PEER_BENCHMARK_UNAVAILABLE)
    elif ratio > _WITHIN_RANGE_RATIO:
        codes.append(ReasonCode.AMOUNT_ABOVE_CATEGORY_RANGE)
    else:
        codes.append(ReasonCode.AMOUNT_WITHIN_CATEGORY_RANGE)

    competitive = features.get("is_competitive_method")
    if competitive is True:
        codes.append(ReasonCode.COMPETITIVE_PROCUREMENT_METHOD)
    elif competitive is False:
        codes.append(ReasonCode.NON_COMPETITIVE_PROCUREMENT_METHOD)

    prior = features.get("supplier_prior_contract_count")
    if prior is None:
        codes.append(ReasonCode.SUPPLIER_HISTORY_UNAVAILABLE)
    elif prior > 0:
        codes.append(ReasonCode.SUPPLIER_HAS_PRIOR_CONTRACTS)
    else:
        codes.append(ReasonCode.SUPPLIER_HAS_NO_PRIOR_CONTRACTS)

    first = features.get("is_first_contract_in_project")
    if first is True:
        codes.append(ReasonCode.FIRST_CONTRACT_IN_PROJECT)
    elif first is False:
        codes.append(ReasonCode.ESTABLISHED_PROJECT)

    domestic = features.get("supplier_is_domestic")
    if domestic is True:
        codes.append(ReasonCode.DOMESTIC_SUPPLIER)
    elif domestic is False:
        codes.append(ReasonCode.FOREIGN_SUPPLIER)

    return codes


def classify_contract(
    record: Mapping[str, Any],
    artefacts: PipelineArtefacts,
    classification_timestamp: datetime,
) -> dict:
    """Run one raw record through the whole pipeline and return the audit record.

    `classification_timestamp` is a required argument, not a default. The
    temptation is to call `datetime.now()` here; resisting it is what makes the
    output reproducible and the reproducibility testable.
    """
    if classification_timestamp.tzinfo is None:
        raise ValueError("classification_timestamp must be timezone-aware")

    enriched: EnrichedRecord = validate_and_enrich(record, artefacts.stats)
    outcome: RuleOutcome = apply_rules(enriched)

    reason_codes: list[str] = list(outcome.reason_codes)
    features = enriched.features or {}
    risk_score: float | None = None
    risk_band: str | None = None
    anomaly_flag = False
    anomaly_explanation: str | None = None
    top_features: list[dict] = []

    # ---- decided by the rules ------------------------------------------
    if outcome.cohort is not None:
        cohort = outcome.cohort
        if enriched.ok:
            reason_codes += [c.value for c in _narrative_codes(features)]
    else:
        # ---- deferred to the model --------------------------------------
        # Build from the NORMALISED fields, not the raw record. The raw record
        # carries the publisher's column names ("Region", "Procurement
        # Category"); the model and detector expect the internal snake_case
        # names. Feeding raw names in left every categorical one-hot at zero and
        # produced confident, wrong scores -- the same failure that once cost 13
        # points of AUC inside the model itself. Features last so they win on
        # any key both dictionaries define.
        frame = pd.DataFrame([{**enriched.normalized, **features}])
        try:
            if artefacts.model is None:
                raise RuntimeError("no model artefact supplied")
            scored = score_record(artefacts.model, frame)
            risk_score = scored["risk_score"]
            risk_band = scored["risk_band"]
            top_features = scored["top_features"]
            above = risk_score >= artefacts.model.threshold
            cohort = Cohort.HIGH_ATTENTION if above else Cohort.ROUTINE
            reason_codes.append(
                (ReasonCode.MODEL_SCORE_ABOVE_THRESHOLD if above
                 else ReasonCode.MODEL_SCORE_BELOW_THRESHOLD).value
            )
        except Exception:
            # Safe default. An unavailable model result is not a low-risk
            # result: we do not know, so a person decides.
            cohort = Cohort.HIGH_ATTENTION
            reason_codes.append(ReasonCode.MODEL_UNAVAILABLE.value)

        # ---- out-of-distribution override -------------------------------
        if artefacts.detector is not None:
            try:
                anomaly_flag = bool(artefacts.detector.is_anomalous(frame)[0])
                if anomaly_flag:
                    anomaly_explanation = artefacts.detector.describe(frame.iloc[0])
                    reason_codes.append(ReasonCode.ANOMALOUS_RELATIVE_TO_TRAINING.value)
                else:
                    reason_codes.append(ReasonCode.WITHIN_TRAINING_DISTRIBUTION.value)
                cohort = apply_anomaly_override(cohort, anomaly_flag)
            except Exception:
                cohort = min(cohort, Cohort.HIGH_ATTENTION)
                reason_codes.append(ReasonCode.MODEL_UNAVAILABLE.value)

        reason_codes += [c.value for c in _narrative_codes(features)]

    fatal = [f for f in enriched.data_quality_flags
             if SEVERITY[DataQualityFlag(f)] is Severity.FATAL]

    return {
        "cohort": str(cohort),
        "risk_score": risk_score,
        "risk_band": risk_band,
        "anomaly_flag": anomaly_flag,
        "anomaly_explanation": anomaly_explanation,
        "reason_codes": sorted(set(reason_codes)),
        # True when data quality actually impaired the assessment -- FATAL or
        # DEGRADED. NOTICE-level flags are recorded but impair nothing, and
        # MULTI_PRACTICE_PROJECT alone fires on 48% of the portfolio, so
        # including them would make this field mean "almost always".
        "data_quality_flag": any(
            SEVERITY[DataQualityFlag(f)] is not Severity.NOTICE
            for f in enriched.data_quality_flags
        ),
        "data_quality_flags": list(enriched.data_quality_flags),
        "data_quality_fatal": bool(fatal),
        "model_version": artefacts.model.version if artefacts.model else None,
        "top_contributing_features": top_features,
        "feature_snapshot": _snapshot(enriched),
        "artefact_versions": artefacts.versions(),
        "classification_timestamp": classification_timestamp.astimezone(timezone.utc)
                                                            .isoformat()
                                                            .replace("+00:00", "Z"),
    }


def _snapshot(enriched: EnrichedRecord) -> dict:
    """Everything needed to reconstruct the decision without the source dataset.

    Carries the cleaned inputs alongside the derived features. A snapshot of the
    features alone would let you re-check the arithmetic but not tell you which
    contract it described, which is the question an auditor actually asks.
    """
    return {
        "features": enriched.features,
        "normalized_inputs": enriched.normalized,
    }


def describe_reason(code: str) -> str:
    """Human-readable meaning for any code appearing in an audit record."""
    try:
        return REASON_DESCRIPTIONS[ReasonCode(code)]
    except ValueError:
        pass
    try:
        from .quality import DESCRIPTIONS
        return DESCRIPTIONS[DataQualityFlag(code)]
    except ValueError:
        pass
    if code.startswith("UNEVALUABLE::"):
        return f"A control could not be evaluated: {code.split('::', 1)[1]}"
    return f"Rule triggered: {code}"
