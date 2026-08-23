"""Deterministic rule engine. Every record passes through here before any model.

The engine either **decides** a record or **defers** it:

  NOT_ELIGIBLE   a required field is missing or unusable -- the record cannot be
                 scored at all, which is a finding to route for correction.
  EXCEPTIONAL    a mandatory control fired. Deterministic policy, not prediction.
  HIGH_ATTENTION a rule could not be evaluated because a feature it depends on is
                 unknown. Ambiguity resolves upward, never downward.
  (defer)        no rule fired and all were evaluable -- pass to the model.

Deferring is why `RuleOutcome.cohort` is optional. A record that clears every
rule is not thereby ROUTINE; it is merely not exceptional, and ROUTINE is a
conclusion only the model gets to draw.

Why any of this is a rule rather than something a model should learn: each of
these conditions is a *policy commitment* that must hold regardless of what the
data happens to support. A model trained on a portfolio where large direct
selections are common would learn that they are unremarkable, which is exactly
backwards for a control. Rules also have to be explainable to the person whose
contract was stopped, and stable enough that a threshold change is a documented
decision rather than a retraining artefact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

from . import config
from .pipeline import EnrichedRecord
from .quality import DataQualityFlag as F


class Cohort(IntEnum):
    """Review cohorts, ordered by precedence.

    Lower wins. Precedence is therefore `min()` over whatever fired, rather than
    a nest of conditionals whose ordering has to be re-read to be trusted.
    """

    NOT_ELIGIBLE = 0
    EXCEPTIONAL = 1
    HIGH_ATTENTION = 2
    ROUTINE = 3

    def __str__(self) -> str:  # audit records carry the name, not the int
        return self.name


# `True` fires, `False` does not, `None` means the rule could not be evaluated
# because a feature it depends on is unknown -- which is a third outcome, not a
# quiet `False`.
Predicate = Callable[[dict], "bool | None"]

# Reason-code prefix for a control that could not be evaluated, as opposed to
# one that was evaluated and did not fire. The distinction is the whole point of
# the safe default, so it has to survive into the audit record.
UNEVALUABLE_PREFIX = "UNEVALUABLE::"


@dataclass(frozen=True)
class Rule:
    """One documented rule. Every field here is required by the write-up."""

    id: str
    cohort: Cohort
    condition: str          # what it checks, in words
    threshold: str          # the value used
    rationale: str          # why this condition indicates elevated risk
    why_hard_rule: str      # why deterministic rather than left to the model
    predicate: Predicate
    active: bool = True     # False = retained but cannot fire on this data


def _kleene_and(*values: "bool | None") -> "bool | None":
    """Three-valued AND, where None means unknown.

    The rule that matters: **False AND unknown is False**, not unknown. A rule
    whose first condition definitively fails has definitively not fired, however
    little we know about its second condition. Treating any unknown operand as
    poisoning the whole conjunction sent 9,666 records to a human because a
    country code was missing -- on a rule that could not have fired anyway, since
    the supplier was demonstrably not registered offshore.

    Being conservative means resolving genuine ambiguity upward. It does not mean
    manufacturing ambiguity that the data has already settled.
    """
    if any(v is False for v in values):
        return False
    if any(v is None for v in values):
        return None
    return True


def _amount_extreme(f: dict) -> bool | None:
    ratio = f.get("amount_vs_category_region_median")
    return None if ratio is None else ratio > config.EXCEPTIONAL_AMOUNT_MEDIAN_MULTIPLE


def _non_competitive_high_value(f: dict) -> bool | None:
    competitive = f.get("is_competitive_method")
    amount = f.get("amount_usd")
    return _kleene_and(
        None if competitive is None else not competitive,
        None if amount is None else amount > config.EXCEPTIONAL_NON_COMPETITIVE_AMOUNT,
    )


def _first_contract_high_value(f: dict) -> bool | None:
    first = f.get("is_first_contract_in_project")
    ratio = f.get("amount_vs_category_region_median")
    return _kleene_and(
        None if first is None else bool(first),
        None if ratio is None else ratio > config.EXCEPTIONAL_FIRST_CONTRACT_RATIO,
    )


def _offshore_foreign_supplier(f: dict) -> bool | None:
    offshore = f.get("supplier_in_secrecy_jurisdiction")
    domestic = f.get("supplier_is_domestic")
    return _kleene_and(
        None if offshore is None else bool(offshore),
        None if domestic is None else not domestic,
    )


def _signed_outside_fy_window(f: dict) -> bool | None:
    days = f.get("days_into_fiscal_year")
    if days is None:
        return None
    return not (0 <= days <= 366)


EXCEPTIONAL_RULES: tuple[Rule, ...] = (
    Rule(
        id="AMOUNT_EXTREME_FOR_PEER_GROUP",
        cohort=Cohort.EXCEPTIONAL,
        condition="Contract amount above a defined multiple of the median for the same "
                  "procurement category and region, as that median stood when the contract "
                  "was signed",
        threshold=f"{config.EXCEPTIONAL_AMOUNT_MEDIAN_MULTIPLE:.0f}x the peer-group median "
                  f"(configurable; the brief's illustrative 5x flags 20.9% of the portfolio)",
        rationale="An award far outside its peer group is either genuinely unusual work "
                  "or a mis-specified one. Both warrant a look before signature, and the "
                  "peer group is what makes the comparison fair -- a $500k consultancy in "
                  "Latin America and a $500k road contract in South Asia are not "
                  "comparable observations.",
        why_hard_rule="A control cannot be a model output: the threshold has to be a "
                      "stated policy that survives retraining, and a reviewer has to be "
                      "able to tell the borrower which number their contract exceeded. "
                      "The multiple is the parameter -- the brief asks for 'a defined "
                      "multiple' and offers five as an illustration. Five would flag 20.9% "
                      "of this portfolio, because amounts are heavy-tailed enough that 5x "
                      "the median is only the 78th percentile; defined here at 150x, which "
                      "yields a queue of roughly 739 contracts a year.",
        predicate=_amount_extreme,
    ),
    Rule(
        id="NON_COMPETITIVE_HIGH_VALUE",
        cohort=Cohort.EXCEPTIONAL,
        condition="Procurement method is non-competitive AND the amount exceeds a fixed "
                  "absolute threshold",
        threshold=f"non-competitive method and amount > ${config.EXCEPTIONAL_NON_COMPETITIVE_AMOUNT:,.0f}",
        rationale="Direct selection, force account and single-source awards are lawful and "
                  "often appropriate. It is the combination with scale that matters: the "
                  "larger the award, the more value rests on a decision no competitor "
                  "tested.",
        why_hard_rule="This is a policy commitment, not an empirical regularity. On a "
                      "portfolio where large direct selections happen to be common a model "
                      "would learn they are unremarkable -- precisely the wrong conclusion "
                      "for a control. An absolute dollar threshold rather than a relative "
                      "one, because fiduciary exposure is absolute.",
        predicate=_non_competitive_high_value,
    ),
    Rule(
        id="OFFSHORE_SUPPLIER_FOREIGN_TO_BORROWER",
        cohort=Cohort.EXCEPTIONAL,
        condition="Supplier registered in a jurisdiction with minimal beneficial-ownership "
                  "transparency AND not registered in the borrower country",
        threshold=f"{len(config.SECRECY_JURISDICTIONS)} jurisdictions "
                  f"({config.SECRECY_JURISDICTIONS_VINTAGE}) and supplier is foreign",
        rationale="Project value leaving through a vehicle whose ownership cannot be "
                  "established is the pattern worth a reviewer's time. The foreign "
                  "condition is what makes it meaningful: several of these jurisdictions "
                  "are borrowers themselves, and a domestic supplier in its own country is "
                  "an ordinary award.",
        why_hard_rule="Deliberately narrow. A blanket transparency-index rule would encode "
                      "geography rather than conduct and is incoherent here, since the "
                      "borrowers are overwhelmingly developing economies -- it would flag "
                      "vast volumes and effectively penalise poverty. Left to a model, "
                      "country would become a proxy variable with no way to audit what it "
                      "had learned.",
        predicate=_offshore_foreign_supplier,
    ),
    Rule(
        id="FIRST_CONTRACT_IN_PROJECT_HIGH_VALUE",
        cohort=Cohort.EXCEPTIONAL,
        condition="No prior contract was observable in this project AND the amount is a "
                  "high multiple of the peer-group median",
        threshold=f"first observable contract and ratio > {config.EXCEPTIONAL_FIRST_CONTRACT_RATIO:.0f}x median",
        rationale="A project's first award sets precedent for the procurement pattern that "
                  "follows it. An outsized first contract is cheaper to question once, at "
                  "the outset, than to unwind after it has been replicated. The multiple "
                  "here is far lower than the standalone amount rule's, because being the "
                  "first contract is itself evidence -- the two conditions together justify "
                  "a bar neither would justify alone.",
        why_hard_rule="An interaction chosen for a governance reason rather than a "
                      "statistical one. A model could find the interaction, but could not "
                      "explain to the borrower why this specific contract was stopped, and "
                      "the explanation is the point of an exception.",
        predicate=_first_contract_high_value,
    ),
    Rule(
        id="SIGNED_OUTSIDE_FISCAL_YEAR_WINDOW",
        cohort=Cohort.EXCEPTIONAL,
        condition="Signing date falls outside the fiscal year the record is reported under",
        threshold="outside 1 July - 30 June of the reported fiscal year",
        rationale="A signing date inconsistent with its reported period suggests a "
                  "backdated award or a reporting error, either of which should be "
                  "resolved by a person.",
        why_hard_rule="INACTIVE ON THIS EXTRACT. The publisher derives Fiscal Year from the "
                      "signing date, so no record can fall outside its own window and this "
                      "rule cannot fire -- verified across all 288,237 records. It is "
                      "retained as a guard for unvalidated upstream data and reported as a "
                      "control with no coverage, because silently deleting it would hide "
                      "that the check is doing no work here.",
        predicate=_signed_outside_fy_window,
        active=False,
    ),
)


@dataclass
class RuleOutcome:
    """What the rule engine concluded, and why.

    `cohort is None` means no rule decided the record: it clears every control
    and proceeds to the model, which alone can call it ROUTINE.
    """

    cohort: Cohort | None
    triggered: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    undecidable: list[str] = field(default_factory=list)

    @property
    def proceeds_to_model(self) -> bool:
        return self.cohort is None


def apply_rules(record: EnrichedRecord) -> RuleOutcome:
    """Classify one enriched record, or defer it to the model."""
    # --- NOT_ELIGIBLE -----------------------------------------------------
    # No new logic: the FATAL data-quality flags already *are* these rules
    # (missing supplier, borrower country, category or amount; non-positive
    # amount; missing or unparseable date; unmappable method). Restating them
    # here as separate predicates would create a second definition free to drift
    # from the one the validator enforces.
    if not record.ok:
        fatal = record.fatal_flags
        return RuleOutcome(
            cohort=Cohort.NOT_ELIGIBLE,
            triggered=[f"NOT_ELIGIBLE::{f}" for f in fatal],
            reason_codes=list(fatal),
        )

    features = record.features or {}
    triggered: list[str] = []
    reasons: list[str] = []
    undecidable: list[str] = []

    for rule in EXCEPTIONAL_RULES:
        result = rule.predicate(features)
        if result is None:
            undecidable.append(rule.id)
        elif result:
            triggered.append(rule.id)
            reasons.append(rule.id)

    if triggered:
        return RuleOutcome(
            cohort=Cohort.EXCEPTIONAL,
            triggered=triggered,
            reason_codes=reasons,
            undecidable=undecidable,
        )

    if undecidable:
        # Safe default. A rule we could not evaluate is not a rule that passed:
        # the record carries an unknown where a control expected an answer, so
        # it goes to a human rather than to the model. Explicit branch, not an
        # accident of control flow.
        return RuleOutcome(
            cohort=Cohort.HIGH_ATTENTION,
            # Name the rules we could not evaluate. An escalation whose audit
            # record cannot say why it was escalated is not auditable, and a
            # reviewer opening this record needs to know which control was blind
            # rather than that "something" was.
            reason_codes=[f"{UNEVALUABLE_PREFIX}{rule_id}" for rule_id in undecidable],
            undecidable=undecidable,
        )

    return RuleOutcome(cohort=None)


def rule_catalogue() -> list[dict]:
    """The documented rule table: condition, threshold, and reasoning."""
    return [
        {
            "rule_id": r.id,
            "cohort": str(r.cohort),
            "status": "active" if r.active else "INACTIVE (cannot fire on this data)",
            "condition": r.condition,
            "threshold": r.threshold,
            "rationale": r.rationale,
            "why_hard_rule": r.why_hard_rule,
        }
        for r in EXCEPTIONAL_RULES
    ]
