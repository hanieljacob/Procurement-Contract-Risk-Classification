# CLAUDE.md

Guidance for working in this repo. Read before changing anything in `src/procurement_risk/`.

## What this is

A pipeline that classifies World Bank contract awards into review cohorts
(`NOT_ELIGIBLE` / `EXCEPTIONAL` / `HIGH_ATTENTION` / `ROUTINE`) so reviewer effort concentrates where
it matters. Every record is scored **as of its contract signing date**, using only information
available at that point.

The brief says "at the time the contract was submitted", which is earlier. There is no submission
date in this extract, so signing is the anchor (`config.ASSESSMENT_ANCHOR`), and it is late by the
submission-to-signature interval. Do not silently re-word this either way: the requirement and the
implemented anchor are different things, and both belong in any description of the pipeline.

All five stages are implemented: ingestion and cleaning, feature preparation, the deterministic rule
engine, the risk model, the anomaly check, and final cohort assignment with an audit record.

## Commands

```bash
pytest tests/ -q                                   # 120 tests, ~85s
python3 tools/build_notebook.py                    # regenerate notebook 01
python3 tools/build_rule_notebook.py               # regenerate notebook 02
python3 tools/build_model_notebook.py              # regenerate notebook 03
python3 tools/build_anomaly_notebook.py            # regenerate notebook 04
python3 tools/build_cohort_notebook.py             # regenerate notebook 05
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/01_data_preparation.ipynb --ExecutePreprocessor.timeout=600
```

There is no package install step — `src/` is added to the path directly
(`sys.path.insert(0, "src")`). Tests do this in `tests/conftest.py`.

## Two rules that govern every change

**1. Unknown is not zero.** A supplier with no track record and a supplier whose track record cannot
be determined are different facts. Any feature that can be unknowable is tri-state — `True` / `False`
/ `None`, or a value / `None` — and carries a `DataQualityFlag` explaining why. Never default a
missing value to `0`, `False`, or a median. If you find yourself writing `.fillna(0)`, stop.

**2. Never impute a required field.** The fields in `config.REQUIRED_FIELDS` either arrive usable or
the record fails with a FATAL flag and `features=None`. An unscoreable record is a finding, not a gap
to patch. Optional fields degrade exactly one feature to `None` and raise a DEGRADED flag; the record
survives.

## Architecture

`pipeline.validate_and_enrich(record, stats) -> EnrichedRecord` is the entry point. It is **pure** —
no clock, no globals, no file reads — which is what makes the audit record reproducible. Keep it that
way; if you need the current time, inject it.

**Every population statistic is date-filtered (`features.py`) — the load-bearing rule.** A statistic
may draw on all years *precisely because* it only ever reads what preceded the record being scored.
There is no fitting window, and `config.TRAIN_FISCAL_YEARS` governs the model split only — never
feature construction.

- **Benchmarks** are monthly vintages: for each `(category, region, month)`, the median and a
  101-point quantile sketch of every contract signed strictly before that month. Look them up with
  the record's signing date; `benchmark_median(category, region, as_of)`.
- **History counts** query the same way — "signed strictly before this record" — via
  `PointInTimeCounter`.

This was not the original design, and the reason matters. Benchmarks used to be fitted over
FY2020–22 and frozen, so a contract signed in July 2019 was divided by a median containing contracts
signed up to three years *later*. Measured: the frozen median ran +3.3% against the true as-of value
in FY2020 and −10.6% by FY2026 — look-ahead at one end of the timeline, staleness at the other. If
you are tempted to reintroduce a fitting window for a new statistic, that is the bug you are
recreating.

**Warm-up is a real outcome.** Early records have too little prior history for any benchmark
(2,706 rows, all FY2020). They get `None` plus `REFERENCE_MEDIAN_UNAVAILABLE` — never a fallback
guess — and downstream must route them to HIGH_ATTENTION, not ROUTINE.

## Gotchas, each of which has already caused a real bug

- **Missing values arrive in three disguises**: `None` from a dict record, `float('nan')` from a
  pandas `.map`, `pd.NA` from an Arrow-backed string column. Always use `cleaning._is_missing()`,
  never `is None`. Checking only for `None` once let `NaN` fall through and silently substitute the
  global median for a benchmark we did not have.
- **Row grain vs contract grain.** 8,584 contract numbers span several supplier rows (joint ventures),
  most repeating the *full* amount on each. Reporting is per row; every population statistic must go
  through `cleaning.contract_grain()` first. Raw-row totals overstate value by ~12%.
- **Procurement methods must be looked up via `config.method_lookup_key()`**, never by raw string.
  The source spells CQS with a double space; exact-string matching failed 15,956 records (5.5%) as
  "unmapped method".
- **`INDIVIDUAL CONSULTANT` is a placeholder, not a supplier** — 63,603 rows (22.1%). Supplier
  history for these returns `None`, never `0`.
- **Both notebooks are generated** — `tools/build_notebook.py` builds notebook 01,
  `tools/build_rule_notebook.py` builds notebook 02 — and regenerating **overwrites Jupyter edits**.
  Edit the builder or the notebook, not both.

## Rule engine (`rules.py`)

`apply_rules(enriched) -> RuleOutcome` either **decides** a record or **defers** it. `cohort is None`
means every rule was evaluable and none fired, so the record proceeds to the model — clearing the
rules does not make it `ROUTINE`, and nothing here may assign `ROUTINE`.

- **Predicates are three-valued**: `True` fired, `False` did not, `None` could not be evaluated
  because a feature it depends on is unknown. Build conjunctions with `_kleene_and`, never with
  `and` — `False AND unknown` must be `False`. Poisoning conjunctions with any unknown operand sent
  9,666 records to a human on a rule that could not have fired anyway.
- **`None` resolves upward** to `HIGH_ATTENTION`, and the audit record names which control was blind
  via `UNEVALUABLE::<rule_id>` reason codes.
- **`NOT_ELIGIBLE` is derived from the FATAL flags**, never restated as predicates. Two definitions
  of the same rule will drift.
- A new rule needs `condition`, `threshold`, `rationale` **and** `why_hard_rule` — the last being why
  it is deterministic policy rather than something the model should infer. `test_every_rule_is_fully_documented`
  enforces this.
- **Thresholds are calibrated on reviewable volume**, targeting 1–2% EXCEPTIONAL, and
  `test_exceptional_volume_stays_reviewable` fails outside 1–3%. The amount rule takes the form the
  brief specifies — a *defined multiple* of the peer-group median — with the multiple as the
  parameter (`EXCEPTIONAL_AMOUNT_MEDIAN_MULTIPLE`, default 150×). The brief's illustrative 5× would
  flag 20.9%; `config.py` carries the full volume-per-setting table. If you change any threshold,
  re-run the calibration table in notebook 02 and update the figures in `README.md`.
- Rules that cannot fire on this data stay in the registry with `active=False` rather than being
  deleted, so the write-up can report a control with no coverage.

## Risk model (`model.py`)

The label is **defined, not observed** — no realised outcome exists in this data. It is computable
from two columns we hold, so **leakage is the default outcome here and has to be actively
prevented**. Three separate leaks were found by measurement:

- the two columns the label names, plus `amount_vs_category_region_median`, `amount_usd` and
  `log_amount`, which rebuild them — all listed in `LABEL_DEFINING_FEATURES`;
- `amount_vs_practice_median`, the amount against a different peer grouping (AUC 0.81 alone);
- `supplier_is_known`, which forces the label to zero for 22.5% of records.

Before adding any feature, ask whether it can reconstruct the amount relative to its peer group, or
the procurement method. If yes it is leakage, whatever its name suggests.
`test_honest_feature_set_does_not_score_suspiciously_well` fails above AUC 0.90 as a tripwire.

- **`structurally_negative` records cannot be fixed by feature selection.** Their missingness pattern
  carries the same signal. Report headline metrics on both populations.
- **`TrainedModel` has two column lists.** `source_columns` are read off a record;
  `columns` are post-one-hot design names. Conflating them silently zeroed every categorical and cost
  13 points of AUC while still returning plausible scores — `test_scoring_uses_source_columns_not_design_columns`
  guards it.
- Thresholds are chosen on the **validation year only** and applied unchanged to test.

## Anomaly check (`anomaly.py`)

Unsupervised, so **the model's leakage discipline does not apply** — there is no label to leak into,
and the full feature set including amount and method is correct here. Do not "fix" it by copying
`LABEL_DEFINING_FEATURES` across.

- Fitted on **training years only**: "unusual relative to the training population" needs that
  population to predate what it judges.
- **Explanations are templated and deterministic.** Never generate them. An audit record needs the
  same contract to yield the same sentence years later, traceable to the feature values behind it.
- **Percentiles must use the midpoint of the tied range.** `np.searchsorted` defaults to
  `side="left"`, which counts values strictly less than the input, so a value shared by most of the
  population reads as maximally extreme. This shipped once and made "a single-supplier award" the
  headline reason on nearly every flagged record.
- Explanations take **at most one phrase per feature family** (`_FAMILY`), so a sentence names two
  different kinds of unusual. Pipeline internals like `benchmark_support_n` stay out of `_PHRASING`.
- `apply_anomaly_override` only ever moves a cohort **up**.

## Cohort assignment (`cohort.py`)

`classify_contract(record, artefacts, timestamp)` is the only entry point downstream consumers use.

- **The timestamp is a required argument.** Never call `datetime.now()` here. That single line is
  what makes reproducibility testable, and `test_timestamp_is_injected_not_read_from_a_clock`
  enforces it.
- **Build the model/detector frame from `enriched.normalized`, not the raw record.** The raw record
  carries the publisher's column names ("Region"); everything downstream expects snake_case. Feeding
  raw names in leaves every categorical one-hot at zero and yields confident wrong scores — the same
  failure that cost 13 points of AUC inside `model.py`.
- **`ROUTINE` is only ever reached when every stage completed and none objected.** Any exception
  resolves upward with a reason code. Do not add a code path that can reach ROUTINE on failure.
- **Reason codes are tri-state and must not overclaim.** An unidentifiable supplier gets
  `SUPPLIER_HISTORY_UNAVAILABLE`, never `SUPPLIER_HAS_NO_PRIOR_CONTRACTS`. Nothing may assert a prior
  contract was "clean" — no outcome data exists in this dataset.
- **Contract-group facts cannot be inferred from one row.** `consortium_size` returns `None` plus a
  flag when unsupplied. Do not reinstate a default of 1.

## Conventions

- New quality signals go in `quality.py` as a `DataQualityFlag` with a severity **and** a description.
  The enum is the shared vocabulary for every downstream stage and for audit reason codes — adding a
  bare string anywhere else breaks that contract.
- Thresholds and taxonomies live in `config.py` with a comment giving the *reason*, not just the
  value. Judgment calls are labelled `JUDGMENT CALL:`.
- Comments explain **why**, not what. This codebase is read as much for its reasoning as its
  behaviour; a comment restating the line below it is noise.
- Any new feature computed in batch must produce identical values to the single-record path. Assert
  it — `test_scalar_and_batch_feature_paths_agree` is what caught three real defects.
- Integration tests assert the specific reconciliation numbers quoted in `README.md`. If a number
  changes, update both, and check the change was intended.

## Data

`data/contract_awards_..._08-22-2026.csv` — 107 MB, 288,237 rows, from
<https://financesone.worldbank.org/> (DS00005). Not in version control. `loading.load_raw()` caches
a parsed Parquet copy keyed on the CSV's size and mtime; delete `data/*.parquet` to force a re-parse.
