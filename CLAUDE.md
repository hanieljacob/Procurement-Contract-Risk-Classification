# CLAUDE.md

Guidance for working in this repo. Read before changing anything in `src/procurement_risk/`.

## What this is

A pipeline that classifies World Bank contract awards into review cohorts
(`NOT_ELIGIBLE` / `EXCEPTIONAL` / `HIGH_ATTENTION` / `ROUTINE`) so reviewer effort concentrates where
it matters. Every record is scored **as if at the moment the contract was signed**, using only
information available at that point.

Implemented so far: ingestion, cleaning, feature preparation. The rule engine, risk model, anomaly
check and cohort assignment are not built yet.

## Commands

```bash
pytest tests/ -q                                   # 38 tests, ~10s
python3 tools/build_notebook.py                    # regenerate the notebook
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
- **The notebook is generated** by `tools/build_notebook.py` and regenerating **overwrites Jupyter
  edits**. Edit one or the other, not both.

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
