# Procurement Contract Risk Classification

Classifies World Bank contract awards into review cohorts so that reviewer effort concentrates
where it adds the most value:

| Cohort | Meaning |
|---|---|
| `NOT_ELIGIBLE` | Incomplete or unscoreable — route to manual data correction |
| `EXCEPTIONAL` | Triggers a mandatory control — route to a senior reviewer |
| `HIGH_ATTENTION` | Elevated risk or anomalous — prioritised review |
| `ROUTINE` | Standard, familiar pattern — reduced scrutiny plus periodic sampling |

Every record is scored **as if at the moment the contract was submitted**, using only information
available at that point.

**Implemented so far: ingestion, cleaning, and feature preparation.** The rule engine, risk model,
anomaly check and cohort assignment build on the interfaces below.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest tests/ -q                                  # 38 tests, ~10s
jupyter lab notebooks/01_data_preparation.ipynb   # the analysis and reasoning
```

Or run the whole pipeline headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/01_data_preparation.ipynb --ExecutePreprocessor.timeout=600
```

Use it directly from Python:

```python
import sys; sys.path.insert(0, "src")
from procurement_risk.loading import load_raw
from procurement_risk.cleaning import clean_frame
from procurement_risk.features import build_reference_stats, engineer_features
from procurement_risk.pipeline import validate_and_enrich

clean    = clean_frame(load_raw())
stats    = build_reference_stats(clean)      # fit benchmarks (FY2020-22), frozen
features = engineer_features(clean, stats)   # batch transform

result = validate_and_enrich(clean.iloc[0].to_dict(), stats)
result.ok                   # False -> NOT_ELIGIBLE
result.features             # 15 model-ready features, or None if validation failed
result.data_quality_flags   # e.g. ['SUPPLIER_UNIDENTIFIABLE']
```

## Data

`data/contract_awards_in_investment_project_financing_since_fy_2020_08-22-2026.csv`
— 107 MB, 288,237 rows, extract dated 2026-08-22, from
<https://financesone.worldbank.org/> (DS00005). Not in version control.

The loader caches the parsed frame as Parquet next to the CSV, keyed on the source file's size and
mtime, so an updated extract invalidates the cache automatically. First load ~1.5s, subsequent
loads ~0.2s. Delete `data/*.parquet` to force a re-parse.

## Layout

```
src/procurement_risk/
  config.py     frozen decisions: fiscal calendar, method taxonomy, thresholds, placeholders
  loading.py    CSV -> DataFrame with a fingerprinted Parquet cache
  cleaning.py   normalisation, supplier resolution, consortium handling
  quality.py    DataQualityFlag vocabulary (FATAL / DEGRADED / NOTICE)
  features.py   ReferenceStats (fit) + engineer_features (transform)
  pipeline.py   validate_and_enrich(record) -> EnrichedRecord     <- entry point
  summary.py    descriptive summary + the data-quality register
notebooks/01_data_preparation.ipynb   analysis, reasoning and results
tools/build_notebook.py               regenerates the notebook (see note below)
tests/test_feature_pipeline.py        38 tests
reports/data_quality_register.csv     generated
```

> `tools/build_notebook.py` regenerates `notebooks/01_data_preparation.ipynb` from scratch and
> **overwrites any edits made in Jupyter**. Edit one or the other, not both.

## Design

**Unknown is not zero.** A supplier with no track record and a supplier whose track record cannot
be determined are different facts. Collapsing them into `0` would be invisible downstream, so every
feature that can be unknowable is tri-state and carries a flag explaining why.

**Benchmarks are fitted; history is queried.** Both are population statistics, but they leak
differently. A median has no date filter, so one computed across all years imports the future —
benchmarks are therefore fitted on FY2020–22 only and frozen into a versioned artefact. History
counts *do* have a date filter ("signed strictly before this record"), so restricting their store
to training years would not reduce leakage, it would just make a FY2025 record wrongly look like a
first-time supplier — that store spans every year. Reversing these two is the easy mistake and it
damages both the model and the audit trail.

**One code path.** The batch table and the single-record call are asserted to produce identical
values on all 15 features. Building that check is what surfaced three real defects (below).

## Results

| | |
|---|---|
| Records | 288,237 supplier-award rows / 276,417 distinct contracts |
| Scoreable | 288,224 · **13** fail validation and become `NOT_ELIGIBLE` |
| Features | 15, tri-state wherever a value can be unknowable |
| Benchmarks | frozen on FY2020–22, versioned `v1.0` |

Findings that shaped the design:

1. **`INDIVIDUAL CONSULTANT` is a placeholder, not a supplier** — 63,603 rows (22.1%) across
   53,905 supplier IDs, appearing only on individual-consultant awards. Counting its "history"
   would make a fifth of the dataset look like the most established vendor in the portfolio, when
   these are the records we know least about. Supplier history returns `None`, not `0`.
2. **Joint ventures repeat the full contract amount** on each supplier row — $130.1B at row grain
   vs $116.6B at contract grain. All population statistics use contract grain.
3. **Missing borrower country codes are not random** — multi-country programmes have no
   domestic-supplier answer, so the feature is `None` rather than defaulting to `False`.
4. **Fiscal-year windows are derived, not recorded** — no record falls outside its published FY,
   so a "signed outside the fiscal year window" rule cannot fire. Reported as an inactive control
   rather than presented as working.
5. **FY2027 is truncated** — 1,170 rows against a ~41,000 norm and 17.3% prior review against ~7%,
   because the extract was frozen seven weeks into the year. Quarantined from the test split, which
   uses FY2024–26.

Three defects caught by the batch-vs-single-record agreement test:

- a `NaN`-vs-`None` gap that silently substituted the global median for a missing global practice;
- a double space in `Consultant Qualification··Selection` that failed 15,956 records (5.5%) as
  "unmapped method";
- greedy legal-suffix stripping that erased four real supplier names to nothing.

Each has a regression test.
