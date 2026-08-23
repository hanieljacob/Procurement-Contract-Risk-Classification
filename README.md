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

**Implemented so far: ingestion, cleaning, feature preparation, and the deterministic rule engine.**
The risk model, anomaly check and final cohort assignment build on the interfaces below.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest tests/ -q                                  # 71 tests, ~45s
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
stats    = build_reference_stats(clean)      # fit monthly benchmark vintages
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
  rules.py      apply_rules(enriched) -> RuleOutcome
  summary.py    descriptive summary + the data-quality register
notebooks/01_data_preparation.ipynb   cleaning, features, data-quality register
notebooks/02_rule_engine.ipynb        rule catalogue, calibration, cohort mix
tools/build_notebook.py               regenerates notebook 01 (see note below)
tools/build_rule_notebook.py          regenerates notebook 02
tests/                                71 tests
reports/data_quality_register.csv     generated
```

> `tools/build_notebook.py` regenerates `notebooks/01_data_preparation.ipynb` from scratch and
> **overwrites any edits made in Jupyter**. Edit one or the other, not both.

## Design

**Unknown is not zero.** A supplier with no track record and a supplier whose track record cannot
be determined are different facts. Collapsing them into `0` would be invisible downstream, so every
feature that can be unknowable is tri-state and carries a flag explaining why.

**Every population statistic reads only the past.** A statistic may draw on all years *precisely
because* it only ever reads what preceded the record being scored. Benchmarks are monthly vintages —
for each `(category, region, month)`, the median and quantile sketch of everything signed strictly
before that month — and history counts query the same way. There is no fitting window, and the
train/test split governs the model alone.

This replaced an earlier design that fitted benchmarks over FY2020–22 and froze them, on the
reasoning that real systems publish benchmark tables periodically. That holds for deployment but not
for scoring history: a contract signed in July 2019 was divided by a median containing contracts
signed up to three years later. Measured, the frozen median ran +3.3% against the true as-of value in
FY2020 and −10.6% by FY2026 — look-ahead at one end of the timeline, staleness at the other.

**One code path.** The batch table and the single-record call are asserted to produce identical
values on all 15 features. Building that check is what surfaced three real defects (below).

## Results

| | |
|---|---|
| Records | 288,237 supplier-award rows / 276,417 distinct contracts |
| Scoreable | 288,224 · **13** fail validation and become `NOT_ELIGIBLE` |
| Features | 15, tri-state wherever a value can be unknowable |
| Benchmarks | 2,410 monthly vintages, versioned `v1.0` |
| No benchmark | 2,706 FY2020 records too early for any peer history — reported, not imputed |

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
6. **Benchmark medians move substantially** — the Works median falls from $149,585 to $70,318
   (less than half), and Non-consulting Services swings 2.9x within the window, dropping to $5,819
   in FY2021 before recovering. No single fixed value could have served both ends of that, which is
   why the frozen design had to go.

Four defects caught by asserting the batch and single-record paths agree on every feature:

- a `NaN`-vs-`None` gap that silently substituted the global median for a missing global practice;
- a double space in `Consultant Qualification··Selection` that failed 15,956 records (5.5%) as
  "unmapped method";
- greedy legal-suffix stripping that erased four real supplier names to nothing;
- country codes canonicalised in one path but not the other.

A fifth — the frozen benchmark's look-ahead — was found while calibrating exception thresholds, and
is the reason benchmarks are now vintaged. Each has a regression test, including one asserting that
no vintage cell can be derived from a contract signed in its own month or later.

## Rule engine

Every record passes through a deterministic engine before any model runs. It either **decides** a
record or **defers** it — clearing every control does not make a record `ROUTINE`, only *not
exceptional*, and `ROUTINE` is a verdict only the model gets to draw.

| Outcome | Contracts | Share |
|---|---|---|
| `NOT_ELIGIBLE` | 13 | 0.00% |
| `EXCEPTIONAL` | 4,750 | 1.65% |
| `HIGH_ATTENTION` (safe default) | 2,762 | 0.96% |
| Deferred to the model | 280,712 | 97.39% |

`NOT_ELIGIBLE` needs no new logic — the brief's not-eligible conditions *are* the ten FATAL
data-quality flags the validator already raises, so the engine reads them rather than restating them
as predicates free to drift.

**The brief's example threshold does not survive measurement.** It suggests flagging contracts above
five times the category-and-region median; that flags **20.8% of this portfolio**, because amounts
are heavy-tailed enough that 5× the median is only the 78th percentile. Amount extremity is therefore
expressed as a **percentile of the peer group**, which is directly volume-controllable. All four
active rules together produce 1.65%, holding between 1.4% and 2.2% in every complete fiscal year.

**Country risk is deliberately narrow.** A blanket transparency-index rule would encode geography
rather than conduct and is incoherent when the borrowers are themselves developing economies. Of
1,360 contracts with suppliers in secrecy jurisdictions, 962 are *domestic* — Belize, Panama and the
Marshall Islands are borrowers in their own right. Only the 329 that are offshore **and** foreign to
the borrower are flagged.

**One rule cannot fire.** `SIGNED_OUTSIDE_FISCAL_YEAR_WINDOW` is structurally inert, because the
publisher derives fiscal year from the signing date. It is retained as a guard for unvalidated
upstream data and reported as a control with no coverage, rather than quietly deleted.

**Three-valued logic.** A predicate returns fired / did-not-fire / **could-not-evaluate**, and the
third resolves upward to `HIGH_ATTENTION`. Getting the conjunction wrong — treating any unknown
operand as poisoning the whole rule — sent 9,666 records to a human because a country code was
missing, on a rule that could not have fired anyway. `False AND unknown` is `False`: conservatism
means resolving genuine ambiguity upward, not manufacturing ambiguity the data has already settled.

**Limitation.** Thresholds are calibrated on *volume*, not outcomes. This extract contains no
realised fraud, dispute or cancellation label, so there is no way to measure whether the 4,750
flagged contracts are the right ones. The calibration guarantees an actionable queue and explicit
reasoning; it cannot guarantee precision, and no threshold chosen from this data could.
