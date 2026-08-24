# Procurement Contract Risk Classification

Classifies World Bank contract awards into review cohorts so that reviewer effort concentrates
where it adds the most value:

| Cohort | Meaning |
|---|---|
| `NOT_ELIGIBLE` | Incomplete or unscoreable — route to manual data correction |
| `EXCEPTIONAL` | Triggers a mandatory control — route to a senior reviewer |
| `HIGH_ATTENTION` | Elevated risk or anomalous — prioritised review |
| `ROUTINE` | Standard, familiar pattern — reduced scrutiny plus periodic sampling |

Every record is scored **as of its contract signing date**, using only information
available at that point.

A contract is assessed at *submission*, which is earlier than signing — prior
review happens before a contract is signed. That cannot be honoured literally here: **the extract
contains no submission date.** The signing date is the only per-record time signal; `Fiscal Year` and
`Contract signed - Calendar year` are both provably derived from it, and `As of Date` holds a single
value for all 288,237 rows. So the anchor is late by the submission-to-signature interval, and the
point-in-time guarantee is mildly optimistic. See "The assessment anchor" below for the measured size
of that.

**All five stages are implemented**: ingestion and cleaning, feature preparation, the deterministic
rule engine, the risk model, the anomaly check, and final cohort assignment with an audit record.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest tests/ -q                                  # 120 tests, ~85s
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

> Every simplifying assumption this pipeline makes — and what each would cost if wrong — is
> documented in **[ASSUMPTIONS.md](ASSUMPTIONS.md)**.

## Layout

```
ASSUMPTIONS.md                        every assumption, its basis, and its cost
src/procurement_risk/
  config.py     frozen decisions: fiscal calendar, method taxonomy, thresholds, placeholders
  loading.py    CSV -> DataFrame with a fingerprinted Parquet cache
  cleaning.py   normalisation, supplier resolution, consortium handling
  quality.py    DataQualityFlag vocabulary (FATAL / DEGRADED / NOTICE)
  features.py   ReferenceStats (fit) + engineer_features (transform)
  pipeline.py   validate_and_enrich(record) -> EnrichedRecord     <- entry point
  rules.py      apply_rules(enriched) -> RuleOutcome
  model.py      build_target / train / score_record
  anomaly.py    fit_detector / describe / apply_anomaly_override
  cohort.py     classify_contract(record, artefacts, timestamp)   <- single entry point
  summary.py    descriptive summary + the data-quality register
notebooks/01_data_preparation.ipynb   cleaning, features, data-quality register
notebooks/02_rule_engine.ipynb        rule catalogue, calibration, cohort mix
notebooks/03_risk_model.ipynb         leakage analysis, models, thresholds
notebooks/04_anomaly_detection.ipynb  out-of-distribution check + explanations
notebooks/05_cohort_assignment.ipynb  audit record, reproducibility, safe default
tests/                                120 tests
reports/data_quality_register.csv     generated
```

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
values on all 16 features. That check caught three of the seven defects listed below; the
others came from the quality register, the requirements audit and the AUC sanity gate.

## Results

| | |
|---|---|
| Records | 288,237 supplier-award rows / 276,417 distinct contracts |
| Scoreable | 288,224 · **13** fail validation and become `NOT_ELIGIBLE` |
| Features | 16, tri-state wherever a value can be unknowable |
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

Seven defects found, by four different checks. Three came from asserting that the batch and
single-record paths agree on every feature:

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
| `EXCEPTIONAL` | 5,178 | 1.80% |
| `HIGH_ATTENTION` (safe default) | 2,762 | 0.96% |
| Deferred to the model | 280,284 | 97.24% |

`NOT_ELIGIBLE` needs no new logic — the not-eligible conditions *are* the ten FATAL data-quality
flags the validator already raises, so the engine reads them rather than restating them
as predicates free to drift.

**The multiple is the parameter.** The rule flags contracts above a defined multiple of the
category-and-region median. Defining that multiple is the judgment work: at 5× the control flags **20.9% of the portfolio**,
about 8,585 contracts a year, because amounts are heavy-tailed enough that 5× the median is only the
78th percentile. Defined at **150×** (`config.EXCEPTIONAL_AMOUNT_MEDIAN_MULTIPLE`), it yields ~739 a
year. All four active rules together produce 1.80%, holding between 1.6% and 2.2% in every complete
fiscal year. The full volume-per-setting table is in `config.py` and notebook 02.

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
realised fraud, dispute or cancellation label, so there is no way to measure whether the 5,178
flagged contracts are the right ones. The calibration guarantees an actionable queue and explicit
reasoning; it cannot guarantee precision, and no threshold chosen from this data could.

## Risk model

Scores the 280,284 contracts the rule engine defers. **The label is defined, not observed** — this
extract contains no realised fraud, dispute or cancellation, so the label is defined by rule: *top
quartile of the category-and-region peer group, and a non-competitive method*.

**Both halves of that definition are columns we already hold**, so a model handed them scores a
perfect AUC of 1.0000. That is target leakage, and it is measured rather than asserted — a test
asserts the tautology exists, and a second asserts the shipped feature set does not reproduce it.

Two further leaks had to be found by measurement:

- `amount_vs_practice_median` is the amount against a *different* peer grouping — AUC 0.81 alone,
  correlation 0.83 with the peer percentile.
- `supplier_is_known` is deterministic: the placeholder supplier appears only on Individual
  Consultant Selection, which is competitive, so **62,992 contracts (22.5%) are guaranteed
  negatives**. Feature selection cannot fix this — the missingness of
  `supplier_prior_contract_count` carries the same signal — so headline metrics are reported on both
  populations.

| Model | Test AUC | On records that *can* be positive | Precision @ top decile (base rate 0.033) |
|---|---|---|---|
| Gradient-boosted trees | 0.845 | 0.799 | **0.150** (4.5× lift) |
| Logistic regression | 0.815 | 0.761 | 0.127 |

Split: train FY2020–22, calibrate on FY2023, test once on FY2024–26. FY2027 quarantined as a
seven-week stub. Calibration is Platt rather than isotonic — isotonic scores marginally better but
collapses 51,783 distinct scores into 107 steps, flattening the threshold curve.

**Threshold chosen on recall**, and the cost stated plainly: reaching 90%
recall means flagging **40% of the portfolio at 7.3% precision**. That is the evidence that this
model should not be a standalone gate. The band is the useful output — `LOW` / `MEDIUM` / `HIGH`
carry high-attention rates of **1.1% / 6.2% / 15.1%**, a 13× gradient that supports prioritising a
queue.

Per-record output is risk score, band, and the top three contributing features via SHAP on the tree
model — so the explanation comes from the model doing the scoring, not a stand-in.

**Limitation.** Every figure here measures agreement with a definition, never with reality. Since the
label is fully computable from two columns, a production system would apply the rule rather than
model it; what the model adds is a graded contextual ranking over the records the rules clear.

## Anomaly detection

An out-of-distribution check beside the model, not inside it. The model asks *"does this look like
what we defined as high attention?"*; this asks *"does this look like anything we have seen before?"*

**The model's leakage rules deliberately do not apply.** That discipline existed because the label was
computable from two of its own inputs. This detector is unsupervised — it never sees the label — so
there is no target to leak into, and it uses the full feature set including amount and method. That
is necessary: a useful explanation cites exactly those features.

Isolation Forest, fitted on the **training years only**, at 1% contamination.

| Split | Flagged | High-attention rate among flagged | Base rate |
|---|---|---|---|
| Train FY2020–22 | 1.00% | 0.228 | 0.042 |
| Test FY2024–26 | **3.27%** | 0.142 | 0.033 |

The rising flag rate is the finding, not a defect — the portfolio drifts away from the distribution
the detector was fitted on, which is what an OOD check exists to reveal. A deployed version needs
refitting on a schedule, with that rate monitored as an alarm in its own right.

**Explanations are templated and deterministic, not generated.** The same contract must produce the
same sentence at an audit two years from now, traceable to the values that caused it. This is a place
where *not* using a language model is the engineering decision.

> *"This contract is unusual for its training population: it has a supplier foreign to the borrower
> country, combined with a non-competitive procurement method."*

The first version reported "a single-supplier award" as the most unusual feature of nearly every
flagged contract — 93% of contracts have one supplier. `np.searchsorted` defaults to `side="left"`,
which counts values *strictly less than* the input, so the most common value landed at percentile 0
and read as maximally extreme. The sentences were fluent, plausible and wrong; only reading the
output caught it.

A low-scoring but anomalous contract becomes `HIGH_ATTENTION`, never `ROUTINE`. The
override only moves records **up** the precedence order. On the test years it promotes 1,991
contracts whose high-attention rate is **2.5% against 0.7%** for low-risk records it does not flag.

## Final cohort assignment

`classify_contract(record, artefacts, timestamp)` is the single entry point: raw record in, complete
audit record out.

| Cohort | Contracts | Share |
|---|---|---|
| `NOT_ELIGIBLE` | 13 | 0.00% |
| `EXCEPTIONAL` | 5,178 | 1.80% |
| `HIGH_ATTENTION` | 117,909 | 40.91% |
| `ROUTINE` | 165,137 | 57.29% |

**HIGH_ATTENTION at 41% is the direct cost of the conservative threshold**, not an accident. Reaching
90% recall on the risk model means flagging roughly 40% of the portfolio (see the risk-model section),
and safe default behaviour matters more than maximising the routine share.
A real deployment would negotiate that recall floor against reviewer capacity — the threshold is one
constant, and the trade-off table in notebook 03 prices every alternative.

**Reproducibility.** The classification timestamp is an injected argument, never read from a clock —
a function calling `datetime.now()` cannot be tested for reproducibility. All five artefact versions
are stamped on every record, because a decision is only reproducible against a *set* of artefacts:
recording the model version while the benchmark table changed underneath would look reproducible
without being so.

**Safe default, tested by breaking things.** Asserting that failures resolve to `HIGH_ATTENTION` is
easy; the test deliberately breaks the model and confirms that zero records reach `ROUTINE`.

**Reason codes state what is known and never more.** An obvious code to emit would be
`SUPPLIER_HAS_PRIOR_CLEAN_CONTRACTS`. We emit `SUPPLIER_HAS_PRIOR_CONTRACTS` and drop *clean*
deliberately: nothing in this extract establishes that any contract was clean — there are no
findings, disputes or cancellations, only that contracts existed. Every code is tri-state, so an
unidentifiable supplier yields `SUPPLIER_HISTORY_UNAVAILABLE`, never `SUPPLIER_HAS_NO_PRIOR_CONTRACTS`.

**What one record cannot know.** `consortium_size` is a property of the contract, not of a supplier
row. The first version defaulted it to 1, which was silently wrong for the 20,404 rows belonging to
joint ventures — and the batch-versus-per-record check caught it, as it has four times before. It is
now `None` with a flag. Supplied with contract-group context the two paths agree exactly (0
disagreements in 600); without it, 8 in 600 resolve conservatively and say why.

## The assessment anchor

Everything in this pipeline is anchored on the **contract signing date** (`config.ASSESSMENT_ANCHOR`).
Assessment happens at *submission*, which is earlier than signing. The gap is real, unfixable from this
data, and small but not zero — so it is stated rather than glossed.

**Why signing:** there is no submission date in the extract. Of the four date-ish columns, `Fiscal
Year` is exactly `year + (month >= 7)` of the signing date and `Contract signed - Calendar year` is
exactly its year — both verified derived. `As of Date` is a single constant (`2026-08-22`), the
publisher's snapshot; anchoring to it would score every contract as of Aug 2026, granting each record
years of its own future. Signing date is the standard proxy, and it is what days-into-fiscal-year measures — *"a proxy for submission timing within the fiscal cycle"*.

**Measured exposure**, if assessment truly precedes signing by 90 days:

| | Extra information the signing anchor grants | Consequence |
|---|---|---|
| Benchmarks | median **597** extra peer contracts | Peer groups run to thousands, so the median moves only **1.2–1.8% per quarter** — immaterial, and finer than the monthly vintage granularity anyway |
| Supplier history | **~1.2** extra prior contracts (mean) | Small in aggregate, decisive at the boundary: it can flip `is_first_contract_in_project` or `supplier_prior_contract_count == 0`, both of which feed an EXCEPTIONAL rule |

`config.ASSESSMENT_LAG_DAYS` exists and is deliberately **0**. Shifting every as-of lookup earlier
would be more literally correct, but no lag value is supportable from this data — that would
trade a stated assumption for an invented one.
