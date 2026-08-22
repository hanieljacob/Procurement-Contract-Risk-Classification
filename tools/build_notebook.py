import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Data Preparation and Feature Engineering

**Procurement contract risk classification · World Bank contract awards in IPF, FY2020 onward**

This notebook is the narrative layer. All logic lives in `src/procurement_risk/` so the same
functions can be imported by the downstream stages — rule engine, risk model, anomaly check,
cohort assignment — and unit-tested (`pytest tests/`, 38 tests).

### What this stage produces

1. Parsed and cleaned dates, amounts, country codes, procurement method and supplier fields.
2. Derived features: amount against two benchmarks, supplier domesticity, first-in-project,
   supplier prior-contract count, days into the fiscal year, method competitiveness.
3. A documented register of missing, inconsistent and out-of-range values — *with the reasoning*.
4. A summary of amount distribution, category mix, regional coverage and data-quality patterns.
5. `validate_and_enrich(record)` — returns a **data quality flag rather than silently filling a value**.

### The principle everything else follows from

> Each record is scored **as if at the moment the contract was submitted**, using only
> information available at that point.

Two things follow, and they are the spine of this notebook:

- **Unknown is not zero.** A missing supplier history and a supplier with no history are different
  facts. Collapsing them into `0` is the single most consequential mistake available here, and it
  would be invisible downstream. Every feature that can be unknowable is tri-state.
- **Population statistics must declare their time window.** Benchmarks and history counts leak
  differently, so they are fitted differently. Section 4 explains why.

### Scope decisions

| Decision | Choice | Reasoning |
|---|---|---|
| Population | All 288,237 rows; `review_type` kept as a feature | Reviewer effort concentrates on prior review, but a prior-review-only population leaves ~20k rows over eight fiscal years and ~200 in the most recent — too thin for a time-based split. A prior-review slice is reported separately instead. |
| Sampling | **None** | 288k rows is ~200 MB, well within memory, so sampling buys nothing and would risk interacting with the time-based split. |
| Currency | Raw USD, no inflation adjustment | The extract is already USD-converted at an undisclosed rate/date. Deflating would layer one unverifiable assumption on another. Benchmark drift is measured instead (Section 6). |
| Grain | One row per **supplier-award**; statistics at **contract** grain | 3.1% of contracts are joint ventures split across rows that each repeat the full amount. |
""")

code("""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

from procurement_risk import config, summary as S
from procurement_risk.loading import load_raw
from procurement_risk.cleaning import clean_frame, contract_grain
from procurement_risk.features import build_reference_stats, engineer_features, FEATURE_COLUMNS
from procurement_risk.pipeline import validate_and_enrich

print("pipeline version:", config.PIPELINE_VERSION, "| extract as-of:", config.DATA_AS_OF_DATE)
""")

md("""
## 1. Load

The loader renames columns to snake_case and coerces nothing else. Dates are deliberately **not**
parsed at load time: pandas would coerce an unreadable date to `NaT`, which is indistinguishable
from a genuinely absent one. Both need to reach the quality register as distinct findings.
""")

code("""
raw = load_raw()
print(f"{len(raw):,} rows x {raw.shape[1]} columns")
raw.head(3).T
""")

md("""
## 2. Cleaning

Cleaning changes *representation* (casing, whitespace, legal suffixes, multi-valued fields).
It never invents a value.
""")

code("""
clean = clean_frame(raw)
print(f"{len(clean):,} rows, {clean.shape[1]} columns after cleaning")
clean[["supplier_name_raw","supplier_name","supplier_key","global_practice_raw","global_practice",
       "procurement_method_raw","method_class","signing_date","fiscal_year",
       "days_into_fiscal_year","supplier_is_domestic","consortium_size"]].head(5)
""")

md("""
### 2.1 The procurement-method taxonomy

An **exhaustive lookup**, never a keyword heuristic. "Method cannot be mapped to a known category"
is a `NOT_ELIGIBLE` rule in the rule engine — a heuristic that always returns *something* would
quietly disable that control. Two entries are genuine judgment calls and are labelled as such.

A concrete trap found here: the source spells CQS as `Consultant Qualification··Selection` with a
**double space**. Any whitespace-collapsing cleaner breaks an exact-string lookup and wrongly fails
15,956 records (5.5%) as "unmapped". Both sides of the lookup are now whitespace- and case-folded,
and there is a regression test for it.
""")

code("""
tax = S.method_taxonomy_table()
counts = clean["procurement_method_raw"].value_counts()
tax["rows"] = tax["method"].map(counts).fillna(0).astype(int)
tax.style.set_properties(subset=["rationale"], **{"white-space": "normal", "text-align": "left"})
""")

code("""
assert clean["method_class"].isna().sum() == 0, "every observed method must map"
print("competitive rows:      {:>7,}".format(int((clean.method_class=='COMPETITIVE').sum())))
print("non-competitive rows:  {:>7,}".format(int((clean.method_class=='NON_COMPETITIVE').sum())))
print("unmapped:              {:>7,}".format(int(clean.method_class.isna().sum())))
""")

md("""
### 2.2 Supplier identity — the biggest single finding

`Supplier` is **not** an entity column for a fifth of this dataset.

`INDIVIDUAL CONSULTANT` appears **63,603 times (22.1%)** across **53,905 distinct Supplier IDs**,
and only ever on `Individual Consultant Selection` rows. It is a category label standing in for a
natural person whose name is withheld.

Why this matters more than it first appears: the feature set needs a count of prior contracts from
the same supplier. Computed naively, 22% of the dataset would share the contract history of the
single most prolific supplier in the portfolio — and would therefore look maximally *familiar* and
*low risk*. That is precisely backwards: these are the records we know least about.

**Decision:** these records are marked supplier-unidentifiable, and supplier-history features return
`None` with a `SUPPLIER_UNIDENTIFIABLE` flag. Not zero — zero would mean "brand new supplier", which
is also a claim we cannot support.

`Supplier ID` is not usable as the entity key either: 183,980 IDs for 121,914 names, and 53,905 of
those IDs belong to the placeholder alone. It behaves like a per-award reference, not a durable
vendor identifier. We key on the normalised name and say so, rather than trusting the field that
merely looks like a primary key.
""")

code("""
ph = clean["supplier_name"] == "INDIVIDUAL CONSULTANT"
print(f"placeholder rows            : {ph.sum():>7,} ({ph.mean()*100:.1f}%)")
print(f"distinct Supplier IDs behind: {raw.loc[ph.values,'supplier_id'].nunique():>7,}")
print(f"methods it appears under    : {clean.loc[ph,'procurement_method_raw'].unique().tolist()}")
print(f"\\nsupplier_key null (placeholder + genuinely missing): {clean.supplier_key.isna().sum():,}")
print(f"resolved distinct suppliers : {clean.supplier_key.nunique():,}")
""")

md("""
### 2.3 Joint ventures duplicate the contract amount

8,584 contract numbers (3.1%) carry more than one supplier row, and most repeat the **full**
contract amount on every row. Any statistic computed over raw rows double-counts money: a
three-way JV would push its amount into the median three times.

**Decision:** reporting stays at row grain (one row per supplier-award, which is what the extract
is), but every *population statistic* — medians, history counts — is computed at contract grain.
""")

code("""
impact = S.consortium_impact(clean, contract_grain(clean))
for k, v in impact.items():
    print(f"{k:<35} {v:>15,}" if isinstance(v,(int,np.integer)) else f"{k:<35} {v:>15}")
print(f"\\ninflation if computed on raw rows: "
      f"{impact['row_grain_total_usd_bn']/impact['contract_grain_total_usd_bn']-1:.1%}")
""")

md("""
### 2.4 Missing borrower country codes are not random

3.35% of rows have no borrower ISO code. Two distinct causes hide behind that one number:

- **Encoding artefacts** — Côte d'Ivoire (4,878 rows), Somalia (2,764). The country is real and
  identifiable; only the code is absent.
- **Genuine multi-country programmes** — "DRC - Angola", "Western Balkans", "Southern Africa",
  "World". There is no single borrower country, so *"is the supplier domestic?" has no answer.*

**Decision:** `supplier_is_domestic` is **tri-state**. `None` for regional programmes and for rows
where either side's code is absent. Returning `False` would assert the supplier is foreign, which
is a claim the data does not support — and one that would push a record toward higher risk on the
basis of a formatting gap.
""")

code("""
print(clean.supplier_is_domestic.value_counts(dropna=False).rename(
    {True:"domestic", False:"foreign"}).rename_axis("supplier_is_domestic").to_frame("rows"))
print(f"\\nregional/multi-country borrower rows: {int(clean.borrower_is_regional.sum()):,}")
print(clean.loc[clean.borrower_is_regional,"borrower_country"].value_counts().head(8).to_frame("rows"))
""")

md("""
### 2.5 Fiscal-year windows — a rule that cannot fire

The WB fiscal year runs 1 July – 30 June. Recomputing FY from the signing date and comparing to the
published column gives **zero** disagreements, and every record sits 0–365 days into its own FY.

That is not a clean bill of health — it is evidence that the publisher **derives** `Fiscal Year`
from the signing date rather than recording it independently. So the check has no power to detect
anything on this extract.

**Consequence for the rule engine, stated now rather than discovered later:** an exception rule of
the form *"contract signed outside the expected fiscal year window"* can never fire on this data.
It is retained as an input guard for unvalidated records, but it will be reported as an inactive
control rather than presented as if it were doing work.
""")

code("""
mismatch = (clean.fiscal_year != clean.fiscal_year_published.astype("Int64")).sum()
print(f"derived vs published FY disagreements: {mismatch}")
print(f"days_into_fiscal_year range: {clean.days_into_fiscal_year.min()} .. {clean.days_into_fiscal_year.max()}")
print(f"\\nsigning dates: {clean.signing_date.min():%Y-%m-%d} .. {clean.signing_date.max():%Y-%m-%d}")
print(clean.fy_quarter.value_counts().sort_index().rename_axis("fy_quarter").to_frame("rows"))
""")

md("""
## 3. Degenerate records

Small enough to enumerate individually rather than describe statistically. These become the
`NOT_ELIGIBLE` population.
""")

code("""
deg = clean[(clean.amount_usd.isna()) | (clean.amount_usd <= 0) | (clean.supplier_name.isna())]
print(f"{len(deg)} degenerate records")
deg[["fiscal_year","region","procurement_category","procurement_method_raw",
     "supplier_name_raw","amount_usd","review_type"]]
""")

md("""
## 4. Feature engineering — what is fitted vs. what is queried

This is the central design decision of the whole pipeline.

Two kinds of population statistic feed the features, and **they leak differently**:

| | Benchmark medians | History counts |
|---|---|---|
| Example | "amount vs. median for this category and region" | "prior contracts from this supplier" |
| Has a date filter? | **No** | **Yes** — count only what was signed strictly before |
| Leakage risk | A median over all years is contaminated by contracts signed *after* the record | None: the date predicate does the work |
| Therefore fitted on | **Training fiscal years only (FY2020–22), then frozen** | **All available years** |

Restricting the *history* store to training years would not reduce leakage — it would simply make a
FY2025 record wrongly look like a first-time supplier. Floating the *median* over all years would
silently import the future. Getting these two backwards is the easy mistake, and it damages both the
model and the audit trail.

Freezing the benchmark also mirrors how a real review system works: a benchmark table is a versioned
artefact refreshed on a schedule, not recomputed per request. That is what makes the guarantee
"same input always returns the same output for a fixed model version" achievable at all. The cost —
drift as the portfolio moves — is measured in Section 6 rather than asserted away.
""")

code("""
stats = build_reference_stats(clean)          # fit — FY2020-22 only for medians
features = engineer_features(clean, stats)    # transform — pure w.r.t. stats

print(f"benchmark window      : FY{stats.median_fiscal_years}")
print(f"category x region cells: {len(stats.category_region_median)}")
print(f"suppliers in history   : {len(stats.supplier_history):,}")
print(f"projects in history    : {len(stats.project_history):,}")
print(f"global median amount   : ${stats.global_median:,.0f}")
""")

md("""
### 4.1 Thin peer groups

Across the full extract, Category × Region cells range from **n=1** (Works × Other) to n=25,773.
Within the FY2020–22 fitting window two cells still fall below the support threshold. A median over
eight observations is not a benchmark. Below a support threshold of 30 the peer group widens up a ladder
(category+region → category → global) and a `THIN_REFERENCE_GROUP` flag is raised. The support
count `benchmark_support_n` travels with the feature so a downstream stage can discount a thin
cell rather than trusting it blindly.
""")

code("""
cr = pd.DataFrame({"n": pd.Series(stats.category_region_count),
                   "median_usd": pd.Series(stats.category_region_median)})
cr["below_support_threshold"] = cr.n < config.MIN_GROUP_SUPPORT
print(f"cells below support threshold ({config.MIN_GROUP_SUPPORT}): {int(cr.below_support_threshold.sum())}")
cr.sort_values("n").head(6)
""")

md("""
### 4.2 "First contract in project" — a deliberate tie rule

`is_first_contract_in_project` means strictly: **nothing in this project was signed before this
contract**. Contracts sharing the project's earliest signing date all qualify — 6,010 rows across
3,104 projects.

That is intentional. The extract records a *date*, not a timestamp, so within a single day there is
no defensible ordering. Breaking ties by row order would make the feature depend on file layout,
which is not a property of the contract. The reviewer-facing meaning is the honest one: *no prior
contract was observable in this project when this one was signed.*
""")

code("""
print(features.supplier_prior_contract_count.describe().to_frame("supplier_prior_contract_count"))
print(f"\\nnull (supplier unidentifiable): {int(features.supplier_prior_contract_count.isna().sum()):,}")
print(f"first-in-project rows: {int((features.is_first_contract_in_project==True).sum()):,} "
      f"over {features.project_id.nunique():,} projects")
""")

md("""
## 5. The data-quality register

Built by running the **same** `validate_and_enrich` that serves single records over every row of the
extract (~33 µs/record). A vectorised second implementation would have been faster and would have
been free to drift from the first — and the register's entire purpose is to describe what the
pipeline actually does.

Severity contract:

- **FATAL** → the record cannot be responsibly scored. Fails, and becomes `NOT_ELIGIBLE`.
- **DEGRADED** → one feature is unavailable. Record survives; that feature is `None` and downstream
  must treat `None` as unknown, never as zero.
- **NOTICE** → recorded for audit; nothing is wrong with the record.
""")

code("""
audit = S.run_quality_audit(features, stats)
register = S.quality_register(audit)
print(f"scoreable: {int(audit.ok.sum()):,}   failed (NOT_ELIGIBLE): {int((~audit.ok).sum()):,}")
register[register.rows > 0].style.hide(axis="index")
""")

md("""
### 5.1 Flags that never fire — and why that is reported, not hidden

A control that cannot trigger is worth as much attention as one that triggers often, because it
tells you where the data gives you no coverage. Reporting these openly is the difference between a
register and a marketing document.
""")

code("""
never = register[register.rows == 0][["flag","severity","meaning"]]
never.style.hide(axis="index")
""")

md("""
Reading of the inactive flags:

- `SIGNING_DATE_*`, `PROCUREMENT_*_MISSING`, `AMOUNT_UNPARSEABLE` — the published extract is
  well-formed on these fields. The checks are retained because `validate_and_enrich` is meant to
  accept *raw* records from an upstream system, not only this cleaned file.
- `FISCAL_YEAR_DISAGREEMENT` / `SIGNING_DATE_OUTSIDE_FY_WINDOW` — inactive by construction
  (Section 2.5): FY is derived from the signing date.
- `PROCUREMENT_METHOD_UNMAPPED` — zero **now**. It fired on 15,956 rows before the double-space
  defect was fixed, which is exactly the case it exists to catch.

### 5.2 One decision to *remove* a check

An early version flagged supplier names that looked like natural persons, on the theory that
individuals have less reliable histories than firms. It fired on **41% of rows** and its hits
included `NOVA GLOBAL SRL`, `SEACOM KENYA` and `RED MANGO`. A flag with that precision is worse
than no flag: it trains a reviewer to ignore it, and it would have fed noise into the model.

It was removed rather than tuned. Recording the removal matters — a register that only lists checks
that survived tells you nothing about the judgment applied.
""")

md("""
## 6. Summary statistics

### 6.1 Contract amounts

Extremely heavy-tailed and spanning nine orders of magnitude: median $33,001, 99th percentile
$7.4M, maximum $553M. Consequences carried into Parts 3–4:

- Raw amount is unusable as a linear feature; `log_amount` and the ratio-to-benchmark features are
  the modelling inputs.
- Amount must always be compared **within** a peer group. A $500k consulting contract in Latin
  America and a $500k works contract in South Asia are not comparable observations.
""")

code("""
S.amount_distribution(features)
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
amt = features.loc[features.amount_usd > 0, "amount_usd"]
axes[0].hist(np.log10(amt), bins=70, color="#4C72B0", edgecolor="white", linewidth=.3)
axes[0].set(xlabel="log10(contract amount USD)", ylabel="contracts",
            title=f"Contract amounts (n={len(amt):,})")
axes[0].axvline(np.log10(amt.median()), color="#C44E52", ls="--",
                label=f"median ${amt.median():,.0f}")
axes[0].legend(frameon=False)

order = features.groupby("procurement_category")["amount_usd"].median().sort_values().index
axes[1].boxplot([np.log10(features.loc[features.procurement_category==c, "amount_usd"].dropna()
                          .pipe(lambda s: s[s>0])) for c in order],
                labels=[c.replace(" ", "\\n") for c in order], showfliers=False)
axes[1].set(ylabel="log10(amount USD)", title="Amount by procurement category")
plt.tight_layout(); plt.show()
""")

md("""
### 6.2 Procurement category mix and regional coverage

Coverage is broad but uneven: the two Africa regions hold 44% of contracts. The `Other` region
(295 rows: "World", "Multi-Regional") is what drives the thin-cell fallback in Section 4.1.
""")

code("""
S.category_region_mix(features)
""")

code("""
S.region_coverage(features)
""")

md("""
### 6.3 Fiscal-year coverage — the caveat the model split depends on

The intended split is FY2020–22 for training, FY2023 for validation, and **the most recent available
fiscal year** for the final test. Taken literally that last one is FY2027 — and FY2027 is a trap:

- **1,170 rows** against a ~41,000 norm, because the extract was frozen seven weeks into it.
- Prior-review share of **17.3%** vs. the ~7% baseline — large contracts clear prior review and get
  reported quickly; small post-review contracts trickle in over months.

Testing on FY2027 would measure **reporting lag, not model skill**, and would do so in a direction
that flatters the model — over-representing exactly the large, non-competitive contracts the risk
target is built from.

**Decision:** FY2027 is quarantined as an explicitly-labelled truncated holdout; FY2024–26 serve as
the test window. Taking "most recent available" literally here would produce a misleading
evaluation, so it is deliberately not taken literally.
""")

code("""
S.fiscal_year_coverage(features)
""")

md("""
### 6.4 Benchmark drift — the cost of freezing

Freezing medians on FY2020–22 buys reproducibility and removes look-ahead bias. It costs accuracy as
the portfolio moves. Quantified rather than asserted, so the refresh cadence can be an evidence-based
decision. Drift within roughly ±25% is tolerable for a *relative* risk benchmark; beyond that the
artefact should be re-fitted and its version bumped.
""")

code("""
S.benchmark_drift(clean)
""")

md("""
### 6.5 Prior-review slice

The population modelled is all records, but the business framing concerns prior review. This slice
confirms the two groups are genuinely different — prior-review contracts have a median ~25x higher
and a 95th percentile ~23x higher —
which is why `review_type` is retained as a feature rather than used as a filter.
""")

code("""
pd.DataFrame({
    "contracts": features.groupby("review_type").size(),
    "median_usd": features.groupby("review_type").amount_usd.median().round(0),
    "p95_usd": features.groupby("review_type").amount_usd.quantile(.95).round(0),
    "non_competitive_pct": (features.groupby("review_type").is_competitive_method
                            .apply(lambda s: (s==False).mean())*100).round(1),
})
""")

md("""
## 7. `validate_and_enrich` — the per-record contract

The entry point every downstream stage builds on.

```python
EnrichedRecord(
    ok: bool,                       # False -> classified NOT_ELIGIBLE
    features: dict | None,          # None when a required field failed
    data_quality_flags: list[str],  # audit-record reason codes
    normalized: dict,               # cleaned passthrough for the audit record
)
```

Guarantees:

- **Never imputes a required field.** A missing amount fails the record; it does not become a median.
- **Degrades precisely.** An optional-field problem nulls exactly one feature and flags it.
- **Pure.** No clock, no globals, no file reads — only the record and the frozen `ReferenceStats`.
  This is what makes the audit record reproducible.
- **One code path.** The batch table and the per-record call are verified to agree on all 15
  features (test: `test_scalar_and_batch_feature_paths_agree`). Building this check is what surfaced
  two real defects — a `NaN`-vs-`None` gap that silently substituted the global median for a missing
  global practice, and the double-space method bug.
""")

code("""
import json
def show(title, rec):
    res = validate_and_enrich(rec, stats)
    print(f"=== {title} ===")
    print(f"ok={res.ok}  flags={res.data_quality_flags}")
    if res.features:
        print(json.dumps({k: (round(v,4) if isinstance(v,float) else v)
                          for k,v in res.features.items()}, indent=2, default=str))
    print()

routine = features[(features.method_class=="COMPETITIVE") &
                   (features.amount_vs_category_region_median.between(0.8,1.2)) &
                   (features.supplier_prior_contract_count>3)].iloc[0].to_dict()
large_direct = features[(features.method_class=="NON_COMPETITIVE") &
                        (features.amount_vs_category_region_median>50)].iloc[0].to_dict()
broken = features[features.amount_usd==0].iloc[0].to_dict()

show("A. Routine: competitive, at benchmark, established supplier", routine)
show("B. Elevated: non-competitive, far above benchmark", large_direct)
show("C. Fails validation: zero amount + missing supplier", broken)
""")

md("""
Record **C** is the one that matters most. Every required field except the amount and supplier is
present and perfectly usable, and it would be trivially easy to fill the amount with a category
median and produce a confident-looking risk score. The function refuses, returns `features=None`,
and hands the rule engine two reason codes. **An unscoreable record is a finding, not a gap to be
patched.**
""")

md("""
## 8. Assumptions, limitations, and what the next stage inherits

### Assumptions
1. **Amounts are comparable across time without deflation** (Section: Scope). Drift measured in 6.4.
2. **First-listed global practice is primary** — 48.2% of records list several and the field carries
   no ordering guarantee. Exploding to one row per practice would break the one-row-per-award grain
   the pipeline depends on.
3. **Normalised supplier name is the entity key**, not `Supplier ID` (Section 2.2).
4. **`Individual Consultant Selection` is competitive** (≥3 CVs compared under WB rules) even though
   its supplier field is a placeholder. The two facts are recorded independently so a rule can act
   on either.
5. **Benchmarks frozen on FY2020–22** and treated as a versioned artefact.

### Limitations, stated plainly
- **Publication lag is invisible to the pipeline.** History features count what was *signed* before a
  record, but a contract signed earlier may have been *published* later. Point-in-time correctness is
  therefore an upper bound on true information availability. Correcting for it would need a
  publication-date column the extract does not provide.
- **22% of records have no usable supplier history**, and they are not a random 22% — they are
  entirely individual-consultant awards. Any feature importance the model assigns to supplier
  history is conditioned on a non-random subpopulation.
- **No contract outcome data exists here.** There is no realised fraud, dispute or cancellation
  label anywhere in the extract. The risk target must therefore be *defined* rather than observed,
  which makes it partly circular — a limitation the modelling stage has to confront directly rather
  than paper over.
- **`Other` region (295 rows)** is a residual bucket, not a place. Its benchmarks fall back to the
  category level.

### Handed downstream
- `features` — 288,237 rows, 15 model-ready features plus cleaned fields.
- `stats` — the frozen `ReferenceStats` artefact (`v1.0`, benchmarks FY2020–22).
- `validate_and_enrich` — the per-record entry point.
- The `DataQualityFlag` vocabulary — **FATAL flags are the `NOT_ELIGIBLE` rules**, already computed
  and reconciled (13 records).
- Two findings that constrain rule design: the fiscal-year-window rule cannot fire (Section 2.5),
  and non-competitive amount thresholds must be set against *contract-grain* benchmarks
  (Section 2.3).
""")

code("""
OUT = Path.cwd().parent / "data"
features.to_parquet(OUT / "contracts_features.parquet", index=False)
audit.to_parquet(OUT / "quality_audit.parquet")
register.to_csv(Path.cwd().parent / "reports" / "data_quality_register.csv", index=False)
print(f"features : {features.shape}")
print(f"scoreable: {int(audit.ok.sum()):,} / {len(audit):,}")
print("artefacts written to data/ and reports/")
""")

nb["cells"] = C
nb.metadata.kernelspec = {"display_name":"Python 3","language":"python","name":"python3"}
nbf.write(nb, "notebooks/01_data_preparation.ipynb")
print("notebook written:", len(C), "cells")
