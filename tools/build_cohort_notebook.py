import nbformat as nbf
nb = nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Final Cohort Assignment and Audit Record

One entry point. `classify_contract(record, artefacts, timestamp)` takes a raw record, runs it
through every stage, and returns the complete structured decision.

```
validate_and_enrich  ->  rule engine  ->  risk model  ->  anomaly check  ->  cohort
      (Part 1)            (Part 2)        (Part 3)        (Part 4)         (here)
```

Two properties govern the design.

**Reproducibility.** The same record must return the same output for a fixed set of artefact
versions — today, and at an audit years from now. That is why the classification timestamp is
*injected* rather than read from a clock: a function calling `datetime.now()` cannot be tested for
reproducibility, and an audit record that cannot be regenerated is not an audit record.

**Safe default.** The brief is explicit that a record with missing data, an unavailable model result
or conflicting rule outputs defaults to `HIGH_ATTENTION`, never `ROUTINE`. `ROUTINE` is a positive
claim — *we looked, and this is ordinary* — reached only when every stage completed and none objected.
""")

code("""
import sys, warnings, json
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 90)

from procurement_risk import config, model as M, anomaly as A
from procurement_risk.loading import load_raw
from procurement_risk.cleaning import clean_frame
from procurement_risk.features import build_reference_stats, engineer_features
from procurement_risk.pipeline import validate_and_enrich
from procurement_risk.rules import apply_rules, Cohort
from procurement_risk.cohort import PipelineArtefacts, classify_contract, describe_reason

raw = load_raw()
clean = clean_frame(raw)
stats = build_reference_stats(clean)
features = engineer_features(clean, stats)

eligible = pd.read_parquet(Path.cwd().parent / "data" / "model_eligible.parquet")
splits = M.split_by_fiscal_year(eligible)
gbt = M.train(splits["train"], splits["validate"], kind="gbt")
yva = M.build_target(splits["validate"]); pva = gbt.predict_proba(splits["validate"])
gbt.threshold = M.choose_threshold(yva, pva, min_recall=0.90)
gbt.band_edges = M.choose_band_edges(pva)
gbt = M.attach_explainer(gbt, splits["train"].head(2000))
detector = A.fit_detector(splits["train"])

artefacts = PipelineArtefacts(stats=stats, model=gbt, detector=detector)
TIMESTAMP = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
print(json.dumps(artefacts.versions(), indent=2))
""")

md("""
## 1. The audit record

Four worked examples, one per cohort. This is what a reviewer receives.
""")

code("""
def show(title, record):
    out = classify_contract(record, artefacts, TIMESTAMP)
    print(f"{'='*95}\\n{title}  ->  {out['cohort']}")
    print(f"  risk_score {out['risk_score']}   band {out['risk_band']}   "
          f"anomaly {out['anomaly_flag']}   data_quality_flag {out['data_quality_flag']}")
    if out["anomaly_explanation"]:
        print(f"  {out['anomaly_explanation']}")
    print("  reason codes:")
    for c in out["reason_codes"]:
        print(f"    - {c:<38} {describe_reason(c)[:80]}")

rows = raw.to_dict("records")
# pick one record per cohort from a scan of the first few thousand
found, i = {}, 0
while len(found) < 4 and i < 4000:
    o = classify_contract(rows[i], artefacts, TIMESTAMP)
    found.setdefault(o["cohort"], rows[i]); i += 1
for cohort in ("NOT_ELIGIBLE", "EXCEPTIONAL", "HIGH_ATTENTION", "ROUTINE"):
    if cohort in found:
        show(cohort, found[cohort])
""")

code("""
# The full structured output for one record
print(json.dumps(classify_contract(found.get("ROUTINE") or rows[1], artefacts, TIMESTAMP),
                 indent=2, default=str)[:2200])
""")

md("""
### 1.1 Reason codes say what is known — and never more

The brief's example output includes `SUPPLIER_HAS_PRIOR_CLEAN_CONTRACTS`. We emit
`SUPPLIER_HAS_PRIOR_CONTRACTS` and drop the word **clean** deliberately.

Nothing in this extract establishes that any prior contract was clean. There are no findings,
disputes, cancellations or audit outcomes anywhere in its 21 columns — only that contracts existed.
Writing "clean" into an audit record would put a claim on file that the evidence cannot support,
which is precisely what an auditor exists to catch.

The same discipline runs through every code. Each is tri-state: a supplier who *cannot be identified*
produces `SUPPLIER_HISTORY_UNAVAILABLE`, never `SUPPLIER_HAS_NO_PRIOR_CONTRACTS` — because "no prior
contracts" would be a false statement about a record where we simply do not know.
""")

code("""
placeholder = raw[raw.supplier_name_raw == "INDIVIDUAL CONSULTANT"].iloc[0].to_dict()
codes = classify_contract(placeholder, artefacts, TIMESTAMP)["reason_codes"]
print("placeholder-supplier record ->")
for c in codes:
    if "SUPPLIER" in c: print(f"   {c}")
""")

md("""
## 2. Portfolio-wide cohort assignment

`classify_contract` costs ~20 ms per record — fine for production (50 contracts/second, and contracts
arrive far slower than that), too slow for 288,237 rows in a notebook cell.

So the mix below is computed **stage by stage in batch**, then checked against the per-record path on
a sample. That batch-versus-single-record check is the same discipline used in every earlier part,
and it found four real defects along the way.
""")

code("""
# --- batch: rules ---------------------------------------------------------
outcomes = [apply_rules(validate_and_enrich(r, stats)) for r in features.to_dict("records")]
cohorts = pd.Series([o.cohort.name if o.cohort is not None else None for o in outcomes],
                    index=features.index, dtype="object")

# --- batch: model + anomaly on the deferred population --------------------
deferred = cohorts.isna()
sub = features[deferred]
scores = gbt.predict_proba(sub)
anom = detector.is_anomalous(sub)
model_cohort = np.where(scores >= gbt.threshold, "HIGH_ATTENTION", "ROUTINE")
model_cohort = np.where(anom, "HIGH_ATTENTION", model_cohort)   # the override
cohorts.loc[deferred] = model_cohort

mix = cohorts.value_counts().reindex([c.name for c in Cohort]).dropna().astype(int).to_frame("contracts")
mix["% of portfolio"] = (mix.contracts / len(features) * 100).round(2)
mix
""")

code("""
# --- agreement check: batch vs the per-record entry point ------------------
rng = np.random.default_rng(0)
idx = rng.choice(len(features), 600, replace=False)
with_context = features.to_dict("records")     # carries consortium_size
bare_raw     = rows                            # publisher's columns only

a = sum(classify_contract(with_context[i], artefacts, TIMESTAMP)["cohort"] != cohorts.iloc[i]
        for i in idx)
b = sum(classify_contract(bare_raw[i], artefacts, TIMESTAMP)["cohort"] != cohorts.iloc[i]
        for i in idx)
print(f"records supplied WITH contract-group context : {a} disagreements of {len(idx)}")
print(f"bare raw records (no consortium size)        : {b} disagreements of {len(idx)}")
""")

md("""
### 2.1 What a single record cannot know

That second number is not a defect, it is a boundary. **How many suppliers share a contract is a
property of the contract, not of one supplier row** — a single record cannot see its siblings, and
20,404 rows in this extract belong to joint ventures.

The first implementation quietly defaulted `consortium_size` to 1 when it was absent. That was
silently wrong for every joint venture, and it was this batch-versus-per-record check that exposed
it: the two paths disagreed on exactly those records. It now returns `None` with a
`CONSORTIUM_SIZE_UNKNOWN` flag, so a caller who cannot supply the context gets a conservative answer
and a note explaining why, rather than a confident wrong one.

Supply the context and the two paths agree exactly. That is the contract: **`classify_contract` is
exact when given what a contract genuinely carries, and honest about it when not.**
""")

md("""
## 3. Reproducibility

The same record, classified twice, must be byte-identical. And changing only the injected timestamp
must change only that field.
""")

code("""
rec = rows[1]
a = classify_contract(rec, artefacts, TIMESTAMP)
b = classify_contract(rec, artefacts, TIMESTAMP)
print("identical on repeat runs :", a == b)

other = classify_contract(rec, artefacts, datetime(2020, 1, 1, tzinfo=timezone.utc))
same_but_time = {k: v for k, v in a.items() if k != "classification_timestamp"} == \\
                {k: v for k, v in other.items() if k != "classification_timestamp"}
print("only the timestamp differs:", same_but_time)
print(f"  {a['classification_timestamp']}  vs  {other['classification_timestamp']}")
""")

md("""
### 3.1 What the snapshot has to carry

An audit record must reconstruct the decision *without the source dataset*. Features alone would let
you re-check the arithmetic but not tell you which contract it described — which is the first
question an auditor asks. So the snapshot carries the cleaned inputs too, including the benchmark
that was in force and the vintage month it came from.
""")

code("""
snap = a["feature_snapshot"]["normalized_inputs"]
for k in ("supplier_key", "borrower_country", "procurement_method", "signing_date",
          "benchmark_median", "benchmark_vintage_month", "reference_version"):
    print(f"  {k:<26} {snap.get(k)}")
""")

md("""
## 4. Safe default under failure

The brief: *"A record with missing data, an unavailable model result, or conflicting rule outputs
should default to high attention rather than routine."*

Asserting that is easy. Below it is **tested by breaking the model on purpose** — the only way to
know the fallback is a real branch rather than an accident of control flow.
""")

code("""
class BrokenModel:
    version, threshold = "broken", 0.5
    def predict_proba(self, df): raise RuntimeError("model service unavailable")

broken = PipelineArtefacts(stats=stats, model=BrokenModel(), detector=detector)
sample = rows[:600]
verdicts = pd.Series([classify_contract(r, broken, TIMESTAMP)["cohort"] for r in sample])
print("with the model deliberately broken:")
print(verdicts.value_counts().to_frame("contracts"))
print(f"\\nrecords that still reached ROUTINE: {int((verdicts=='ROUTINE').sum())}   <- must be zero")
""")

md("""
## 5. Limitations

- **Reproducible against *these* artefacts, not for all time.** The guarantee holds for a fixed set
  of versions, which is why all five are stamped on every record. Refit the benchmark vintages or
  retrain the model and the same input legitimately yields a different answer — the version stamp is
  what makes that visible rather than silent.
- **`ROUTINE` means "nothing objected", not "verified safe".** Every stage upstream measures
  conformity to patterns, and no stage has ever seen a real outcome.
- **The audit record cannot explain the rule engine's *absence* of a trigger** beyond the affirmative
  codes. A reviewer asking "why didn't rule X fire?" needs the rule catalogue in notebook 02.
- **Per-record cost is ~20 ms**, dominated by single-row DataFrame construction and per-call
  estimator overhead rather than by anything conceptual. Fine for contracts arriving in real time;
  a bulk re-scoring run would want the batch path used in section 2.
- **Contract-group context must be supplied, not inferred.** `consortium_size` cannot be derived
  from a single record. Absent it, the record is flagged and treated conservatively — correct, but it
  means a caller integrating this pipeline has to pass the whole contract, not one supplier line.
- **The timestamp is injected, so its accuracy is the caller's responsibility.** That is the right
  trade — it buys testable reproducibility — but a caller passing a wrong clock puts a wrong time on
  the record, and nothing here can detect that.
""")

nb["cells"]=C
nb.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
nbf.write(nb, "notebooks/05_cohort_assignment.ipynb")
print("notebook written:", len(C), "cells")
