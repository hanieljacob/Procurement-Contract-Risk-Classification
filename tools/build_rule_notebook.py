import nbformat as nbf

nb = nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Rule-Based Cohort Classification

Every record passes through this deterministic engine **before** any model runs. The engine either
**decides** a record or **defers** it:

| Outcome | Meaning |
|---|---|
| `NOT_ELIGIBLE` | A required field is missing or unusable — the record cannot be scored at all |
| `EXCEPTIONAL` | A mandatory control fired — route to a senior reviewer with the reason |
| `HIGH_ATTENTION` | A rule could not be *evaluated* because a feature it depends on is unknown |
| *(defer)* | Every rule was evaluable and none fired — pass to the model |

Deferring is the important one. A record that clears every control is **not thereby ROUTINE** — it is
merely not exceptional. `ROUTINE` is a conclusion only the model gets to draw, so `apply_rules`
returns `cohort=None` rather than pretending to a verdict it has not earned.

### Why these are rules and not model features

Each condition is a **policy commitment** that must hold regardless of what the data supports. A
model trained on a portfolio where large direct selections are common would learn that they are
unremarkable — exactly backwards for a control. Rules also have to be explainable to the person whose
contract was stopped, and stable enough that changing a threshold is a documented decision rather
than a retraining artefact.
""")

code("""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from collections import Counter
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 80)

from procurement_risk import config
from procurement_risk.loading import load_raw
from procurement_risk.cleaning import clean_frame
from procurement_risk.features import build_reference_stats, engineer_features
from procurement_risk.pipeline import validate_and_enrich
from procurement_risk.rules import apply_rules, rule_catalogue, Cohort, EXCEPTIONAL_RULES

clean = clean_frame(load_raw())
stats = build_reference_stats(clean)
features = engineer_features(clean, stats)
print(f"{len(features):,} records, {len(stats.category_region):,} benchmark vintages")
""")

md("""
## 1. The documented rule table

The brief requires each rule to state the condition it checks, the threshold used, and the reason it
is treated as a hard classification rather than a model prediction.
""")

code("""
cat = pd.DataFrame(rule_catalogue())
for _, r in cat.iterrows():
    print(f"{'='*100}\\n{r.rule_id}   [{r.status}]  ->  {r.cohort}")
    print(f"  CONDITION : {r.condition}")
    print(f"  THRESHOLD : {r.threshold}")
    print(f"  RATIONALE : {r.rationale}")
    print(f"  WHY A RULE: {r.why_hard_rule}")
""")

md("""
## 2. `NOT_ELIGIBLE` needs no new logic

The brief lists the not-eligible conditions as: missing supplier, borrower country, procurement
category or amount; zero or negative amount; missing or unparseable signing date; and an unmappable
procurement method.

Those are **exactly** the FATAL data-quality flags the validator already raises. Restating them here
as separate predicates would create a second definition free to drift from the one actually enforced,
so the engine reads `record.ok` and reports the flags as reason codes.
""")

code("""
from procurement_risk.quality import DataQualityFlag as F, SEVERITY, Severity
fatal = [f.value for f in F if SEVERITY[f] is Severity.FATAL]
print(f"{len(fatal)} FATAL flags serve as the NOT_ELIGIBLE rules:")
for f in fatal: print("   ", f)
""")

md("""
## 3. Calibration — defining the multiple

The brief asks for *"contract amount above **a defined multiple** of the regional and category
median, for example more than five times the median."* The requirement is a defined multiple; five is
the illustration. So the rule takes exactly that form, and the multiple is the parameter
(`config.EXCEPTIONAL_AMOUNT_MEDIAN_MULTIPLE`).

Defining it is the judgment work, and the table below is the evidence. **Five would flag 20.9% of the
portfolio** — roughly 8,585 contracts a year routed to senior review. That is not a priority queue,
it is a second inbox. The cause is the shape of the distribution rather than anything wrong with the
rule: amounts are heavy-tailed enough that 5× the median sits at only the **78th percentile**.

Defined at **150×**, the control yields ~739 contracts a year, which a senior reviewer or control
function could actually work through. Change the constant and the volume moves with it — the table is
the record of what each setting costs.
""")

code("""
r = features.amount_vs_category_region_median
p = features.amount_percentile_in_category_region
n = len(features)

nc = features.is_competitive_method == False
first = features.is_first_contract_in_project == True
off = (features.supplier_in_secrecy_jurisdiction == True) & (features.supplier_is_domestic == False)
other = (nc & (features.amount_usd > config.EXCEPTIONAL_NON_COMPETITIVE_AMOUNT)) | off \
        | (first & (r > config.EXCEPTIONAL_FIRST_CONTRACT_RATIO))

rows = []
for m in (5, 10, 25, 50, 75, 100, 150, 250):
    union = (r > m) | other
    note = "<- the brief's example" if m == 5 else (
           "<- adopted" if m == config.EXCEPTIONAL_AMOUNT_MEDIAN_MULTIPLE else "")
    rows.append({"multiple": f"{m}x median", "this rule alone": int((r > m).sum()),
                 "TOTAL exceptional": int(union.sum()),
                 "% of portfolio": round(union.mean()*100, 2),
                 "per year": int(union.sum()/7), "note": note})
pd.DataFrame(rows)
""")

md("""
### 3.1 Threshold selection for the other rules

Each threshold is set on the volume it produces, targeting a total `EXCEPTIONAL` share of 1–2% —
roughly 680 contracts a year, which a senior reviewer or control function could actually action.
""")

code("""
nc = features.is_competitive_method == False
first = features.is_first_contract_in_project == True
grid = []
for t in (250_000, 500_000, 1_000_000, 2_000_000, 5_000_000):
    k = int((nc & (features.amount_usd > t)).sum())
    grid.append({"rule": "non-competitive & amount >", "threshold": f"${t:,}",
                 "flagged": k, "% of portfolio": round(k/n*100, 2),
                 "note": "<- adopted" if t == config.EXCEPTIONAL_NON_COMPETITIVE_AMOUNT else ""})
for m in (5, 10, 20, 50):
    k = int((first & (r > m)).sum())
    grid.append({"rule": "first in project & ratio >", "threshold": f"{m}x",
                 "flagged": k, "% of portfolio": round(k/n*100, 2),
                 "note": "<- adopted" if m == config.EXCEPTIONAL_FIRST_CONTRACT_RATIO else ""})
pd.DataFrame(grid)
""")

md("""
### 3.2 Supplier-country risk — a deliberately narrow rule

The brief offers *"supplier country is flagged as high risk based on publicly available transparency
indices."* Taken at face value on this dataset that is incoherent: the **borrowers** are themselves
overwhelmingly developing economies, so a transparency-index cutoff would flag enormous volumes and
would encode geography rather than conduct — effectively penalising poverty.

The measurement below is what settled the design. 1,360 contracts have suppliers registered in
secrecy jurisdictions, but **most are domestic** — Belize, Panama and the Marshall Islands are
borrowers in their own right, and a domestic supplier in its own country is an ordinary award. Only
the intersection with *foreign to the borrower* isolates the pattern actually worth a reviewer's
time: project value leaving through a vehicle whose ownership cannot be established.
""")

code("""
off = features.supplier_in_secrecy_jurisdiction == True
print(f"suppliers registered in a secrecy jurisdiction : {int(off.sum()):>6,}")
print(f"  ...of which DOMESTIC (borrower is that country): {int((off & (features.supplier_is_domestic==True)).sum()):>5,}")
print(f"  ...of which FOREIGN to the borrower            : {int((off & (features.supplier_is_domestic==False)).sum()):>5,}  <- the rule")
print()
print(features.loc[off, "supplier_country"].value_counts().head(8).to_frame("contracts"))
""")

md("""
## 4. Three-valued logic — and a bug it caught

A predicate returns `True` (fired), `False` (did not fire), or `None` (**could not be evaluated**,
because a feature it depends on is unknown). That third outcome is the whole point: a rule we could
not evaluate is not a rule that passed.

The first implementation treated *any* unknown operand as poisoning the whole conjunction. That sent
**9,666 records to a human because a country code was missing** — on the offshore rule, which could
not have fired anyway, since those suppliers were demonstrably *not* registered in a secrecy
jurisdiction.

The fix is Kleene's three-valued AND: **`False AND unknown` is `False`**, not unknown. Being
conservative means resolving genuine ambiguity upward. It does not mean manufacturing ambiguity that
the data has already settled. Safe-default volume fell from 4.21% to 0.96%, and every one of those
0.96% is now a record with a genuinely missing benchmark.
""")

code("""
from procurement_risk.rules import _kleene_and
tt = pd.DataFrame([{"A": str(a), "B": str(b), "A AND B": str(_kleene_and(a,b))}
                   for a in (True, False, None) for b in (True, False, None)])
tt.pivot(index="A", columns="B", values="A AND B")
""")

md("""
## 5. Running the engine
""")

code("""
outcomes = [apply_rules(validate_and_enrich(rec, stats)) for rec in features.to_dict("records")]

labels = [str(o.cohort) if o.cohort is not None else "-> MODEL" for o in outcomes]
mix = pd.Series(labels).value_counts().to_frame("contracts")
mix["% of portfolio"] = (mix.contracts / len(features) * 100).round(2)
mix
""")

code("""
fired = Counter(rid for o in outcomes for rid in o.triggered)
undec = Counter(rid for o in outcomes for rid in o.undecidable)
pd.DataFrame([{"rule": rl.id, "status": "active" if rl.active else "INACTIVE",
               "fired": fired.get(rl.id, 0), "could not evaluate": undec.get(rl.id, 0)}
              for rl in EXCEPTIONAL_RULES])
""")

md("""
### 5.1 The rule that cannot fire

`SIGNED_OUTSIDE_FISCAL_YEAR_WINDOW` fires **zero** times, and that is not a tuning problem — it is
structural. The publisher derives `Fiscal Year` from the signing date, so no record in this extract
can fall outside its own window.

It is retained as an input guard for unvalidated upstream data and reported as **a control with no
coverage here**, because deleting it would hide the fact that the check does no work on this source.
A control that cannot trigger is worth as much attention as one that triggers often.
""")

md("""
### 5.2 Stability over time

A threshold calibrated once on the whole extract could still be wildly uneven year to year. It is
not: `EXCEPTIONAL` holds between 1.4% and 2.2% in every complete fiscal year. FY2020 runs slightly
high because it also contains every warm-up record, and FY2027 is the truncated seven-week stub.
""")

code("""
tmp = features.assign(outcome=labels)
by_fy = pd.crosstab(tmp.fiscal_year, tmp.outcome, normalize="index").mul(100).round(2)
by_fy["contracts"] = tmp.groupby("fiscal_year").size()
by_fy
""")

md("""
## 6. Worked examples

One record per outcome, showing what the reviewer would receive.
""")

code("""
def show(title, rec):
    e = validate_and_enrich(rec, stats)
    o = apply_rules(e)
    print(f"=== {title} ===")
    print(f"  cohort      : {o.cohort if o.cohort is not None else 'proceeds to model'}")
    print(f"  triggered   : {o.triggered or '-'}")
    print(f"  reason codes: {o.reason_codes or '-'}")
    print(f"  undecidable : {o.undecidable or '-'}")
    if e.features:
        print(f"  amount ${e.features['amount_usd']:,.0f} | peer pct "
              f"{e.features['amount_percentile_in_category_region']} | competitive "
              f"{e.features['is_competitive_method']}")
    print()

idx = {lab: i for i, lab in enumerate(labels)}
show("NOT_ELIGIBLE", features.iloc[labels.index("NOT_ELIGIBLE")].to_dict())
show("EXCEPTIONAL", features.iloc[labels.index("EXCEPTIONAL")].to_dict())
show("HIGH_ATTENTION (safe default)", features.iloc[labels.index("HIGH_ATTENTION")].to_dict())
show("Deferred to the model", features.iloc[labels.index("-> MODEL")].to_dict())
""")

md("""
## 7. What the model inherits

| | contracts | share |
|---|---|---|
| Decided here | 7,953 | 2.76% |
| Passed to the model | 280,284 | 97.24% |

The model never sees a record that is unscoreable, that tripped a mandatory control, or whose
controls could not be evaluated. It is handed a population where every rule was evaluable and none
fired — which is what makes a `ROUTINE` verdict meaningful rather than a default.

**Carried forward to the model stage:**

- The `Cohort` enum and its precedence ordering, so `min()` resolves conflicts.
- `reason_codes` in the shared `DataQualityFlag` / rule-id vocabulary, ready for the audit record.
- The safe-default posture: anything the model cannot score must land in `HIGH_ATTENTION`, never
  `ROUTINE`. The rule engine establishes the pattern; the model stage has to preserve it.

**Limitation worth stating.** These thresholds are calibrated on *volume*, not on outcomes — this
extract contains no realised fraud, dispute or cancellation label, so there is no way to measure
whether the 5,178 contracts flagged are the *right* 5,178. What the calibration guarantees is that
the queue is actionable and the reasoning is explicit; it cannot guarantee precision, and no
threshold chosen from this data could.
""")

nb["cells"]=C
nb.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
nbf.write(nb, "notebooks/02_rule_engine.ipynb")
print("notebook written:", len(C), "cells")
