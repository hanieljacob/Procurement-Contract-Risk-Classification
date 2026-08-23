import nbformat as nbf
nb = nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Risk Classification Model

Scores the 280,284 contracts the rule engine defers — everything not already unscoreable,
exceptional, or escalated because a control could not be evaluated.

## The label is defined, not observed

This extract contains **no realised outcome** — no fraud, dispute, cancellation or audit finding
anywhere in its 21 columns. So there is nothing to predict in the ordinary sense, and the brief
supplies a label by rule instead:

> a contract is **high attention** when its amount is in the top quartile of contracts within the
> same procurement category and region, **and** the procurement method is non-competitive.

Defining a proxy label this way is the normal thing to do when none exists. Two consequences follow,
and both belong in any honest reading of the numbers below:

1. **Any score measures agreement with a definition, never with reality.** The definition encodes an
   assumption — that large and non-competitive means risky — which nothing in this data verifies.
2. **Both halves of the definition are columns we already hold**, so a model handed those columns
   scores a perfect AUC. That is target leakage, and the fix is to withhold them.
""")

code("""
import sys, warnings, pickle
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option("display.width", 210); pd.set_option("display.max_columns", 40)

from procurement_risk import config, model as M

eligible = pd.read_parquet(Path.cwd().parent / "data" / "model_eligible.parquet")
splits = M.split_by_fiscal_year(eligible)
y_all = M.build_target(eligible)
print(f"model-eligible contracts : {len(eligible):,}")
print(f"high attention (label=1) : {int(y_all.sum()):,}  ({y_all.mean()*100:.2f}%)")
""")

md("""
## 1. Why this definition is a reasonable one

The brief asks for the definition to be justified as well as criticised, and it deserves the
justification. Absent any outcome data, it is a well-chosen proxy:

**It isolates exposure that no market test has checked.** The two conditions together pick out
contracts where a large sum was committed on a decision no competitor challenged. That is precisely
the combination a procurement control function worries about — concentrated spend with no
independent price discovery behind it.

**Neither condition alone would work, and the conjunction is the insight.** Large-but-competitive is
ordinary: major infrastructure is *supposed* to be expensive, and competition supplies the price
check. Small-but-non-competitive is also ordinary and cheap to get wrong: direct-selecting a $5,000
consultant is routine and the exposure is trivial. Only together do they describe something worth a
senior reviewer's time.

**It is peer-relative rather than absolute.** "Top quartile within the same category and region"
compares a Latin American consultancy against Latin American consultancies, not against South Asian
road works. An absolute dollar threshold would simply relabel the definition "is this a Works
contract", which carries no information a reviewer does not already have.

**It is observable when the decision has to be made.** Both halves are known at signing — and, being
properties of the award itself rather than of its consequences, at submission too. A label that
required waiting for an outcome would be useless for prior review, whatever its statistical merits.

**It is auditable.** A borrower can be told exactly why their contract was routed for closer review,
in one sentence, with the threshold named. That matters more for a fiduciary control than a marginal
gain in predictive accuracy would.

It also mirrors how the Bank already works: prior-review thresholds are themselves a function of
contract value and procurement method, so the definition encodes existing policy rather than
inventing a new theory of risk.

**What it is not** is a measure of wrongdoing. It flags *structural* exposure, not misconduct — most
of the contracts it selects will be perfectly proper, and some genuinely problematic contracts will
be small, competitive, and invisible to it. Section 7 sets out the limitations in full.
""")

md("""
## 2. Leakage — measured, not asserted

Train with the label's own inputs present and both models return a **perfect AUC of 1.0000** on
held-out years. That is not skill. It is the definition being read back.
""")

code("""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
tr, va = splits["train"], splits["validate"]
ytr, yva = M.build_target(tr), M.build_target(va)

def quick_auc(cols):
    Xtr = M.build_design_matrix(tr, cols)
    Xva = M.align_columns(M.build_design_matrix(va, cols), Xtr.columns)
    g = HistGradientBoostingClassifier(max_iter=60, random_state=0).fit(Xtr, ytr)
    return roc_auc_score(yva, g.predict_proba(Xva)[:, 1])

print(f"WITH the label's own inputs : AUC {quick_auc(M.feature_columns(True)):.4f}   <- tautology")
print(f"WITHOUT them                : AUC {quick_auc(M.feature_columns()):.4f}   <- the model that ships")
print("\\nwithheld:", list(M.LABEL_DEFINING_FEATURES))
""")

md("""
### 2.1 Two leaks that were not obvious

Removing the two columns the brief names is not sufficient. Two more had to be found by measurement.

**`amount_vs_practice_median`** is not named by the label, but it is the amount measured against a
*different* peer grouping. On its own it scores **AUC 0.81** and correlates **0.83** with the
category-and-region percentile — the amount in disguise. Any amount-relative-to-a-benchmark feature
reconstructs the label's first condition.

**`supplier_is_known`** is worse, because it is deterministic. The placeholder supplier
(`INDIVIDUAL CONSULTANT`) appears *only* on Individual Consultant Selection, which is a competitive
method — so the label's second condition fails outright and those records are guaranteed negatives.
That is **62,992 contracts, 22.5% of the population**, whose label is decided before the model looks
at anything.

Dropping the flag does not remove the signal: `supplier_prior_contract_count` is missing on exactly
those rows, and a tree can split on missingness. This is not something feature selection can fix —
it is a property of the label definition meeting a property of the data. It matters for **reporting**:
free negatives inflate every aggregate metric, so headline figures are quoted on both populations.
""")

code("""
forced = M.structurally_negative(eligible)
print(f"records that CANNOT be positive : {int(forced.sum()):,}  ({forced.mean()*100:.1f}%)")
print(f"  their label rate               : {float(y_all[forced].mean()):.4f}  (zero, by construction)")
print(f"  their procurement method       : {eligible.loc[forced,'procurement_method_raw'].unique().tolist()}")
""")

md("""
## 3. Training

Both models the brief requires. Fitted on FY2020–22, probabilities calibrated on FY2023, tested once
on FY2024–26. FY2027 is quarantined — a seven-week stub where testing would measure reporting lag
rather than skill.

Calibration uses **Platt scaling rather than isotonic**. Isotonic calibrates marginally better here
(2.8pp vs 5.3pp maximum deviation) but collapses 51,783 distinct scores into 107 steps, which
flattens the threshold curve into unusable plateaus. A risk score has to rank as well as calibrate.
""")

code("""
for name, part in splits.items():
    yy = M.build_target(part)
    print(f"  {name:<12} FY{sorted(part.fiscal_year.unique())}  n={len(part):>7,}  "
          f"positives={int(yy.sum()):>5,} ({yy.mean()*100:.2f}%)")
""")

code("""
gbt = M.train(splits["train"], splits["validate"], kind="gbt")
lr  = M.train(splits["train"], splits["validate"], kind="logistic")
for m in (gbt, lr):
    p = m.predict_proba(splits["validate"])
    m.threshold  = M.choose_threshold(yva, p, min_recall=0.90)
    m.band_edges = M.choose_band_edges(p)
gbt = M.attach_explainer(gbt, splits["train"].head(2000))
print(f"features {len(M.feature_columns())} -> design columns {len(gbt.columns)}")
print("features:", M.feature_columns())
""")

md("""
## 4. Evaluation — the three measures the brief names

**Calibration**, **precision at the top decile**, and **the share of flagged contracts that fall in
the high-attention group**. Reported for both models, on validation and test, with the quarantined
FY2027 stub shown separately and the "can be positive" population alongside — because the free 22.5%
of guaranteed negatives inflates every aggregate figure.
""")

code("""
rows = []
for nm, m in (("GBT", gbt), ("LogReg", lr)):
    for s in ("validate", "test", "quarantined"):
        r = M.evaluate(m, splits[s]); r.update(model=nm, split=s); rows.append(r)
    sub = splits["test"][~M.structurally_negative(splits["test"])]
    r = M.evaluate(m, sub); r.update(model=nm, split="test (can be positive)"); rows.append(r)
pd.DataFrame(rows)[["model","split","n","base_rate","auc","brier",
                    "precision_at_top_decile","share_flagged_in_target","flagged_share_of_population"]]
""")

md("""
Reading this: the gradient-boosted model reaches **AUC 0.845 on test**, falling to **0.799** on the
population that can actually be positive — the gap is the free negatives. Precision in the top decile
is **0.150 against a 0.033 base rate**, a **4.5× lift**, which is the most defensible single statement
about what the model is worth. Logistic regression trails it consistently but not enormously.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for nm, m in (("GBT", gbt), ("LogReg", lr)):
    yte = M.build_target(splits["test"]); p = m.predict_proba(splits["test"])
    t = M.calibration_table(yte, p); t = t[t.n > 200]
    axes[0].plot(t.mean_predicted, t.observed_rate, "o-", label=f"{nm}")
    tbl = M.threshold_table(yte, p)
    axes[1].plot(tbl["flagged_%"], tbl.recall, "o-", label=nm)
axes[0].plot([0, .4], [0, .4], "k--", lw=.8, label="perfect")
axes[0].set(xlabel="mean predicted probability", ylabel="observed rate", title="Calibration (test years)")
axes[0].legend(frameon=False)
axes[1].axhline(0.90, color="#C44E52", ls="--", lw=.8, label="90% recall floor")
axes[1].set(xlabel="% of portfolio flagged", ylabel="recall", title="What recall costs in reviewer volume")
axes[1].legend(frameon=False)
plt.tight_layout(); plt.show()
""")

md("""
## 5. Threshold — chosen conservatively, and what that costs

The brief is explicit: *missing a genuinely high-attention contract carries more risk than
over-flagging a routine one.* So **recall is the constraint and reviewer volume is the price**. The
threshold is the tightest cut-off on the **validation year alone** that still catches 90% of
high-attention contracts, then applied unchanged to test.

The table below is the justification. It also carries an uncomfortable finding worth stating plainly:
**at 90% recall the model flags 40% of the validation year at 7.3% precision** (35% of the test
years). That is what conservatism costs with a model this strong, and it is the evidence for the
conclusion in section 6 — this model should not be a standalone gate.
""")

code("""
M.threshold_table(yva, gbt.predict_proba(splits["validate"]))
""")

code("""
print(f"chosen threshold (GBT) : {gbt.threshold:.6f}   [tightest cut-off with >=90% recall on FY2023]")
print(f"risk bands             : LOW < {gbt.band_edges[0]:.4f} <= MEDIUM < {gbt.band_edges[1]:.4f} <= HIGH")
te = splits["test"]; p = gbt.predict_proba(te)
bands = pd.Series([gbt.band(s) for s in p]).value_counts()
out = pd.DataFrame({"contracts": bands})
out["% of test"] = (out.contracts / len(te) * 100).round(2)
out["high-attention rate"] = [float(M.build_target(te)[[gbt.band(s)==b for s in p]].mean()) for b in out.index]
out.round(4)
""")

md("""
## 6. Per-record output

Risk score, band, and the three features moving it most — via SHAP on the tree model, so the
explanation comes from the model doing the scoring rather than a stand-in.
""")

code("""
import json
scores = gbt.predict_proba(te)
for label, rec in (("HIGHEST-SCORING CONTRACT", te.iloc[[int(scores.argmax())]]),
                   ("A MID-BAND CONTRACT",      te.iloc[[int(np.argsort(scores)[len(scores)//2])]])):
    print(f"=== {label} ===")
    print(json.dumps(M.score_record(gbt, rec), indent=2)); print()
""")

md("""
## 7. What this model is actually for — and its limitations

**It is not a predictor of risk.** It is a predictor of a definition, and the definition is an
assumption. Every number above measures agreement with that assumption.

**The rule engine already does the deterministic part.** Since the label is fully computable from two
columns, a production system would simply apply the rule rather than model it. What the model adds is
a **graded contextual score** over the records the rules clear — useful for ranking the routine
population for periodic sampling, and for records where the peer-group percentile is unknown.

**It should not be a standalone gate.** At the conservative threshold the brief asks for, it flags
40% of the portfolio to reach 90% recall. The band is the more useful output: `LOW` / `MEDIUM` /
`HIGH` carry high-attention rates of **1.1% / 6.2% / 15.1%** on the test years — a 13x gradient that
supports prioritising a review queue, which a binary flag at this precision does not.

### Stated limitations

- **No ground truth exists.** No fraud, dispute or cancellation label anywhere in the extract. The
  target is a definition; performance against it cannot establish real-world precision.
- **22.5% of the population is a guaranteed negative** by construction, and it is not a random 22.5%
  — it is entirely individual-consultant awards. Aggregate metrics are inflated accordingly, which is
  why both populations are reported.
- **The label inherits every Part 1 caveat**, including that the peer-group percentile is unavailable
  for 2,706 warm-up records, which the rule engine routes to a human rather than the model.
- **Performance degrades across time** — AUC 0.856 on FY2023, 0.845 on FY2024–26 — consistent with a
  portfolio that shifts. A deployed version would need scheduled retraining and drift monitoring.
- **Logistic regression is not a fair baseline for missingness.** It cannot take NaN, so it receives
  median imputation plus a missingness indicator, while the tree model uses NaN natively. Part of the
  gap between them is that handling, not model capacity.
""")

nb["cells"]=C
nb.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
nbf.write(nb, "notebooks/03_risk_model.ipynb")
print("notebook written:", len(C), "cells")
