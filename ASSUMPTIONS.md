# Assumptions

Every simplifying assumption this pipeline makes, why it was made, and what it would cost if wrong.

Two principles run through all of them:

- **Unknown is not zero.** Anything that can be unknowable is tri-state — `True` / `False` / `None`,
  or a value / `None` — and carries a flag saying why. `0` and `None` are different facts: a supplier
  with no track record and a supplier we cannot trace are not the same, and collapsing them asserts
  something false rather than merely losing information.
- **State it rather than hide it.** Where a choice could reasonably have gone another way, the
  alternative and its cost are recorded here, not just the decision.

---

## 1. Time

### 1.1 The signing date is the assessment moment
The brief asks for assessment *"at the time the contract was submitted"*. Submission precedes
signing — prior review happens before a contract is signed — so these are not the same moment.

**We anchor on the signing date, because this extract has no submission date.** It is the only
per-record time signal: `Fiscal Year` is exactly `year + (month >= 7)` of it, `Contract signed -
Calendar year` is exactly its year (both verified derived), and `As of Date` holds one value across
all 288,237 rows, so anchoring there would score every contract as of Aug 2026 — granting each record
years of its own future. The brief itself uses the signing date as the proxy, describing
days-into-fiscal-year as *"a proxy for submission timing within the fiscal cycle"*.

**Cost.** The anchor is late by the submission-to-signature interval, so the point-in-time guarantee
is mildly optimistic. Measured at a hypothetical 90-day lag:

| | Extra information granted | Consequence |
|---|---|---|
| Benchmarks | median 597 extra peer contracts | median moves only 1.2–1.8% per quarter — immaterial |
| Supplier history | ~1.2 extra prior contracts (mean) | small in aggregate, but can flip `is_first_contract_in_project` at the boundary, which feeds an EXCEPTIONAL rule |

`config.ASSESSMENT_LAG_DAYS` exists and is deliberately `0` — no lag value is supportable from this
data, and guessing one would trade a stated assumption for an invented one.

### 1.2 Publication lag is unmodelled
Every statistic filters on the *signing* date, but a contract signed earlier may have been
*published* later — so at the true moment of assessment some "prior" history would not yet have been
visible. **This is the only remaining look-ahead in the pipeline and it cannot be fixed from this
data**: there is no publication-date column. Point-in-time correctness is therefore an upper bound.

### 1.3 The World Bank fiscal year runs 1 July – 30 June
Recomputed from the signing date and cross-checked against the published column: zero disagreements
across all 288,237 rows. That is evidence the publisher *derives* fiscal year from the signing date,
which is why the "signed outside the fiscal year window" rule can never fire here.

---

## 2. Contract identity and grain

### 2.1 `WB Contract Number` identifies a contract
288,237 rows collapse to 276,417 contracts; 8,584 numbers appear more than once. For rows sharing a
number, project, signing date, description, borrower and procurement method are **100% identical** —
only supplier and amount vary. That is a joint venture, not a reused key.

### 2.2 Population statistics use contract grain, reporting uses row grain
Joint-venture rows repeat the **full** contract amount on each row, so raw-row totals reach $130.1B
against $116.6B at contract grain — a $13.5B overstatement. Every median, history count and project
position therefore runs through `cleaning.contract_grain()` first, while per-row output stays at row
grain because that is genuinely what the extract is.

### 2.3 The representative row for a contract is the alphabetically-first supplier
**This one is imperfect and worth knowing.** In 6,859 of 8,584 joint ventures every row carries the
identical full amount, so any representative is correct. But in **1,725 (20.1%)** the amounts differ —
presumably each firm's actual share — and taking one row understates those contracts.

Summing the shares instead would be wrong for the 79.9% that repeat the total, and **no field
distinguishes the two cases**. The choice is deterministic (alphabetical, so it is stable across
runs) and accepts a known understatement in a fifth of joint ventures rather than a known
overstatement in four fifths.

---

## 3. Fields and entities

### 3.1 `INDIVIDUAL CONSULTANT` is a placeholder, not a supplier
63,603 rows (22.1%) across 53,905 supplier IDs, appearing only on Individual Consultant Selection.
It is a category label standing in for a natural person whose name is withheld. Supplier history for
these returns `None`, never `0` — counting them as one vendor would give a fifth of the portfolio the
contract history of the most prolific supplier in the Bank.

**But the conclusion drawn from this went one step too far.** That the *name* is a placeholder does
not make the *supplier* unidentifiable, and `Supplier ID` was never re-examined. It should have been:
see §9.2.

### 3.2 The normalised supplier *name* is the entity key, not `Supplier ID`
**Both keys fail, in opposite directions — and the direction is what decides it.**

Excluding the placeholder there are 130,074 supplier IDs for 117,945 names, so the ID does group
repeat contracts: 26.2% of IDs appear on more than one contract, one of them on 728. It is a real
vendor identifier, not a per-award reference.

The problem is *fragmentation*. Keying on ID splits one firm across many IDs — 7.35% of names map to
more than one — while keying on name merges distinct firms sharing a name, which affects only 0.30%
of IDs. Fragmentation is **24× more common**, and it errs in the worse direction:

> **ERNST & YOUNG appears under 39 different supplier IDs. CFAO MOTORS under 34.** Keying on ID would
> treat Ernst & Young as 39 unrelated first-time suppliers — and since `supplier_prior_contract_count`
> feeds an EXCEPTIONAL rule, that turns an established firm into 39 apparent newcomers.

So the normalised name is the key, because merging a few same-named firms is less damaging than
fragmenting the large ones.

**An earlier version of this justified the choice badly**, citing 183,980 IDs against 121,914 names
as evidence the ID was unreliable. That gap is almost entirely the placeholder: strip it out and it
falls from ~62,000 to ~8,000. The decision was right; the argument for it was not.

For the placeholder population the ID is the *only* identifier available, and it works — see §9.2.

**Limitation:** matching is exact-after-normalisation. Genuine variants ("ACME LTD" vs "ACME COMPANY
LTD") remain distinct entities. Proper resolution would need fuzzy matching and beneficial-ownership
data.

### 3.3 Legal suffixes are stripped only from the end, and never to nothing
"Acme Trading Co., Ltd." → "ACME TRADING". Conservative by design: stripping inside a name would
mangle firms whose name genuinely contains such a token. Four real suppliers ("CIE SARL", "CIA SARL")
consist *entirely* of suffix tokens, so the stripper stops rather than erasing them.

### 3.4 The first-listed global practice is the primary one
48.2% of records list several separated by `;`. The field carries no ordering guarantee, so "first"
is a convention, not a claim about which practice dominates. Exploding to one row per practice would
break the one-row-per-award grain the whole pipeline depends on.

### 3.5 A missing borrower country code with a regional name means a multi-country programme
2,002 rows. For these, "is the supplier domestic?" has no answer, so the feature is `None` — not
`False`. Returning `False` would assert the supplier is foreign on the basis of a formatting gap, and
push the record toward higher risk for no reason.

### 3.6 Contract-group facts cannot be inferred from one row
`consortium_size` is a property of the contract; a single supplier row cannot see its siblings. When
not supplied it is `None` with a flag, never defaulted to 1 — that default was silently wrong for the
20,404 joint-venture rows.

---

## 4. Amounts and benchmarks

### 4.1 Amounts are comparable across time without inflation adjustment
The extract is already USD-converted at an undisclosed rate and date. Deflating would layer one
unverifiable assumption on another. Benchmark drift is handled structurally instead (§4.2), and
measured: the Works median more than halves across the window while Non-consulting Services swings
2.9×.

### 4.2 Benchmarks are monthly vintages, not daily
For each `(category, region, month)`, the median of everything signed strictly before that month.
Monthly because a real control function consumes a periodically published benchmark table, and
because it is the more conservative reading of point-in-time — a contract is never compared against
anything signed in its own month.

### 4.3 A peer group needs 30 prior contracts to be a benchmark
Below `MIN_GROUP_SUPPORT`, widen up a ladder (category+region → category → overall) and raise
`THIN_REFERENCE_GROUP`. Below even that, the benchmark is `None` — never a guess. **2,706 records
(all FY2020) have no benchmark at all**, being too early in the extract for any peer history to
exist. They are routed to a human rather than given a fabricated comparison.

### 4.4 Percentiles are stored as a 101-point sketch
Whole-percentile resolution. Finer thresholds are not representable — deliberately, since claiming
99.9th-percentile precision from a 30-observation peer group would be false precision.

---

## 5. Procurement method

### 5.1 The taxonomy is exhaustive, never inferred
All 18 observed methods map explicitly to `COMPETITIVE` / `NON_COMPETITIVE`. An unrecognised method
is a NOT_ELIGIBLE condition, so a keyword heuristic that always returned *something* would silently
disable that control.

### 5.2 Two classifications are judgment calls
- **Community Driven Development** (n=484) → non-competitive. Participatory rather than
  price-competitive, and no comparison of offers is evidenced. Deliberately conservative: these
  contracts are small, so a false elevation costs little.
- **Public Private Partnership** (n=10) → non-competitive. PPPs are usually competitively tendered,
  but the record does not evidence the tender and 100% of them sit above the 99th amount percentile.
  Classing them non-competitive routes every one to human attention.

### 5.3 Individual Consultant Selection is competitive
At least three CVs are compared under Bank rules. Recorded independently of the fact that its
supplier field is a placeholder, so a rule can act on either.

---

## 6. Scope and modelling

### 6.1 All records are modelled; review type is a feature, not a filter
The narrative concerns prior review (7.1% of rows), but a prior-review-only population leaves ~20k
records across eight fiscal years and ~200 in the most recent — too thin for a time-based split. A
prior-review slice is reported separately.

### 6.2 No sampling
288,237 rows is ~200 MB. Sampling was permitted but declined, which removes any interaction between
sampling bias and the time-based split.

### 6.3 The risk target is a definition, not an observation
**There is no realised outcome anywhere in this extract** — no fraud, dispute, cancellation or audit
finding. The target is defined by rule: top quartile of the peer group *and* non-competitive method.
Every metric therefore measures agreement with that definition, never with reality.

### 6.4 Features that reconstruct the label are withheld
The label's own inputs, plus `amount_vs_category_region_median`, `amount_usd`, `log_amount` (which
rebuild the percentile), `amount_vs_practice_median` (AUC 0.81 alone, correlation 0.83 with the peer
percentile) and `supplier_is_known` (forces the label to zero for 22.5% of records).

### 6.5 22.5% of the modelled population cannot be positive
Placeholder supplier → individual consultant selection → competitive → label zero. Not fixable by
feature selection, since the missingness of `supplier_prior_contract_count` carries the same signal.
Headline metrics are therefore reported on both populations.

### 6.6 FY2027 is quarantined from the test window
1,139 rows against a ~41,000 norm and 17.3% prior review against ~7%, because the extract was frozen
seven weeks into the year. Testing there would measure reporting lag, not model skill. Reported
separately rather than dropped.

### 6.7 Platt calibration, not isotonic
Isotonic calibrates marginally better (2.8pp vs 5.3pp maximum deviation) but collapses 51,783
distinct scores into 107 steps, flattening the threshold curve into unusable plateaus. A risk score
has to rank as well as calibrate.

---

## 7. Thresholds

Every threshold is a policy choice, not a discovery. All are calibrated on **reviewable volume**,
because nothing in this data can establish which contracts are genuinely risky.

| Threshold | Value | Basis |
|---|---|---|
| Amount extremity | 150× peer median | The brief's illustrative 5× flags 20.9% of the portfolio; 150× gives 1.80%, ~739/year. Configurable — the volume at every setting is in `config.py`. |
| Non-competitive high value | > $2M | Direct selection is lawful; it is the combination with scale that warrants a named reviewer. Absolute, because fiduciary exposure is absolute. |
| First-in-project | > 20× median | Lower bar than the standalone rule, because being first is itself evidence. |
| Model review threshold | 90% recall | The brief's asymmetry: missing a real one costs more than over-flagging. The price is flagging 40% of the portfolio at 7.3% precision, which is reported rather than hidden. |
| Anomaly contamination | 1% | Isolation Forest has no natural threshold; calibrated on reviewer capacity. |

---

## 8. Rules and outputs

### 8.1 Supplier-country risk is narrow by design
A blanket transparency-index rule would encode geography rather than conduct and is incoherent when
the borrowers are themselves developing economies. Of 1,360 contracts with suppliers in secrecy
jurisdictions, **962 are domestic** — Belize, Panama and the Marshall Islands are borrowers in their
own right. Only the 329 that are offshore *and* foreign are flagged.

The jurisdiction list is assembled from the TJN Financial Secrecy Index (2022) and the EU
non-cooperative list. **A production deployment should replace it with an official list on a
maintained refresh cycle**; the vintage is stamped so the artefact can be aged.

### 8.2 Anomaly explanations are templated, never generated
A procurement decision must reproduce at an audit years later, traceable to the values that caused
it. A generated sentence varying between runs would break the audit record and buy nothing in
accuracy.

### 8.3 No reason code claims a prior contract was "clean"
The brief's example output includes `SUPPLIER_HAS_PRIOR_CLEAN_CONTRACTS`. We emit
`SUPPLIER_HAS_PRIOR_CONTRACTS`. Nothing in this extract establishes that any contract was clean —
only that contracts existed. Putting "clean" in an audit record asserts what the evidence cannot
support.

### 8.4 `data_quality_flag` is true only for FATAL or DEGRADED
NOTICE-level flags are recorded but impair nothing, and `MULTI_PRACTICE_PROJECT` alone fires on 48%
of the portfolio — including them would make the field mean "almost always".

### 8.5 The classification timestamp is supplied by the caller
Never read from a clock, because a function calling `datetime.now()` cannot be tested for
reproducibility. The trade-off: a caller passing a wrong clock puts a wrong time on the record, and
nothing here can detect that.

---

## 9. Known defects

Distinct from the sections above. Those are **choices** — decisions that could reasonably have gone
another way. These are **errors**: places where the pipeline does something demonstrably wrong. Both
were found late, by questioning a framing that had gone unchallenged for most of the build.

### 9.1 Joint-venture partners lose credit for contracts they won

`cleaning.contract_grain()` collapses each contract to a single row. That is **correct for amounts** —
joint-venture rows repeat the full contract value, so without it totals overstate the portfolio by
$13.5B. It is **wrong for supplier history**, where every partner genuinely won that contract.

Because the history store is built from the collapsed table, only the alphabetically-first partner is
credited:

| | |
|---|---|
| Partner rows receiving no credit | **11,086** |
| Distinct suppliers affected | **7,743** |
| Worst-affected supplier | loses **25** contracts from its history |
| Suppliers losing 5 or more | **184** |

Against a median supplier history of 1 contract, losing 25 is severe: a firm that partners frequently
is presented to the pipeline as a stranger. That understates `supplier_prior_contract_count`, can
wrongly set `is_first_contract_in_project`, and both feed rules and the model.

**The cause is a single tool serving two purposes.** Deduplication is required for money and
forbidden for history; one collapsed table cannot do both. The fix is two grains — contract grain for
value statistics, row grain for participation — not a change to either rule.

### 9.2 Individual consultants are treated as less identifiable than they are

The supplier *name* is a placeholder for 63,603 rows, which is why `supplier_key` is `None` for them.
But `Supplier ID` was not re-examined, and it carries real signal:

| | |
|---|---|
| Individuals with more than one contract | **6,474** |
| Rows with recoverable history | **16,172** — a quarter of the placeholder population |
| Of those individuals, working across >1 project | **39.8%** (537 across >1 borrower country) |

Working across multiple projects and countries means the ID is **portfolio-level identity**, not a
per-award reference. So for a quarter of these records the pipeline reports "unknowable" when the data
could give a real number.

This is conservative in the right direction — it never invents history — but it is still an
overstatement of ignorance, and the tri-state discipline is supposed to cut both ways: `None` should
mean *genuinely* unknown, not *unexamined*.

**The fix** is to key supplier history on the resolved name where it exists and fall back to
`Supplier ID` for placeholder rows, leaving `None` only for the ~75% with no repeat history.

### 9.3 Why neither is fixed here

Both change `supplier_prior_contract_count`, which moves features, the model, its thresholds, the
cohort mix, and every figure quoted across the README and five notebooks. They are recorded rather
than patched so the record is accurate about what this pipeline currently does, not what it should do.

---

## 10. What would most change these conclusions

1. **Outcome labels.** Even a small hand-audited sample would turn the target from a definition into
   something testable, and would show whether any of this correlates with real risk.
2. **A submission or publication date.** Either would close the remaining look-ahead (§1.1, §1.2).
3. **A field distinguishing joint-venture shares from repeated totals**, which would resolve §2.3.
4. **Beneficial ownership data**, which would replace the jurisdiction proxy in §8.1 with something
   about conduct rather than geography.
