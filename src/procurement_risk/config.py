"""Frozen configuration for the procurement risk pipeline.

Everything here is a decision, not a derived value. Each constant carries the
reasoning behind it. Nothing in this module reads the dataset, that keeps the pipeline
reproducible for a fixed model version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Paths & versioning
# --------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
RAW_CSV: Final[Path] = (
    DATA_DIR
    / "contract_awards_in_investment_project_financing_since_fy_2020_08-22-2026.csv"
)
PARQUET_CACHE: Final[Path] = DATA_DIR / "contracts_raw.parquet"

# Bumped whenever cleaning rules, feature definitions, or thresholds change.
# The audit record pins this so a decision can be reconstructed later.
PIPELINE_VERSION: Final[str] = "v1.0"

# The dataset is a point-in-time extract. Its "As of Date" is the moment the
# publisher froze it; we treat it as the ceiling of observable information.
DATA_AS_OF_DATE: Final[str] = "2026-08-22"

# --------------------------------------------------------------------------
# World Bank fiscal calendar
# --------------------------------------------------------------------------
# WB fiscal year N runs 1 July (N-1) .. 30 June (N). Verified against the data:
# all 288,237 rows fall inside their published FY window, which tells us the
# publisher *derives* Fiscal Year from the signing date rather than recording it
# independently. We still recompute and cross-check rather than trust it.

FY_START_MONTH: Final[int] = 7
FY_START_DAY: Final[int] = 1

# --------------------------------------------------------------------------
# Source column names -> internal snake_case names
# --------------------------------------------------------------------------

COLUMN_RENAMES: Final[dict[str, str]] = {
    "As of Date": "as_of_date",
    "Fiscal Year": "fiscal_year_published",
    "Region": "region",
    "Borrower Country / Economy": "borrower_country",
    "Borrower Country / Economy Code": "borrower_country_code",
    "Project ID": "project_id",
    "Project Name": "project_name",
    "Project Global Practice": "global_practice_raw",
    "Procurement Category": "procurement_category",
    "Procurement Method": "procurement_method_raw",
    "WB Contract Number": "contract_id",
    "Contract Description": "contract_description",
    "Borrower Contract Reference Number": "borrower_reference",
    "Contract Signing Date": "signing_date_raw",
    "Supplier ID": "supplier_id",
    "Supplier": "supplier_name_raw",
    "Supplier Country / Economy": "supplier_country",
    "Supplier Country / Economy Code": "supplier_country_code",
    "Supplier Contract Amount (USD)": "amount_usd",
    "Review type": "review_type",
    "Contract signed - Calendar year": "signed_calendar_year",
}

SIGNING_DATE_FORMAT: Final[str] = "%m/%d/%Y"

# Fields without which a record cannot be responsibly assessed. A problem here
# fails the record (the rule engine maps this to NOT_ELIGIBLE); we never impute these.
REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "supplier_name_raw",
    "borrower_country",
    "procurement_category",
    "amount_usd",
    "signing_date_raw",
    "procurement_method_raw",
)

# --------------------------------------------------------------------------
# Procurement method taxonomy
# --------------------------------------------------------------------------
# Why an explicit exhaustive map instead of a keyword heuristic: "unmappable
# procurement method" is itself a NOT_ELIGIBLE rule. A regex that
# silently classifies an unfamiliar method would defeat that rule -- the whole
# point is that a value we have never seen must surface, not be guessed at.
#
# The competitive/non-competitive split follows the World Bank Procurement
# Regulations for IPF Borrowers: a method is competitive when price or quality
# is tested against more than one bidder.

COMPETITIVE: Final[str] = "COMPETITIVE"
NON_COMPETITIVE: Final[str] = "NON_COMPETITIVE"

PROCUREMENT_METHOD_TAXONOMY: Final[dict[str, tuple[str, str]]] = {
    # value -> (class, rationale)
    "Request for Bids": (
        COMPETITIVE,
        "Open advertised bidding; the Bank's default competitive method for Goods/Works.",
    ),
    "Request for Quotations": (
        COMPETITIVE,
        "Shopping against multiple quotations; competitive though lightweight and low-value.",
    ),
    "Request for Proposals": (
        COMPETITIVE,
        "Competitive proposals evaluated on technical merit and price.",
    ),
    "Quality And Cost-Based Selection": (
        COMPETITIVE,
        "QCBS: shortlisted firms compete on combined quality and cost score.",
    ),
    "Quality Based Selection": (
        COMPETITIVE,
        "QBS: shortlist competes on quality; cost negotiated after ranking, still multi-firm.",
    ),
    "Least Cost Selection": (
        COMPETITIVE,
        "LCS: shortlisted firms meeting a quality floor compete on price.",
    ),
    "Fixed Budget Selection": (
        COMPETITIVE,
        "FBS: firms compete on quality within a disclosed fixed budget.",
    ),
    "Consultant Qualification  Selection": (  # note: double space is in the source data
        COMPETITIVE,
        "CQS: shortlist compared on qualifications. Weakest competitive method "
        "(no price competition, small shortlist) but still comparative, so classed "
        "competitive with the caveat noted.",
    ),
    "Individual Consultant Selection": (
        COMPETITIVE,
        "ICS: at least three CVs compared. Classed competitive, but note that 22% of "
        "the dataset is ICS and its supplier field is a placeholder, not a firm.",
    ),
    "E-Auctions": (
        COMPETITIVE,
        "Electronic reverse auction; competition on price is the mechanism itself.",
    ),
    "Direct Selection": (
        NON_COMPETITIVE,
        "Single-source award without competition. The canonical elevated-risk method.",
    ),
    "Force Account": (
        NON_COMPETITIVE,
        "Work executed by the borrower's own government units; no market test at all.",
    ),
    "UN Agencies (Direct)": (
        NON_COMPETITIVE,
        "Direct award to a UN agency. Low fiduciary risk in practice, but structurally "
        "non-competitive; we classify on the mechanism, not on reputation.",
    ),
    "Commercial Practices": (
        NON_COMPETITIVE,
        "Private-sector/FI borrower using its own established practices; no Bank-defined "
        "competition is evidenced in the record.",
    ),
    "Non-Profit Organizations": (
        NON_COMPETITIVE,
        "Direct engagement of an NGO/CSO, typically for reach rather than price.",
    ),
    "Procurement Agents": (
        NON_COMPETITIVE,
        "A third-party agent procures on the borrower's behalf; the award recorded here "
        "is the agent appointment, not a competed contract.",
    ),
    "Community Driven Development": (
        NON_COMPETITIVE,
        "JUDGMENT CALL: CDD delegates procurement to community groups under simplified "
        "rules. It is participatory rather than price-competitive, and no comparison of "
        "offers is evidenced in the record, so we classify it non-competitive. This is "
        "deliberately conservative -- CDD contracts are small (n=484) and a false "
        "elevation costs little, while wrongly calling it competitive would suppress a "
        "genuinely unsupervised award.",
    ),
    "Public Private Partnership": (
        NON_COMPETITIVE,
        "JUDGMENT CALL: PPPs are usually competitively tendered, but the record does not "
        "evidence the tender, n=10, and 100% of them sit above the 99th amount "
        "percentile. Classing them non-competitive routes every one to human attention, "
        "which is the correct posture for the largest, most complex awards in the file.",
    ),
}

def method_lookup_key(value: str) -> str:
    """Canonical form used for taxonomy lookup.

    The published data contains "Consultant Qualification  Selection" with a
    DOUBLE space (15,956 rows, 5.5% of the extract). Any cleaning step that
    collapses whitespace -- as ours does -- breaks an exact-string lookup, and
    the record then fails as an unmapped method. Both the taxonomy keys and the
    incoming value are folded through this function so the two cannot disagree
    over whitespace or casing.
    """
    return " ".join(str(value).split()).upper()


METHOD_CLASS: Final[dict[str, str]] = {
    method_lookup_key(k): v[0] for k, v in PROCUREMENT_METHOD_TAXONOMY.items()
}
METHOD_RATIONALE: Final[dict[str, str]] = {
    method_lookup_key(k): v[1] for k, v in PROCUREMENT_METHOD_TAXONOMY.items()
}

# --------------------------------------------------------------------------
# Supplier placeholders
# --------------------------------------------------------------------------
# 63,603 rows (22.1%) carry the literal supplier name "INDIVIDUAL CONSULTANT",
# spread across 53,905 distinct Supplier IDs. It is a category label standing in
# for a natural person whose name is withheld -- not a vendor. Counting its
# "prior contracts" would hand a fifth of the dataset the history of the most
# prolific supplier in the portfolio. These records are marked
# supplier-unidentifiable and their history features return None, not zero.

SUPPLIER_PLACEHOLDERS: Final[frozenset[str]] = frozenset(
    {
        "INDIVIDUAL CONSULTANT",
        "INDIVIDUAL CONSULTANTS",
        "N/A",
        "NA",
        "NONE",
        "UNKNOWN",
        "TBD",
        "-",
        ".",
        "",
    }
)

# Stripped when normalizing supplier names so that "ACME LTD" and "ACME LIMITED"
# resolve to one entity. Order matters: longest form first.
LEGAL_SUFFIXES: Final[tuple[str, ...]] = (
    "PRIVATE LIMITED", "PVT LTD", "CO LTD", "COMPANY LIMITED", "LIMITED",
    "INCORPORATED", "CORPORATION", "LLC", "LLP", "PLC", "LTDA", "LTD",
    "INC", "CORP", "GMBH", "S A R L", "SARL", "S A S", "SAS", "S P A", "SPA",
    "S A", "SA", "BV", "NV", "AB", "AS", "OY", "PTY", "PT", "CIA", "CIE",
)

# --------------------------------------------------------------------------
# Non-country borrowers
# --------------------------------------------------------------------------
# 3.35% of rows have no borrower country code. Some are encoding artefacts
# (Cote d'Ivoire), but many are genuine multi-country programmes. For those,
# "is the supplier domestic?" has no answer -- it is undefined, not False. The
# feature is therefore tri-state and these rows return None.

REGIONAL_BORROWER_MARKERS: Final[tuple[str, ...]] = (
    "AFRICA", "ASIA", "PACIFIC", "CARIBBEAN", "BALKANS", "REGIONAL",
    "MULTI-REGIONAL", "MULTIREGIONAL", "WORLD", "OCEAN", "EASTERN AND",
    "WESTERN AND", "LATIN AMERICA", "MIDDLE EAST", " - ",
)

# --------------------------------------------------------------------------
# Reference-statistic thresholds
# --------------------------------------------------------------------------
# Category x Region cells range from n=1 (Works x Other) to n=25,773. A median
# over a single observation is not a median. Below MIN_GROUP_SUPPORT we fall
# back up a ladder (category+region -> category -> global) and always emit the
# support count so downstream stages can discount a thin cell.

MIN_GROUP_SUPPORT: Final[int] = 30

# Guard for the ratio denominator. A median of exactly 0 would make the ratio
# infinite; we flag instead of dividing.
MIN_MEDIAN_DENOMINATOR: Final[float] = 1.0

# --------------------------------------------------------------------------
# Time-based split (defined here, consumed by the risk model)
# --------------------------------------------------------------------------
# The intended design is FY2020-2022 for training, FY2023 for validation, and
# the most recent available fiscal year for the final test. The most recent FY
# in this extract is FY2027, which
# has only 1,170 rows (vs ~45,000 typical) because the extract was frozen seven
# weeks into it, and its prior-review share is 17.3% vs the ~7% norm. Testing on
# it would measure reporting lag, not model skill. We therefore reserve FY2027
# as an explicitly-labelled holdout and use FY2024-2026 as the test window.

TRAIN_FISCAL_YEARS: Final[tuple[int, ...]] = (2020, 2021, 2022)
VALIDATION_FISCAL_YEARS: Final[tuple[int, ...]] = (2023,)
TEST_FISCAL_YEARS: Final[tuple[int, ...]] = (2024, 2025, 2026)
TRUNCATED_FISCAL_YEARS: Final[tuple[int, ...]] = (2027,)
