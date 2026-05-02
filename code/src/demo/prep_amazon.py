"""Prepare amazon.csv for the brand-level Bayesian concentration demo.

Produces four files in data/processed/:
  amazon_brand_totals.csv        — model input: brand + total rev_proxy, NO category
  amazon_brand_category_true.csv — held-out ground truth: true brand x category split
  amazon_evidence.csv            — 20% stratified sample with revealed categories
  amazon_keyword_prior.csv       — keyword-count Dirichlet alpha per brand x category

Run:
    python -m src.demo.prep_amazon
"""

from __future__ import annotations

import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — all auditable, no hidden logic
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = _REPO_ROOT / "data" / "processed" / "amazon.csv"
OUT_DIR = _REPO_ROOT / "data" / "processed"

# Brands whose first token is a known variant of a canonical name.
# Format: raw_token → canonical_name
BRAND_CORRECTIONS: dict[str, str] = {
    "boat": "boat",       # lowercase of boAt; already handled by .lower() but explicit here
    "mi": "mi",           # Xiaomi sub-brand; kept separate from redmi/xiaomi
    "amazonbasics": "amazonbasics",  # Amazon private label; kept separate from amazon
    "redmi": "redmi",     # Xiaomi sub-brand; separate from mi by design
    "fire-boltt": "fire_boltt",  # hyphen → underscore for clean column names
    "tp-link": "tp_link",
    "7seven®": "7seven",
    "digitek®": "digitek",
    "smashtronics®": "smashtronics",
    "crypo™": "crypo",
    "rts™": "rts",
    "prolegend®": "prolegend",
    "zorbes®": "zorbes",
    "opentech®": "opentech",
    "memeho®": "memeho",
    "envie®": "envie",
    "verilux®": "verilux",
    "zebronics,": "zebronics",   # trailing comma
    "lunagariya®,": "lunagariya",
    "noise_colorfit": "noise",   # noise sub-brand
    "saleon™": "saleon",
}

# Maps raw top-level category strings to three clean labels + Other.
CATEGORY_MAP: dict[str, str] = {
    "Electronics": "Electronics",
    "Computers&Accessories": "Computers",
    "Home&Kitchen": "HomeKitchen",
    "OfficeProducts": "Other",
    "MusicalInstruments": "Other",
    "HomeImprovement": "Other",
    "Toys&Games": "Other",
    "Car&Motorbike": "Other",
    "Health&PersonalCare": "Other",
}

CATEGORIES = ["Electronics", "Computers", "HomeKitchen"]  # ordered; Other excluded from model

# Keywords used to build the Dirichlet prior.
# Each keyword contributes +1 to that category's alpha for the brand if found in text.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Electronics": [
        "cable", "usb", "charger", "charging", "smartwatch", "watch", "wearable",
        "tv", "television", "remote", "headphone", "earphone", "earbud", "speaker",
        "bluetooth", "mobile", "smartphone", "phone", "power bank", "powerbank",
        "audio", "wifi", "wireless", "earbuds",
    ],
    "Computers": [
        "laptop", "keyboard", "mouse", "networking", "network", "adapter",
        "monitor", "hub", "webcam", "lapdesk", "computer", "pc ", "desktop",
        "printer", "scanner", "tablet", "graphic tablet", "usb hub",
    ],
    "HomeKitchen": [
        "mixer", "grinder", "iron", "kettle", "heater", "water heater",
        "vacuum", "kitchen", "blender", "juicer", "purifier", "cooler",
        "fan", "egg boiler", "sandwich", "toaster", "induction", "oven",
        "laundry", "basket", "geysers",
    ],
}

# Minimum product count for a brand to be included in the model
MIN_PRODUCTS = 5
# Evidence fraction — 20% of products per brand with category revealed
EVIDENCE_FRACTION = 0.20
EVIDENCE_SEED = 42


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_price(value: str) -> float:
    """Convert a rupee price string like '₹1,099' or '₹176.63' to float."""
    if pd.isna(value):
        return float("nan")
    cleaned = re.sub(r"[₹,\s]", "", str(value))
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def parse_rating_count(value) -> float:
    """Convert an Indian-formatted count string like '1,79,691' to float."""
    if pd.isna(value):
        return float("nan")
    cleaned = re.sub(r"[,\s]", "", str(value))
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def extract_brand(product_name: str) -> str:
    """Return the canonical brand for a product name."""
    if pd.isna(product_name):
        return "unknown"
    token = str(product_name).split()[0].lower().strip()
    return BRAND_CORRECTIONS.get(token, token)


def roll_up_category(category_str: str) -> str:
    """Map a pipe-delimited category hierarchy to a clean top-level label."""
    if pd.isna(category_str):
        return "Other"
    top = str(category_str).split("|")[0].strip()
    return CATEGORY_MAP.get(top, "Other")


# ---------------------------------------------------------------------------
# Keyword prior builder
# ---------------------------------------------------------------------------

def _keyword_hits(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def build_keyword_prior(df: pd.DataFrame) -> pd.DataFrame:
    """Compute keyword hit counts per brand x category from product text.

    Returns a DataFrame with columns: brand, Electronics_hits, Computers_hits,
    HomeKitchen_hits.  All values are >= 1 (smoothed) so Dirichlet alpha is valid.
    """
    df = df.copy()
    df["text"] = (
        df["product_name"].fillna("") + " " + df["about_product"].fillna("")
    )

    rows = []
    for brand, grp in df.groupby("brand"):
        combined_text = " ".join(grp["text"].tolist())
        row = {"brand": brand}
        for cat in CATEGORIES:
            hits = _keyword_hits(combined_text, CATEGORY_KEYWORDS[cat])
            row[f"{cat}_hits"] = hits
        rows.append(row)

    prior = pd.DataFrame(rows)
    # Add 1 for Laplace smoothing — ensures all alphas > 0
    for cat in CATEGORIES:
        prior[f"{cat}_hits"] = prior[f"{cat}_hits"] + 1

    return prior


# ---------------------------------------------------------------------------
# Evidence set builder
# ---------------------------------------------------------------------------

def build_evidence_set(df: pd.DataFrame, fraction: float = EVIDENCE_FRACTION,
                       seed: int = EVIDENCE_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into evidence (revealed) and non-evidence rows.

    Stratified by brand: every brand with >= 1 product gets at least 1 evidence row
    (ceil of fraction * count, minimum 1).  Returns (evidence_df, rest_df).
    """
    rng = np.random.default_rng(seed)
    evidence_rows = []
    rest_rows = []

    for brand, grp in df.groupby("brand"):
        n = len(grp)
        n_evidence = max(1, int(np.ceil(n * fraction)))
        chosen_idx = rng.choice(grp.index, size=n_evidence, replace=False)
        evidence_rows.append(grp.loc[chosen_idx])
        rest_rows.append(grp.drop(index=chosen_idx))

    evidence = pd.concat(evidence_rows).reset_index(drop=True)
    rest = pd.concat(rest_rows).reset_index(drop=True) if rest_rows else pd.DataFrame()
    return evidence, rest


# ---------------------------------------------------------------------------
# Main prep pipeline
# ---------------------------------------------------------------------------

def run_prep(raw_path: Path = RAW_PATH, out_dir: Path = OUT_DIR,
             min_products: int = MIN_PRODUCTS) -> dict[str, pd.DataFrame]:
    """Full preparation pipeline.  Returns dict of output DataFrames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_path)
    n_raw = len(df)

    # --- Parse numeric columns ---
    df["price"] = df["discounted_price"].apply(parse_price)
    df["n_ratings"] = df["rating_count"].apply(parse_rating_count)
    df["rev_proxy"] = df["price"] * df["n_ratings"]

    n_nan = df["rev_proxy"].isna().sum()
    n_zero = (df["rev_proxy"] == 0).sum()
    logger.info("Rev proxy: %d NaN, %d zero out of %d rows", n_nan, n_zero, n_raw)
    df = df[df["rev_proxy"].notna() & (df["rev_proxy"] > 0)].copy()
    logger.info("Kept %d rows after dropping NaN/zero rev_proxy", len(df))

    # --- Brand and category ---
    df["brand"] = df["product_name"].apply(extract_brand)
    df["category"] = df["category"].apply(roll_up_category)

    # Drop "Other" category (too small and heterogeneous for modeling)
    n_other = (df["category"] == "Other").sum()
    logger.info("Dropping %d rows in 'Other' category", n_other)
    df = df[df["category"].isin(CATEGORIES)].copy()

    # --- Filter to brands with >= min_products ---
    brand_counts = df.groupby("brand").size()
    valid_brands = brand_counts[brand_counts >= min_products].index
    n_dropped_brands = brand_counts[brand_counts < min_products].sum()
    logger.info(
        "Keeping %d brands (>= %d products); dropping %d products from %d small brands",
        len(valid_brands), min_products, n_dropped_brands,
        (brand_counts < min_products).sum(),
    )
    df = df[df["brand"].isin(valid_brands)].copy()

    # --- Build keyword prior (before hiding category) ---
    keyword_prior = build_keyword_prior(df)

    # --- Build evidence set (with category revealed) ---
    evidence, _ = build_evidence_set(df)

    # --- Ground truth: brand x category aggregation ---
    true_panel = (
        df.groupby(["brand", "category"])
        .agg(true_rev_proxy=("rev_proxy", "sum"), product_count=("rev_proxy", "count"))
        .reset_index()
    )

    # --- Brand totals (NO category column — the model's input) ---
    brand_totals = (
        df.groupby("brand")
        .agg(total_rev_proxy=("rev_proxy", "sum"), total_product_count=("rev_proxy", "count"))
        .reset_index()
    )

    # --- Evidence aggregation by brand x category ---
    evidence_agg = (
        evidence.groupby(["brand", "category"])
        .agg(evidence_rev_proxy=("rev_proxy", "sum"), evidence_count=("rev_proxy", "count"))
        .reset_index()
    )

    # --- Save ---
    brand_totals.to_csv(out_dir / "amazon_brand_totals.csv", index=False)
    true_panel.to_csv(out_dir / "amazon_brand_category_true.csv", index=False)
    evidence_agg.to_csv(out_dir / "amazon_evidence.csv", index=False)
    keyword_prior.to_csv(out_dir / "amazon_keyword_prior.csv", index=False)

    # --- Summary stats ---
    n_brands = len(brand_totals)
    n_multi = (
        true_panel.groupby("brand")["category"].nunique()
        .pipe(lambda s: (s > 1).sum())
    )
    top5 = brand_totals.nlargest(5, "total_rev_proxy")[["brand", "total_rev_proxy"]]

    print(f"\n=== Prep summary ===")
    print(f"Products used: {len(df):,}  (dropped {n_raw - len(df):,})")
    print(f"Brands: {n_brands}  ({n_multi} span >1 category)")
    print(f"Evidence rows: {len(evidence_agg)}")
    print(f"\nTop-5 brands by revenue proxy:")
    print(top5.to_string(index=False))

    # Naive CR4 per category
    cat_shares = (
        true_panel.assign(
            cat_total=true_panel.groupby("category")["true_rev_proxy"].transform("sum")
        )
        .assign(share=lambda d: d["true_rev_proxy"] / d["cat_total"])
        .sort_values(["category", "share"], ascending=[True, False])
    )
    print("\nNaive CR4 per category (point estimate, full data):")
    for cat, grp in cat_shares.groupby("category"):
        cr4 = grp["share"].head(4).sum()
        top4 = grp["brand"].head(4).tolist()
        print(f"  {cat}: CR4 = {cr4:.3f}  top-4 = {top4}")

    return {
        "brand_totals": brand_totals,
        "true_panel": true_panel,
        "evidence": evidence_agg,
        "keyword_prior": keyword_prior,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_prep()
