"""Bayesian brand x category concentration model for the Amazon product demo.

The model treats each brand's category allocation as a latent Dirichlet variable.
It observes only:
  1. brand-level total revenue proxy (no category breakdown)
  2. a 20% evidence sample of products with revealed categories

The posterior over allocation propagates into posterior CR4 and HHI per category.

Run end-to-end:
    python -m src.demo.bayes_brand
"""

from __future__ import annotations

import logging
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.demo.prep_amazon import CATEGORIES, run_prep

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = _REPO_ROOT / "output" / "traces" / "demo_brand.nc"
OUT_DIR = _REPO_ROOT / "output" / "tables" / "demo"
EVIDENCE_FRACTION = 0.20   # must match prep_amazon.EVIDENCE_FRACTION


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_brand_model(
    brand_totals_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    keyword_prior_df: pd.DataFrame,
) -> tuple[pm.Model, dict]:
    """Construct the PyMC model.

    Parameters
    ----------
    brand_totals_df : columns [brand, total_rev_proxy]
    evidence_df     : columns [brand, category, evidence_rev_proxy]
    keyword_prior_df: columns [brand, Electronics_hits, Computers_hits, HomeKitchen_hits]

    Returns
    -------
    model : pm.Model
    coords: dict of coordinates used in the model (brands, categories)
    """
    K = len(CATEGORIES)

    # Align all tables on the same ordered brand list
    brands = brand_totals_df["brand"].tolist()
    B = len(brands)
    brand_idx = {b: i for i, b in enumerate(brands)}

    # Build alpha matrix (B x K) from keyword prior
    kp = keyword_prior_df.set_index("brand")
    alpha = np.ones((B, K))
    for i, brand in enumerate(brands):
        if brand in kp.index:
            for k, cat in enumerate(CATEGORIES):
                alpha[i, k] = float(kp.loc[brand, f"{cat}_hits"])

    # Total revenue per brand (B,)
    bt = brand_totals_df.set_index("brand")["total_rev_proxy"]
    total_rev = np.array([bt.loc[b] for b in brands], dtype=float)

    # Build observed evidence arrays: one entry per (brand, category) cell with evidence > 0
    # observed_val[j] = evidence_rev_proxy for that cell
    # expected    [j] = EVIDENCE_FRACTION * inferred_rev[brand_j, cat_j]
    obs_vals = []
    obs_brand_idx = []
    obs_cat_idx = []

    cat_idx_map = {c: i for i, c in enumerate(CATEGORIES)}

    for _, row in evidence_df.iterrows():
        if row["brand"] not in brand_idx:
            continue
        if row["category"] not in cat_idx_map:
            continue
        if row["evidence_rev_proxy"] <= 0:
            continue
        obs_brand_idx.append(brand_idx[row["brand"]])
        obs_cat_idx.append(cat_idx_map[row["category"]])
        obs_vals.append(row["evidence_rev_proxy"])

    obs_vals = np.array(obs_vals, dtype=float)
    obs_brand_idx = np.array(obs_brand_idx, dtype=int)
    obs_cat_idx = np.array(obs_cat_idx, dtype=int)

    coords = {"brand": brands, "category": CATEGORIES}

    with pm.Model(coords=coords) as model:
        # Latent allocation: brand x category Dirichlet
        allocation = pm.Dirichlet(
            "allocation",
            a=alpha,
            shape=(B, K),
        )

        # Inferred revenue per brand x category
        inferred_rev = total_rev[:, None] * allocation   # (B, K)

        # Noise on evidence observations
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        # Likelihood: evidence rev_proxy per (brand, category) cell
        # Expected value = EVIDENCE_FRACTION * inferred_rev for that cell
        expected = EVIDENCE_FRACTION * inferred_rev[obs_brand_idx, obs_cat_idx]
        pm.LogNormal(
            "evidence_obs",
            mu=pm.math.log(expected + 1e-9),   # +epsilon avoids log(0) if allocation → 0
            sigma=sigma,
            observed=obs_vals,
        )

    return model, coords


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def fit_brand_model(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    target_accept: float = 0.9,
) -> az.InferenceData:
    """Sample the model and save the trace.

    Prints R-hat summary and divergence count.  Returns the InferenceData object.
    """
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            progressbar=True,
            return_inferencedata=True,
        )

    # Diagnostics
    rhat = az.rhat(trace)
    max_rhat = float(max(rhat[v].values.max() for v in rhat.data_vars))
    n_div = int(trace.sample_stats["diverging"].values.sum())
    logger.info("Max R-hat: %.4f | Divergences: %d", max_rhat, n_div)
    print(f"\nMax R-hat: {max_rhat:.4f}  |  Divergences: {n_div}")

    if max_rhat > 1.05:
        logger.warning("R-hat > 1.05 — chains may not have converged")

    az.to_netcdf(trace, TRACE_PATH)
    logger.info("Trace saved to %s", TRACE_PATH)
    return trace


# ---------------------------------------------------------------------------
# Posterior summaries
# ---------------------------------------------------------------------------

def compute_posterior_concentration(
    trace: az.InferenceData,
    brand_totals_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Compute per-category posterior CR4 and HHI from the trace.

    Returns dict with keys:
      '<category>_cr4'  — shape (n_samples,)
      '<category>_hhi'  — shape (n_samples,)
    """
    brands = brand_totals_df["brand"].tolist()
    bt = brand_totals_df.set_index("brand")["total_rev_proxy"]
    total_rev = np.array([bt.loc[b] for b in brands], dtype=float)  # (B,)

    # allocation posterior: shape (chains, draws, B, K) → flatten to (S, B, K)
    alloc = trace.posterior["allocation"].values
    S = alloc.shape[0] * alloc.shape[1]
    alloc_flat = alloc.reshape(S, len(brands), len(CATEGORIES))  # (S, B, K)

    # Inferred revenue per sample: (S, B, K)
    inferred = total_rev[None, :, None] * alloc_flat

    results = {}
    for k, cat in enumerate(CATEGORIES):
        rev_k = inferred[:, :, k]                           # (S, B)
        cat_total = rev_k.sum(axis=1, keepdims=True)        # (S, 1)
        shares = rev_k / (cat_total + 1e-12)                # (S, B)

        # CR4: top-4 shares per sample
        top4 = np.sort(shares, axis=1)[:, -4:]
        cr4 = top4.sum(axis=1)                              # (S,)

        # HHI
        hhi = (shares ** 2).sum(axis=1) * 10_000           # (S,)

        results[f"{cat}_cr4"] = cr4
        results[f"{cat}_hhi"] = hhi

    return results


def summarize_concentration(
    posterior: dict[str, np.ndarray],
    naive_cr4: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Tabulate posterior CR4 and HHI with credible intervals.

    Parameters
    ----------
    posterior  : output of compute_posterior_concentration
    naive_cr4  : optional dict of point-estimate CR4 per category for comparison
    """
    rows = []
    for cat in CATEGORIES:
        cr4_samples = posterior[f"{cat}_cr4"]
        hhi_samples = posterior[f"{cat}_hhi"]
        row = {
            "category": cat,
            "cr4_mean": cr4_samples.mean(),
            "cr4_p10": np.percentile(cr4_samples, 10),
            "cr4_p90": np.percentile(cr4_samples, 90),
            "hhi_mean": hhi_samples.mean(),
            "hhi_p10": np.percentile(hhi_samples, 10),
            "hhi_p90": np.percentile(hhi_samples, 90),
        }
        if naive_cr4:
            row["naive_cr4"] = naive_cr4.get(cat, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------

def run(draws: int = 1000, tune: int = 1000, chains: int = 2) -> None:
    """Prepare data, fit model, save summary table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data = run_prep()
    brand_totals = data["brand_totals"]
    evidence = data["evidence"]
    keyword_prior = data["keyword_prior"]
    true_panel = data["true_panel"]

    # Naive CR4 from ground truth (oracle, for comparison)
    naive_cr4 = {}
    for cat, grp in true_panel.groupby("category"):
        cat_total = grp["true_rev_proxy"].sum()
        top4_sum = grp.nlargest(4, "true_rev_proxy")["true_rev_proxy"].sum()
        naive_cr4[cat] = top4_sum / cat_total

    model, _ = build_brand_model(brand_totals, evidence, keyword_prior)
    trace = fit_brand_model(model, draws=draws, tune=tune, chains=chains)
    posterior = compute_posterior_concentration(trace, brand_totals)
    summary = summarize_concentration(posterior, naive_cr4=naive_cr4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "concentration_summary.csv", index=False)
    print("\n=== Posterior concentration summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
