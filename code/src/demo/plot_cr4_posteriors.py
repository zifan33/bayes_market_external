"""Generate the headline CR4 posterior distribution figure.

Produces output/figures/demo/posterior_cr4_distributions.png —
one smooth KDE curve per category, overlaid on a shared axis, with
naive point-estimate markers and a DOJ high-concentration threshold.
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import gaussian_kde

from src.demo.bayes_brand import compute_posterior_concentration
from src.demo.prep_amazon import CATEGORIES, run_prep

_REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_PATH = _REPO_ROOT / "output" / "figures" / "demo" / "posterior_cr4_distributions.png"

# DOJ "highly concentrated" threshold in CR4 terms.
# DOJ HMG flags HHI > 2500; empirically that corresponds to CR4 ≈ 0.70 in
# symmetric-firm models, but 0.5 is a common rule-of-thumb for CR4 alone.
DOJ_CR4_THRESHOLD = 0.50

PALETTE = {
    "Electronics":  "#1565C0",   # deep blue
    "Computers":    "#E65100",   # deep orange
    "HomeKitchen":  "#2E7D32",   # deep green
}

LABEL_DISPLAY = {
    "Electronics":  "Electronics",
    "Computers":    "Computers & Accessories",
    "HomeKitchen":  "Home & Kitchen",
}


def make_figure(
    posterior: dict[str, np.ndarray],
    naive_cr4: dict[str, float],
    *,
    save_path: Path = FIG_PATH,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x_grid = np.linspace(0.0, 1.0, 500)

    for cat in CATEGORIES:
        samples = posterior[f"{cat}_cr4"]
        color = PALETTE[cat]
        label = LABEL_DISPLAY[cat]

        kde = gaussian_kde(samples, bw_method="scott")
        density = kde(x_grid)

        # Filled curve
        ax.fill_between(x_grid, density, alpha=0.20, color=color)
        ax.plot(x_grid, density, color=color, linewidth=2.2, label=label)

        # Posterior median tick
        median = float(np.median(samples))
        ax.axvline(median, color=color, linewidth=1.2, linestyle="--", alpha=0.7)

        # 80 % CI bracket on x-axis
        p10, p90 = np.percentile(samples, [10, 90])
        ax.annotate(
            "",
            xy=(p90, -0.35), xytext=(p10, -0.35),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops=dict(
                arrowstyle="|-|,widthA=0.4,widthB=0.4",
                color=color, lw=1.5,
            ),
            annotation_clip=False,
        )

        # Naive point-estimate marker (the "prior literature" number)
        naive = naive_cr4[cat]
        ax.scatter(
            naive, -0.45, s=80, color=color, marker="v", zorder=5,
            clip_on=False, transform=ax.get_xaxis_transform(),
        )

    # DOJ threshold line
    ax.axvline(
        DOJ_CR4_THRESHOLD, color="black", linewidth=1.0,
        linestyle=(0, (4, 3)), alpha=0.55,
    )
    ax.text(
        DOJ_CR4_THRESHOLD + 0.007, 0.97, "CR4 = 0.5\n(reference)",
        transform=ax.get_xaxis_transform(),
        fontsize=7.5, color="black", alpha=0.7,
        va="top", ha="left",
    )

    # Axis formatting
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("CR4 (four-firm concentration ratio)", fontsize=11)
    ax.set_ylabel("Posterior density", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(bottom=0)

    # Legend for lines
    handles = [
        mpatches.Patch(facecolor=PALETTE[c], alpha=0.55, label=LABEL_DISPLAY[c])
        for c in CATEGORIES
    ]
    # Naive marker legend entry
    naive_handle = plt.scatter(
        [], [], marker="v", s=80, color="dimgray",
        label="Naive point estimate\n(prior literature)",
    )
    handles.append(naive_handle)
    ax.legend(handles=handles, fontsize=8.5, frameon=False, loc="upper left")

    ax.set_title(
        "Posterior CR4 per product category\n"
        "— Bayesian latent-allocation model recovers a distribution, not a point —",
        fontsize=10.5, pad=10,
    )

    # Bottom annotation
    fig.text(
        0.5, -0.06,
        "Triangles (▼) = naive point estimate from full data  ·  "
        "Brackets = 80 % credible interval  ·  Dashed lines = posterior median",
        ha="center", fontsize=7.5, color="dimgray",
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved → {save_path}")
    return fig


if __name__ == "__main__":
    data = run_prep()
    trace = az.from_netcdf(_REPO_ROOT / "output" / "traces" / "demo_brand.nc")
    posterior = compute_posterior_concentration(trace, data["brand_totals"])

    naive_cr4 = {}
    for cat, grp in data["true_panel"].groupby("category"):
        cat_total = grp["true_rev_proxy"].sum()
        naive_cr4[cat] = grp.nlargest(4, "true_rev_proxy")["true_rev_proxy"].sum() / cat_total

    make_figure(posterior, naive_cr4)
