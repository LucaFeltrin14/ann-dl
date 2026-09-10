"""Exercise 1 - Point Clouds: Geometry and Spread in 2D.

Generates four 2D Gaussian point clouds, measures how their spread changes the
difficulty of the classification problem, and writes Figures 1, 1b, 2 and 3.

Every number quoted in the report is printed by this script. Run it with:

    python docs/exercises/data/code/exercise1_point_clouds.py
"""

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no interactive display needed: we only save files
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Setup: one seed for the whole report, so every figure and number is reproducible
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Generating parameters given in the statement: one row per class, [x, y].
MEANS = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 1.0], [15.0, 4.0]])
STDS = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])
N_PER_CLASS = 100
N_CLASSES = len(MEANS)
SCALES = [0.5, 1.0, 2.0, 4.0]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
LABELS = [f"Class {k}" for k in range(N_CLASSES)]

# The standard normal draws are taken ONCE and reused for every scale factor, so
# x = mu + s * sigma * z. Two consequences, both wanted: the dataset of item A is
# exactly the s = 1 dataset, and the four panels of Figure 2 differ only by spread
# (same underlying points), which is what makes the comparison across s honest.
Z = rng.standard_normal((N_CLASSES, N_PER_CLASS, 2))

# Labels never change either: 100 points of class 0, then 100 of class 1, ...
Y = np.repeat(np.arange(N_CLASSES), N_PER_CLASS)


def make_dataset(scale: float) -> np.ndarray:
    """Return the (400, 2) cloud for a given spread scale factor `s`."""
    points = MEANS[:, None, :] + scale * STDS[:, None, :] * Z
    return points.reshape(-1, 2)


def separation_ratios() -> list[tuple[int, int, float, float, float]]:
    """r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j) for the 6 class pairs at s = 1."""
    sigma_bar = STDS.mean(axis=1)  # (sigma_x + sigma_y) / 2, per class
    out = []
    for i, j in combinations(range(N_CLASSES), 2):
        distance = float(np.linalg.norm(MEANS[i] - MEANS[j]))
        denominator = float(sigma_bar[i] + sigma_bar[j])
        out.append((i, j, distance, denominator, distance / denominator))
    return out


def mixing_rate(points: np.ndarray) -> float:
    """Fraction of points whose nearest class centre is not their own class centre.

    Purely geometric: the centres are the generating means (not sample means) and
    nothing is trained - it is a nearest-centroid assignment computed with NumPy.
    """
    distances = np.linalg.norm(points[:, None, :] - MEANS[None, :, :], axis=2)
    return float((distances.argmin(axis=1) != Y).mean())


def scatter_classes(ax: plt.Axes, points: np.ndarray, *, mark_centres: bool = True) -> None:
    """Scatter the four classes on `ax`, optionally marking the cloud centres."""
    for k in range(N_CLASSES):
        cloud = points[Y == k]
        ax.scatter(cloud[:, 0], cloud[:, 1], s=14, alpha=0.75, color=COLORS[k], label=LABELS[k])
    if mark_centres:
        ax.scatter(
            MEANS[:, 0],
            MEANS[:, 1],
            marker="X",
            s=200,
            c="black",
            edgecolors="white",
            linewidths=1.5,
            zorder=5,
            label="Class centre (mean)",
        )
    ax.set_xlabel("Feature 1 ($x_1$)")
    ax.set_ylabel("Feature 2 ($x_2$)")
    ax.grid(alpha=0.25, linestyle=":")


# ---------------------------------------------------------------------------
# A - Generate the clouds (Figure 1)
# ---------------------------------------------------------------------------
base = make_dataset(1.0)

fig, ax = plt.subplots(figsize=(8, 6))
scatter_classes(ax, base)
ax.set_title("Figure 1 - Four Gaussian point clouds in 2D (400 samples, $s = 1$)")
ax.legend(loc="upper left", framealpha=0.95)
fig.tight_layout()
fig.savefig(FIGURES / "fig1_clouds.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# C - the same clouds with the decision boundaries sketched on top (Figure 1b)
# The sketch is the nearest-centroid partition of the plane: the piecewise-linear
# boundaries that a network would have to approximate to tell the clouds apart.
# ---------------------------------------------------------------------------
pad = 1.5
grid_x = np.linspace(base[:, 0].min() - pad, base[:, 0].max() + pad, 600)
grid_y = np.linspace(base[:, 1].min() - pad, base[:, 1].max() + pad, 600)
mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
mesh = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
regions = np.linalg.norm(mesh[:, None, :] - MEANS[None, :, :], axis=2).argmin(axis=1)
regions = regions.reshape(mesh_x.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(mesh_x, mesh_y, regions, levels=np.arange(-0.5, N_CLASSES), colors=COLORS, alpha=0.15)
ax.contour(mesh_x, mesh_y, regions, levels=np.arange(0.5, N_CLASSES - 0.5), colors="black",
           linewidths=1.4, linestyles="--")
scatter_classes(ax, base)
ax.set_xlim(grid_x[0], grid_x[-1])
ax.set_ylim(grid_y[0], grid_y[-1])
ax.set_title("Figure 1b - Figure 1 with the decision boundaries sketched on top")
ax.legend(loc="upper left", framealpha=0.95)
fig.tight_layout()
fig.savefig(FIGURES / "fig1_boundaries.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# B - More or less spread out (Figures 2 and 3)
# ---------------------------------------------------------------------------
datasets = {s: make_dataset(s) for s in SCALES}

# Shared axis limits across the four panels, taken from the widest dataset, so the
# panels are directly comparable instead of each one being auto-zoomed.
widest = datasets[max(SCALES)]
xlim = (widest[:, 0].min() - pad, widest[:, 0].max() + pad)
ylim = (widest[:, 1].min() - pad, widest[:, 1].max() + pad)

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
for ax, s in zip(axes.ravel(), SCALES):
    scatter_classes(ax, datasets[s])
    ax.set_title(f"$s = {s}$  (mixing rate = {mixing_rate(datasets[s]):.2%})")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.label_outer()  # keep the axis labels only on the outer edges of the grid
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
fig.suptitle("Figure 2 - The same four classes at four spread scales (shared axes)", fontsize=13)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(FIGURES / "fig2_scales.png", dpi=150)
plt.close(fig)

rates = [mixing_rate(datasets[s]) for s in SCALES]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(SCALES, rates, marker="o", color="#1f77b4", linewidth=2, label="Mixing rate")
for s, rate in zip(SCALES, rates):
    ax.annotate(f"{rate:.2%}", (s, rate), textcoords="offset points", xytext=(0, 10), ha="center")
ax.axhline(0.75, color="grey", linestyle=":", linewidth=1,
           label="Chance level for 4 balanced classes (75%)")
ax.axvline(2.0, color="#d62728", linestyle="--", linewidth=1.2,
           label="$s = 2$: smallest $r_{ij}$ drops below 1")
ax.set_xlabel("Spread scale factor $s$")
ax.set_ylabel("Mixing rate (fraction of points closer to another class centre)")
ax.set_title("Figure 3 - Mixing rate $\\times$ spread scale factor $s$")
ax.set_xticks(SCALES)
ax.set_ylim(0, 0.8)
ax.grid(alpha=0.25, linestyle=":")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES / "fig3_mixing_rate.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Reported numbers
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 78)
    print("EXERCISE 1 - POINT CLOUDS: GEOMETRY AND SPREAD IN 2D")
    print(f"seed = {SEED} | {N_CLASSES} classes x {N_PER_CLASS} samples = {len(base)} samples")
    print("=" * 78)

    print("\n[A] mean and std per class, and the resulting sample statistics at s = 1")
    for k in range(N_CLASSES):
        cloud = base[Y == k]
        print(
            f"  class {k}: mu={MEANS[k].tolist()} sigma={STDS[k].tolist()}"
            f" | sample mean={np.round(cloud.mean(axis=0), 3).tolist()}"
            f" sample std={np.round(cloud.std(axis=0), 3).tolist()}"
        )

    print("\n[B] separation ratio r_ij at s = 1 (markdown table)")
    ratios = separation_ratios()
    print("| Pair | ||mu_i - mu_j|| | sigma_bar_i + sigma_bar_j | r_ij |")
    print("|---|---|---|---|")
    for i, j, distance, denominator, ratio in ratios:
        print(f"| ({i}, {j}) | {distance:.4f} | {denominator:.4f} | {ratio:.4f} |")
    smallest = min(ratios, key=lambda row: row[-1])
    print(
        f"  smallest r_ij = {smallest[-1]:.4f} for pair ({smallest[0]}, {smallest[1]})"
        f" -> at s = 2 it becomes {smallest[-1]:.4f} / 2 = {smallest[-1] / 2:.4f}"
    )
    print(f"  sigma_bar per class = {np.round(STDS.mean(axis=1), 4).tolist()}")

    print("\n[B] mixing rate per scale factor")
    for s, rate in zip(SCALES, rates):
        wrong = int(round(rate * len(base)))
        print(f"  s = {s:<4} mixing rate = {rate:.4f}  ({wrong}/{len(base)} points)")

    print("\n[figures]")
    for name in ("fig1_clouds.png", "fig1_boundaries.png", "fig2_scales.png", "fig3_mixing_rate.png"):
        print(f"  saved {FIGURES / name}")
