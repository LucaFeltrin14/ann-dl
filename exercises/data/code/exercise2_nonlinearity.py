"""Exercise 2 - Non-Linearity in Higher Dimensions.

Builds two 5-dimensional, two-class datasets - shifted Gaussians (Dataset I) and
concentric shells (Dataset II) - projects both with PCA and measures them in the
original 5D space. Writes Figures 4 and 5.

Every number quoted in the report is printed by this script. Run it with:

    python docs/exercises/data/code/exercise2_nonlinearity.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

N_PER_CLASS = 500
N_FEATURES = 5

# --- Dataset I: two shifted Gaussians with different covariance structures -----
MU_A = np.zeros(N_FEATURES)
SIGMA_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
MU_B = np.full(N_FEATURES, 1.5)
SIGMA_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

# --- Dataset II: two concentric shells ----------------------------------------
# "radius rho ~ N(2.0, 0.4)" is read as mean 2.0 and STANDARD DEVIATION 0.4,
# matching how the parameters are given in Exercise 1 and numpy's rng.normal(loc, scale).
RHO_C = (2.0, 0.4)
RHO_D = (5.0, 0.4)

COLORS = {"I": ["#1f77b4", "#ff7f0e"], "II": ["#2ca02c", "#d62728"]}
NAMES = {"I": ["Class A", "Class B"], "II": ["Class C (core)", "Class D (shell)"]}


def make_dataset_i() -> tuple[np.ndarray, np.ndarray]:
    """Dataset I - 500 + 500 samples from two multivariate normals in 5D."""
    class_a = rng.multivariate_normal(MU_A, SIGMA_A, size=N_PER_CLASS)
    class_b = rng.multivariate_normal(MU_B, SIGMA_B, size=N_PER_CLASS)
    return np.vstack([class_a, class_b]), np.repeat([0, 1], N_PER_CLASS)


def sample_shell(mean_radius: float, std_radius: float, n: int) -> np.ndarray:
    """n points at a random direction on the unit sphere of R^5, at a random radius.

    Directions are drawn as v ~ N(0, I_5) and normalised to u = v / ||v||, which is
    the standard way of getting a uniform direction on the sphere; the radius is an
    independent Gaussian, so the points sit on a fuzzy spherical shell.
    """
    directions = rng.standard_normal((n, N_FEATURES))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.normal(mean_radius, std_radius, size=n)
    return radii[:, None] * directions


def make_dataset_ii() -> tuple[np.ndarray, np.ndarray]:
    """Dataset II - a dense core (Class C) inside a hollow shell (Class D)."""
    class_c = sample_shell(*RHO_C, N_PER_CLASS)
    class_d = sample_shell(*RHO_D, N_PER_CLASS)
    return np.vstack([class_c, class_d]), np.repeat([0, 1], N_PER_CLASS)


def centre_distance(points: np.ndarray, labels: np.ndarray) -> float:
    """||mu_1 - mu_2|| computed in the original 5D space."""
    return float(np.linalg.norm(points[labels == 0].mean(axis=0) - points[labels == 1].mean(axis=0)))


X1, y1 = make_dataset_i()
X2, y2 = make_dataset_ii()
DATASETS = {"I": (X1, y1), "II": (X2, y2)}

# ---------------------------------------------------------------------------
# C - Figure 4: PCA projection of both datasets, side by side
# ---------------------------------------------------------------------------
projections, variances = {}, {}
for key, (points, labels) in DATASETS.items():
    pca = PCA(n_components=2, random_state=SEED)
    projections[key] = pca.fit_transform(points)
    variances[key] = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, key in zip(axes, DATASETS):
    labels = DATASETS[key][1]
    for value in (0, 1):
        cloud = projections[key][labels == value]
        ax.scatter(cloud[:, 0], cloud[:, 1], s=12, alpha=0.6,
                   color=COLORS[key][value], label=NAMES[key][value])
    total = variances[key].sum()
    ax.set_title(f"Dataset {key} - PC1 + PC2 = {total:.2%} of the variance")
    ax.set_xlabel(f"PC1 ({variances[key][0]:.2%})")
    ax.set_ylabel(f"PC2 ({variances[key][1]:.2%})")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="upper right")
fig.suptitle("Figure 4 - PCA projection of the two 5D datasets onto 2 dimensions", fontsize=13)
fig.tight_layout()
fig.savefig(FIGURES / "fig4_pca.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# C - Figure 5: histogram of the radius ||x|| of each point, per dataset
# ---------------------------------------------------------------------------
radii = {key: np.linalg.norm(points, axis=1) for key, (points, _) in DATASETS.items()}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, key in zip(axes, DATASETS):
    labels = DATASETS[key][1]
    bins = np.linspace(0, radii[key].max() * 1.02, 45)
    for value in (0, 1):
        ax.hist(radii[key][labels == value], bins=bins, alpha=0.6,
                color=COLORS[key][value], label=NAMES[key][value], edgecolor="white", linewidth=0.4)
    if key == "II":
        ax.axvline(3.5, color="black", linestyle="--", linewidth=1.4,
                   label=r"$\|x\| = 3.5$ (separating radius)")
    ax.set_title(f"Dataset {key}")
    ax.set_xlabel(r"Radius $\|x\|$")
    ax.set_ylabel("Number of samples")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="upper right")
fig.suptitle(r"Figure 5 - Distribution of the radius $\|x\|$ per class, in the original 5D space",
             fontsize=13)
fig.tight_layout()
fig.savefig(FIGURES / "fig5_radius_hist.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# D - the quadratic function that separates Dataset II
# ---------------------------------------------------------------------------
THRESHOLD = 3.5  # midway between the two shell radii (2.0 and 5.0)


def quadratic_rule(points: np.ndarray) -> np.ndarray:
    """f(x) = ||x||^2 - 3.5^2 ; predict the outer class where f(x) > 0."""
    return (np.sum(points ** 2, axis=1) - THRESHOLD ** 2 > 0).astype(int)


def projection_gaps(points: np.ndarray, labels: np.ndarray, n_directions: int = 6):
    """Where the two classes land when projected on random unit directions w.

    A hyperplane classifies by thresholding w.x, so if both classes project to the
    same place along *every* direction there is no threshold left to pick. This is
    descriptive geometry - no direction is optimised and nothing is trained.
    """
    directions = rng.standard_normal((n_directions, N_FEATURES))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    projected = points @ directions.T
    inner = projected[labels == 0].mean(axis=0)
    outer = projected[labels == 1].mean(axis=0)
    return inner, outer, np.abs(inner - outer)


if __name__ == "__main__":
    print("=" * 78)
    print("EXERCISE 2 - NON-LINEARITY IN HIGHER DIMENSIONS")
    print(f"seed = {SEED} | 2 datasets x 2 classes x {N_PER_CLASS} samples in {N_FEATURES}D")
    print("=" * 78)

    print("\n[A/B] sanity checks on the generated data")
    for name, matrix in (("Sigma_A", SIGMA_A), ("Sigma_B", SIGMA_B)):
        eigenvalues = np.linalg.eigvalsh(matrix)
        print(f"  {name}: symmetric={np.allclose(matrix, matrix.T)}"
              f" positive-definite={bool((eigenvalues > 0).all())}"
              f" (min eigenvalue {eigenvalues.min():.4f})")
    unit_norms = np.linalg.norm(sample_shell(1.0, 0.0, 5), axis=1)
    print(f"  shell directions are unit vectors: radii with std 0 -> {np.round(unit_norms, 6).tolist()}")
    print(f"  Dataset I  shape={X1.shape}  class sizes={np.bincount(y1).tolist()}")
    print(f"  Dataset II shape={X2.shape}  class sizes={np.bincount(y2).tolist()}")

    print("\n[C] PCA - explained variance of the first two components")
    for key in DATASETS:
        pc1, pc2 = variances[key]
        print(f"  Dataset {key}: PC1={pc1:.4f} PC2={pc2:.4f} PC1+PC2={pc1 + pc2:.4f} ({pc1 + pc2:.2%})")

    print("\n[C] geometry measured in the original 5D space")
    print(f"  theoretical ||mu_A - mu_B|| for Dataset I = 1.5 * sqrt(5) = {1.5 * np.sqrt(5):.4f}")
    for key, (points, labels) in DATASETS.items():
        print(f"  Dataset {key}: ||mu_1 - mu_2|| = {centre_distance(points, labels):.4f}")
        for value in (0, 1):
            sub = radii[key][labels == value]
            print(f"      {NAMES[key][value]:<16} radius mean={sub.mean():.4f} std={sub.std():.4f}"
                  f" min={sub.min():.4f} max={sub.max():.4f}")
        inner_max = radii[key][labels == 0].max()
        outer_min = radii[key][labels == 1].min()
        overlap = "OVERLAP" if inner_max > outer_min else "DISJOINT"
        print(f"      radius ranges are {overlap} (inner max {inner_max:.4f} vs outer min {outer_min:.4f})")

    print("\n[D] projection of both classes onto 6 random unit directions w")
    print("    (mean of w.x per class, and the gap between them)")
    for key, (points, labels) in DATASETS.items():
        inner, outer, gap = projection_gaps(points, labels)
        first, second = NAMES[key]
        print(f"  Dataset {key}:")
        print(f"      {first:<16} mean w.x = {np.round(inner, 4).tolist()}")
        print(f"      {second:<16} mean w.x = {np.round(outer, 4).tolist()}")
        print(f"      {'|gap|':<16}          = {np.round(gap, 4).tolist()}"
              f"  (largest {gap.max():.4f})")

    print("\n[D] the quadratic rule f(x) = ||x||^2 - 3.5^2 on each dataset")
    for key, (points, labels) in DATASETS.items():
        accuracy = float((quadratic_rule(points) == labels).mean())
        print(f"  Dataset {key}: {accuracy:.2%} of the points correctly separated")

    print("\n[figures]")
    for name in ("fig4_pca.png", "fig5_radius_hist.png"):
        print(f"  saved {FIGURES / name}")
