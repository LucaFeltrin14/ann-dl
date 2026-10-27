"""Exercise 2 - Binary classification a straight line cannot do.

Generates the three-blob dataset of the statement, fits the logistic-regression baseline, trains the
MLP of mlp.py with a sigmoid output and binary cross-entropy, and writes Figures 1, 2 and 3.

Every number quoted in the report is printed by this script. Run it from the repository root:

    python docs/exercises/mlp/code/exercise2_binary.py
"""

import math

import matplotlib

matplotlib.use("Agg")  # no interactive display needed: we only save files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

from common import COLORS, FIGURES, SEED, split
from mlp import MLP

# ---------------------------------------------------------------------------
# Setup: one seed, one generator, used for everything below
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

LAYER_SIZES = [2, 8, 1]   # 2 inputs, one hidden layer of 8 tanh units, 1 sigmoid output
LEARNING_RATE = 0.5
EPOCHS = 2000

# ---------------------------------------------------------------------------
# A - Generate the data (Figure 1), with the statement's parameters unchanged
# ---------------------------------------------------------------------------
centers = np.array([[0., 0.], [3., 3.], [-3., -3.]])
X, c = make_blobs(n_samples=[500, 250, 250], centers=centers,
                  cluster_std=1.2, random_state=42)
y = (c > 0).astype(int)      # class 0: 1 cluster | class 1: 2 clusters

# The blob index c is split alongside, only to read the results per cluster in items B and D.
X_train, X_test, y_train, y_test, c_train, c_test = split(X, y, c)

print("=== A - data ===")
print(f"samples: {len(X)} (class 0: {np.sum(y == 0)}, class 1: {np.sum(y == 1)}); "
      f"train {len(X_train)}, test {len(X_test)}")
for k in range(3):
    print(f"  blob {k} centred at {centers[k]}: {np.sum(c_train == k)} train / {np.sum(c_test == k)} test points")
print(f"class means (train): class 0 {X_train[y_train == 0].mean(axis=0).round(4)}, "
      f"class 1 {X_train[y_train == 1].mean(axis=0).round(4)}")

CLUSTER_NAMES = ["centre blob (0, 0)", "blob (3, 3)", "blob (-3, -3)"]


def scatter_classes(ax, X, y, **kwargs):
    """Scatter the two classes on `ax`, one colour each, with the axis labels every figure needs."""
    for k in (0, 1):
        ax.scatter(X[y == k, 0], X[y == k, 1], color=COLORS[k], label=f"Class {k} ({np.sum(y == k)} points)",
                   **kwargs)
    ax.set_xlabel("Feature 1 ($x_1$)")
    ax.set_ylabel("Feature 2 ($x_2$)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, linestyle=":")


fig, ax = plt.subplots(figsize=(7.5, 7))
scatter_classes(ax, X, y, s=12, alpha=0.6)
ax.set_title("Figure 1 - Exercise 2 data: class 1 is two blobs on opposite sides of class 0")
ax.legend(loc="upper left", framealpha=0.95)
fig.tight_layout()
fig.savefig(FIGURES / "fig1_data.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# B - Baseline: one straight line (logistic regression, allowed here as the baseline only)
# ---------------------------------------------------------------------------
baseline = LogisticRegression().fit(X_train, y_train)
w, b = baseline.coef_[0], baseline.intercept_[0]
baseline_test = baseline.score(X_test, y_test)
p_test = baseline.predict_proba(X_test)[:, 1]
pred_baseline = baseline.predict(X_test)


def phi(t):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))


# Accuracy the fitted line would have on the generating distribution itself (an infinite test set): each
# blob is an isotropic Gaussian, so the share of it on the class-1 side of w.x + b = 0 is a normal CDF.
share_positive = [phi((w @ mu + b) / (1.2 * np.linalg.norm(w))) for mu in centers]
expected_accuracy = 0.5 * (1 - share_positive[0]) + 0.25 * share_positive[1] + 0.25 * share_positive[2]

print("\n=== B - linear baseline ===")
print(f"w = {w.round(4)}, b = {b:.4f}; |w| = {np.linalg.norm(w):.4f}; "
      f"angle of w {np.degrees(np.arctan2(w[1], w[0])):.2f} deg; line offset -b/|w| = {-b / np.linalg.norm(w):.4f}")
print(f"train accuracy {baseline.score(X_train, y_train):.2%}, TEST ACCURACY {baseline_test:.2%} "
      f"({int(round(baseline_test * len(y_test)))} of {len(y_test)})")
print(f"P(class 1) on the test points ranges over [{p_test.min():.4f}, {p_test.max():.4f}]")
for k in range(3):
    s = c_test == k
    print(f"  {CLUSTER_NAMES[k]:<16}: {s.sum():3d} test points, {pred_baseline[s].mean():.0%} predicted class 1, "
          f"{np.sum(pred_baseline[s] == y_test[s])} correct")
print(f"expected accuracy of this line on the generating distribution: {expected_accuracy:.2%}")

# ---------------------------------------------------------------------------
# C - The MLP (mlp.py), trained with full-batch gradient descent (Figures 2 and 3)
# ---------------------------------------------------------------------------
model = MLP(LAYER_SIZES, output="sigmoid", rng=rng)
history = model.fit(X_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE)
mlp_test = model.accuracy(X_test, y_test)
pred_mlp = model.predict(X_test)

print("\n=== C - MLP ===")
print(f"architecture {LAYER_SIZES} (tanh hidden, sigmoid output, binary cross-entropy), "
      f"{model.n_parameters} parameters; learning rate {LEARNING_RATE}; {EPOCHS} epochs of full-batch GD")
print(f"training loss: {history[0]:.4f} at epoch 0 -> {history[100]:.4f} at epoch 100 -> "
      f"{history[500]:.4f} at epoch 500 -> {history[-1]:.4f} at epoch {EPOCHS}")
print(f"train accuracy {model.accuracy(X_train, y_train):.2%}, TEST ACCURACY {mlp_test:.2%} "
      f"({int(round(mlp_test * len(y_test)))} of {len(y_test)})")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.arange(EPOCHS + 1), history, color="black", linewidth=1.6, label=f"MLP {LAYER_SIZES}, $\\eta$ = {LEARNING_RATE}")
ax.set_title("Figure 2 - Exercise 2: training loss (binary cross-entropy) per epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss (binary cross-entropy)")
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3, linestyle=":")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "fig2_loss.png", dpi=150)
plt.close(fig)

# Decision regions: evaluate the network on a grid and shade where P(class 1) >= 0.5.
lim = 7.5
g1, g2 = np.meshgrid(np.linspace(-lim, lim, 601), np.linspace(-lim, lim, 601))
P_grid = model.predict_proba(np.c_[g1.ravel(), g2.ravel()])[:, 0].reshape(g1.shape)

fig, ax = plt.subplots(figsize=(8, 7.5))
ax.contourf(g1, g2, P_grid, levels=[0, 0.5, 1], colors=[COLORS[0], COLORS[1]], alpha=0.18)
ax.contour(g1, g2, P_grid, levels=[0.5], colors="black", linewidths=1.6)
scatter_classes(ax, X_test, y_test, s=22, alpha=0.85, edgecolors="white", linewidths=0.4)
wrong = pred_mlp != y_test
ax.scatter(X_test[wrong, 0], X_test[wrong, 1], s=70, facecolors="none", edgecolors="black", linewidths=1.0)
xs = np.linspace(-lim, lim, 2)  # the baseline's line, solved for x1 since it is nearly vertical
ax.plot(-(w[1] * xs + b) / w[0], xs, color="dimgray", linestyle="--", linewidth=1.4)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
handles, labels = ax.get_legend_handles_labels()
handles += [Patch(color=COLORS[0], alpha=0.3), Patch(color=COLORS[1], alpha=0.3),
            plt.Line2D([], [], color="black", linewidth=1.6),
            plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="black"),
            plt.Line2D([], [], color="dimgray", linestyle="--")]
labels += ["MLP region: class 0", "MLP region: class 1", "MLP boundary P(class 1) = 0.5",
           f"Misclassified by the MLP ({wrong.sum()})", f"Linear baseline boundary ({baseline_test:.1%})"]
ax.legend(handles, labels, loc="upper left", framealpha=0.95, fontsize=8.5)
ax.set_title(f"Figure 3 - MLP decision regions over the {len(X_test)} test points "
             f"(test accuracy {mlp_test:.1%})", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIGURES / "fig3_boundary.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# D - Analysis: the shape of the class-1 region
# ---------------------------------------------------------------------------
# Walk along the diagonal through the three centres, t (1, 1)/sqrt(2), and along the perpendicular
# anti-diagonal t (1, -1)/sqrt(2), and record where the network's prediction changes.
t = np.linspace(-lim * np.sqrt(2), lim * np.sqrt(2), 20001)
on_diagonal = model.predict(np.c_[t, t] / np.sqrt(2))
crossings = t[np.flatnonzero(np.diff(on_diagonal))]
on_antidiagonal = model.predict(np.c_[t, -t] / np.sqrt(2))

print("\n=== D - analysis ===")
print(f"along the diagonal (distance t from the origin) the prediction changes at t = {crossings.round(3)}")
print(f"  i.e. class 1 for t < {crossings[0]:.3f} and t > {crossings[-1]:.3f}: a class-0 band of width "
      f"{crossings[-1] - crossings[0]:.3f} around the anti-diagonal")
print(f"along the anti-diagonal (|t| <= {t.max():.1f}): {on_antidiagonal.mean():.0%} predicted class 1")
print(f"midpoint of the two class-1 centres, (0, 0): predicted class {model.predict(np.zeros((1, 2)))[0]}")

# Reference, not fitted: projected on the diagonal, class 0 is N(0, 1.2^2) with weight 1/2 and each class-1
# blob is N(+-3 sqrt2, 1.2^2) with weight 1/4. The best possible boundary sits where the weighted densities
# cross (the far blob's contribution there is negligible), and its accuracy is the best any model can expect.
d = 3 * np.sqrt(2)
t_bayes = d / 2 + 1.2 ** 2 * np.log(2) / d
bayes_accuracy = 0.5 * (2 * phi(t_bayes / 1.2) - 1) + 0.5 * (1 - phi((t_bayes - d) / 1.2))
print(f"Bayes-optimal boundary: t = +-{t_bayes:.3f} along the diagonal; expected accuracy {bayes_accuracy:.2%}")
for k in range(3):
    s = c_test == k
    print(f"  {CLUSTER_NAMES[k]:<16}: MLP correct on {np.sum(pred_mlp[s] == y_test[s])} of {s.sum()} "
          f"({np.mean(pred_mlp[s] == y_test[s]):.1%})")
