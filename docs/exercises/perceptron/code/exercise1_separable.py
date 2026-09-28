"""Exercise 1 - Separable data: the case the perceptron was designed for.

Generates two well-separated Gaussian classes, trains the perceptron of perceptron.py on them,
repeats the training with eta = 1.0 from the same starting weights, checks that from a zero
start eta only rescales the weights, and writes Figures 1, 2, 2b and 3.

Every number quoted in the report is printed by this script. Run it from the repository root:

    python docs/exercises/perceptron/code/exercise1_separable.py
"""

import matplotlib

matplotlib.use("Agg")  # no interactive display needed: we only save files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from common import (COLORS, FIGURES, SEED, angle_deg, boundary_offset, draw_boundary,
                    generate_two_classes, mark_misclassified, replay_training, scatter_classes)
from perceptron import Perceptron

# ---------------------------------------------------------------------------
# Setup: one seed, one generator, used for everything below
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

# Generating parameters given in the statement
MEAN_0 = [1.5, 1.5]
MEAN_1 = [5.0, 5.0]
COV = [[0.5, 0.0], [0.0, 0.5]]
N_PER_CLASS = 1000

ETA = 0.01          # learning rate of items B and C
ETA_LARGE = 1.0     # learning rate of the re-run in item D
MAX_EPOCHS = 100

# ---------------------------------------------------------------------------
# A - Generate the data (Figure 1)
# ---------------------------------------------------------------------------
X, y = generate_two_classes(rng, MEAN_0, MEAN_1, COV, N_PER_CLASS)

fig, ax = plt.subplots(figsize=(8, 6.5))
scatter_classes(ax, X, y)
ax.set_title("Figure 1 - Exercise 1 data: two separable Gaussian classes (2000 points)")
ax.set_aspect("equal")
ax.legend(loc="upper left", framealpha=0.95)
fig.tight_layout()
fig.savefig(FIGURES / "fig1_data.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# B - The perceptron (perceptron.py) and its initial weights
# w0 is drawn ONCE and reused by the eta = 1.0 run of item D, which must change nothing but eta.
# ---------------------------------------------------------------------------
w0 = rng.normal(0, 0.01, size=2)
initial_accuracy = Perceptron(w0, 0.0, ETA).accuracy(X, y)

# ---------------------------------------------------------------------------
# C - Train and measure (Figures 2 and 3)
# ---------------------------------------------------------------------------
model = Perceptron(w0, b0=0.0, eta=ETA)
history = model.fit(X, y, max_epochs=MAX_EPOCHS)
wrong = model.predict(X) != y

fig, ax = plt.subplots(figsize=(8, 6.5))
scatter_classes(ax, X, y)
ax.set_aspect("equal")
draw_boundary(ax, model.w, model.b, color="black", linewidth=2,
              label=r"Decision boundary $\mathbf{w}\cdot\mathbf{x} + b = 0$")
mark_misclassified(ax, X, wrong)
ax.set_title(f"Figure 2 - Perceptron decision boundary ($\\eta = {ETA}$, "
             f"{history.epochs} epochs, accuracy {model.accuracy(X, y):.2%})")
ax.legend(loc="upper left", framealpha=0.95)
fig.tight_layout()
fig.savefig(FIGURES / "fig2_boundary.png", dpi=150)
plt.close(fig)

epochs = np.arange(0, history.epochs + 1)
accuracy_curve = [initial_accuracy] + history.accuracy          # epoch 0 = initial weights
updates = np.array(history.updates)                             # (epochs, 2): class 0, class 1

fig, (ax_acc, ax_upd) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 2]})
ax_acc.plot(epochs, np.array(accuracy_curve) * 100, marker="o", markersize=4, color="black",
            label="Accuracy on the full dataset (end of epoch)")
ax_acc.axhline(100, color="grey", linestyle=":", linewidth=1, label="100%")
ax_acc.set_ylabel("Accuracy (%)")
ax_acc.set_title(f"Figure 3 - Accuracy $\\times$ epoch ($\\eta = {ETA}$); epoch 0 = initial weights")
ax_acc.grid(alpha=0.25, linestyle=":")
ax_acc.legend(loc="lower right")
ax_upd.bar(epochs[1:], updates[:, 0], color=COLORS[0], label="Updates triggered by class 0")
ax_upd.bar(epochs[1:], updates[:, 1], bottom=updates[:, 0], color=COLORS[1],
           label="Updates triggered by class 1")
ax_upd.set_xlabel("Epoch")
ax_upd.set_ylabel("Updates in the epoch")
ax_upd.yaxis.set_major_locator(MaxNLocator(integer=True))
ax_upd.grid(alpha=0.25, linestyle=":")
ax_upd.legend(loc="upper right")
fig.tight_layout()
fig.savefig(FIGURES / "fig3_accuracy.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# D - Analysis: where in the epoch the updates happen, and how far the bias has to walk
# ---------------------------------------------------------------------------
replayed = replay_training(w0, 0.0, ETA, X, y, history)
mistake_positions = [i for mistakes, _, _ in replayed for i in mistakes]
block_starts = {0, N_PER_CLASS}                              # first sample of each class block
n_on_block_start = sum(i in block_starts for i in mistake_positions)
# Every update moves b by exactly +eta (class-1 mistake) or -eta (class-0 mistake), so
# b_final = b0 + eta * (updates on class 1 - updates on class 0).
b_walk = ETA * (updates[:, 1].sum() - updates[:, 0].sum())

# ---------------------------------------------------------------------------
# D - Analysis: the same training with eta = 1.0, and the zero start
# ---------------------------------------------------------------------------
model_large = Perceptron(w0, b0=0.0, eta=ETA_LARGE)       # same w0, same b0, same data order
history_large = model_large.fit(X, y, max_epochs=MAX_EPOCHS)

direction = model.w / np.linalg.norm(model.w)
direction_large = model_large.w / np.linalg.norm(model_large.w)
angle_between = float(np.degrees(np.arccos(np.clip(direction @ direction_large, -1, 1))))

# From w = 0, b = 0 the claim is that eta only rescales the whole trajectory.
zero_small = Perceptron(np.zeros(2), 0.0, ETA)
zero_large = Perceptron(np.zeros(2), 0.0, ETA_LARGE)
history_zero_small = zero_small.fit(X, y, max_epochs=MAX_EPOCHS)
history_zero_large = zero_large.fit(X, y, max_epochs=MAX_EPOCHS)
ratio = ETA_LARGE / ETA
same_trajectory = all(
    np.allclose(w_l, ratio * w_s) and np.isclose(b_l, ratio * b_s)
    for w_s, w_l, b_s, b_l in zip(history_zero_small.w, history_zero_large.w,
                                  history_zero_small.b, history_zero_large.b)
)

# Figure 2b: both learned boundaries, zoomed on the gap between the classes
fig, ax = plt.subplots(figsize=(8, 6.5))
scatter_classes(ax, X, y)
ax.set_xlim(0.5, 6.0)
ax.set_ylim(0.5, 6.0)
ax.set_aspect("equal")
draw_boundary(ax, model.w, model.b, color="black", linewidth=2,
              label=f"$\\eta = {ETA}$: {history.epochs} epochs")
draw_boundary(ax, model_large.w, model_large.b, color="#d62728", linewidth=2, linestyle="--",
              label=f"$\\eta = {ETA_LARGE}$: {history_large.epochs} epochs")
ax.set_title("Figure 2b - Both learned boundaries, same data and same $\\mathbf{w}_0$ (zoom on the gap)")
ax.legend(loc="upper left", framealpha=0.95)
fig.tight_layout()
fig.savefig(FIGURES / "fig2b_eta_boundaries.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Reported numbers
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    norms = np.linalg.norm(X, axis=1)

    print("=" * 78)
    print("EXERCISE 1 - SEPARABLE DATA")
    print(f"seed = {SEED} | 2 classes x {N_PER_CLASS} samples | order: all class 0, then all class 1")
    print("=" * 78)

    print("\n[A] generating parameters and sample statistics")
    for k, mean in enumerate((MEAN_0, MEAN_1)):
        cloud = X[y == k]
        print(f"  class {k}: mean={mean} cov={COV} | sample mean={cloud.mean(axis=0)}"
              f" sample std={cloud.std(axis=0)}")
    print(f"  ||x|| over the dataset: mean {norms.mean():.4f}, min {norms.min():.4f}, max {norms.max():.4f}")

    print("\n[B] initialisation")
    print(f"  w0 = {w0}  (||w0|| = {np.linalg.norm(w0):.5f}),  b0 = 0.0")
    print(f"  accuracy of the initial weights = {initial_accuracy:.4f}")

    print(f"\n[C] training with eta = {ETA}")
    print(f"  final w = {model.w},  final b = {model.b:.4f}")
    print(f"  epochs = {history.epochs} (converged = {history.converged}; "
          f"last update in epoch {history.epochs - 1}, epoch {history.epochs} is the update-free pass)")
    print(f"  final accuracy = {model.accuracy(X, y):.4f}  ({int(wrong.sum())} misclassified)")
    print(f"  accuracy per epoch = {np.round(history.accuracy, 4).tolist()}")

    print("\n[D] updates per epoch (class 0, class 1) and ||w|| at the end of each epoch")
    for e, ((u0, u1), w_e) in enumerate(zip(history.updates, history.w), start=1):
        print(f"  epoch {e:>2}: {u0} + {u1} = {u0 + u1} updates | ||w|| = {np.linalg.norm(w_e):.4f}")
    print(f"  total updates = {int(updates.sum())} "
          f"(class 0: {int(updates[:, 0].sum())}, class 1: {int(updates[:, 1].sum())}) "
          f"over {history.epochs * len(X)} sample presentations "
          f"({1 - updates.sum() / (history.epochs * len(X)):.2%} of the presentations change nothing)")
    print(f"  samples that ever triggered an update: {sorted(set(mistake_positions))}")
    print(f"  updates triggered by the FIRST sample of a class block (index 0 or {N_PER_CLASS}): "
          f"{n_on_block_start} of {len(mistake_positions)}")
    print(f"  bias walk: eta * (class-1 updates - class-0 updates) = {b_walk:.4f} = final b ({model.b:.4f})")
    print(f"  per mistake: |delta b| = eta = {ETA}, ||delta w|| = eta * ||x|| = {ETA * norms.mean():.4f} on average")
    print(f"  offset the boundary needs, -b/||w||, at the end = {boundary_offset(model.w, model.b):.4f} "
          f"(midpoint of the class means along w: "
          f"{float(((np.array(MEAN_0) + np.array(MEAN_1)) / 2) @ model.w / np.linalg.norm(model.w)):.4f})")

    print(f"\n[D] eta = {ETA} against eta = {ETA_LARGE} (same data, same w0, same order)")
    for name, m, h in (("eta=0.01", model, history), ("eta=1.0 ", model_large, history_large)):
        d = m.w / np.linalg.norm(m.w)
        print(f"  {name}: w = {m.w}, b = {m.b:.4f}, epochs = {h.epochs}, accuracy = {m.accuracy(X, y):.4f}"
              f" | w/||w|| = {d}, angle = {angle_deg(m.w):.3f} deg,"
              f" boundary offset -b/||w|| = {boundary_offset(m.w, m.b):.4f}")
    print(f"  angle between the two directions = {angle_between:.4f} deg")
    print(f"  offset gap between the two boundaries = "
          f"{abs(boundary_offset(model.w, model.b) - boundary_offset(model_large.w, model_large.b)):.4f}")
    print(f"  ||w0|| / eta: eta=0.01 -> {np.linalg.norm(w0) / ETA:.4f}, "
          f"eta=1.0 -> {np.linalg.norm(w0) / ETA_LARGE:.5f}  (one update moves w by ||x||, mean {norms.mean():.2f})")

    print("\n[D] zero start w = 0, b = 0")
    for name, m, h in (("eta=0.01", zero_small, history_zero_small),
                       ("eta=1.0 ", zero_large, history_zero_large)):
        print(f"  {name}: w = {m.w}, b = {m.b:.4f}, epochs = {h.epochs}, accuracy = {m.accuracy(X, y):.4f}")
    print(f"  w(eta=1.0) / w(eta=0.01) = {zero_large.w / zero_small.w}, "
          f"b ratio = {zero_large.b / zero_small.b:.4f}")
    print(f"  every end-of-epoch (w, b) of the eta=1.0 run equals {ratio:.0f} x the eta=0.01 one: {same_trajectory}")
    print(f"  identical predictions on all 2000 points: "
          f"{bool(np.array_equal(zero_small.predict(X), zero_large.predict(X)))}")
    print(f"  eta=1.0 from w0 minus eta=1.0 from zero: w diff = {model_large.w - zero_large.w} "
          f"(w0 = {w0}), b diff = {model_large.b - zero_large.b:.4f}, epochs {history_large.epochs} vs "
          f"{history_zero_large.epochs} -> same mistakes: "
          f"{bool(np.allclose(model_large.w - zero_large.w, w0) and np.isclose(model_large.b, zero_large.b))}")

    print("\n[figures]")
    for name in ("fig1_data.png", "fig2_boundary.png", "fig2b_eta_boundaries.png", "fig3_accuracy.png"):
        print(f"  saved {FIGURES / name}")
