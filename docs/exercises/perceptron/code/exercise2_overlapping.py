"""Exercise 2 - Overlapping data: the case the perceptron cannot solve.

Generates two heavily overlapping Gaussian classes, trains the SAME perceptron of perceptron.py
(eta = 0.01, 100 epochs), compares the final weights with the pocket (best-so-far) weights,
measures why the final ones land where they do, and writes Figures 4, 5, 6 and 6b.

Every number quoted in the report is printed by this script. Run it from the repository root:

    python docs/exercises/perceptron/code/exercise2_overlapping.py

The checks of item D (1000 epochs, eta = 0.001, shuffled orders) take about 10 seconds in total.
"""

from math import erf, sqrt

import matplotlib

matplotlib.use("Agg")  # no interactive display needed: we only save files
import matplotlib.pyplot as plt
import numpy as np

from common import (FIGURES, SEED, angle_deg, boundary_offset, draw_boundary, generate_two_classes,
                    mark_misclassified, replay_training, scatter_classes)
from perceptron import Perceptron, step

# ---------------------------------------------------------------------------
# Setup: one seed, one generator, used for everything below
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

# Generating parameters given in the statement
MEAN_0 = [3.0, 3.0]
MEAN_1 = [4.0, 4.0]
COV = [[1.5, 0.0], [0.0, 1.5]]
N_PER_CLASS = 1000

ETA = 0.01
MAX_EPOCHS = 100

FINAL_STYLE = dict(color="#d62728", linewidth=2.2)                   # final weights: solid red
POCKET_STYLE = dict(color="#2ca02c", linewidth=2.2, linestyle="--")  # pocket weights: dashed green

# ---------------------------------------------------------------------------
# A - Generate the data (Figure 4)
# ---------------------------------------------------------------------------
X, y = generate_two_classes(rng, MEAN_0, MEAN_1, COV, N_PER_CLASS)

fig, ax = plt.subplots(figsize=(8, 6.5))
scatter_classes(ax, X, y, alpha=0.45)
ax.set_title("Figure 4 - Exercise 2 data: two overlapping Gaussian classes (2000 points)")
ax.set_aspect("equal")
ax.legend(loc="upper left", framealpha=0.95)
fig.tight_layout()
fig.savefig(FIGURES / "fig4_data.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# B - Train with the Exercise 1 implementation, unchanged; the pocket is tracked inside `fit`
# ---------------------------------------------------------------------------
w0 = rng.normal(0, 0.01, size=2)
initial_accuracy = Perceptron(w0, 0.0, ETA).accuracy(X, y)

model = Perceptron(w0, b0=0.0, eta=ETA)
history = model.fit(X, y, max_epochs=MAX_EPOCHS)

final_w, final_b = model.w.copy(), model.b
pocket_w, pocket_b = model.pocket_w.copy(), model.pocket_b
pocket_model = Perceptron(pocket_w, pocket_b, ETA)       # only used to predict with the pocket weights
final_accuracy = model.accuracy(X, y)
pocket_accuracy = pocket_model.accuracy(X, y)
wrong_final = model.predict(X) != y
wrong_pocket = pocket_model.predict(X) != y

# Reference only - nothing is fitted: for two Gaussians with equal covariance the Bayes-optimal
# boundary is the perpendicular bisector of the means, x1 + x2 = 7.
bayes_w, bayes_b = np.array([1.0, 1.0]), -7.0
bayes_accuracy = float(np.mean(step(X @ bayes_w + bayes_b) == y))
# Its expected accuracy is Phi(||mu1 - mu0|| / (2 sigma)) = Phi(sqrt(2) / (2 sqrt(1.5))).
separation = np.linalg.norm(np.array(MEAN_1) - np.array(MEAN_0)) / (2 * sqrt(COV[0][0]))
bayes_expected = 0.5 * (1 + erf(separation / sqrt(2)))

# ---------------------------------------------------------------------------
# C - Figures 5 and 6
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
panels = (
    ("final", final_w, final_b, wrong_final, final_accuracy),
    ("pocket", pocket_w, pocket_b, wrong_pocket, pocket_accuracy),
)
for ax, (name, w, b, wrong, acc) in zip(axes, panels):
    scatter_classes(ax, X, y, s=10, alpha=0.35)
    ax.set_aspect("equal")
    draw_boundary(ax, final_w, final_b, label=f"Final weights ({final_accuracy:.2%})", **FINAL_STYLE)
    draw_boundary(ax, pocket_w, pocket_b, label=f"Pocket weights ({pocket_accuracy:.2%})", **POCKET_STYLE)
    mark_misclassified(ax, X, wrong, label=f"Misclassified by the {name} weights")
    ax.set_title(f"Misclassified points marked for the {name} weights (accuracy {acc:.2%})")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
fig.suptitle("Figure 5 - Final and pocket decision boundaries on the overlapping data", fontsize=14)
fig.tight_layout()
fig.savefig(FIGURES / "fig5_boundaries.png", dpi=150)
plt.close(fig)

epochs = np.arange(0, history.epochs + 1)
current_curve = np.array([initial_accuracy] + history.accuracy) * 100   # epoch 0 = initial weights
best_curve = np.array([initial_accuracy] + history.best_accuracy) * 100

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs, current_curve, color="#d62728", linewidth=1.3, marker="o", markersize=2.5,
        label="Current weights (end of each epoch)")
ax.plot(epochs, best_curve, color="#2ca02c", linewidth=2.2, drawstyle="steps-post",
        label="Best so far (pocket)")
ax.axhline(bayes_accuracy * 100, color="black", linestyle=":", linewidth=1.2,
           label=f"Reference: Bayes line $x_1 + x_2 = 7$ ({bayes_accuracy:.2%})")
ax.axhline(50, color="grey", linestyle="--", linewidth=1, label="Chance level (50%)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy on the full dataset (%)")
ax.set_title(f"Figure 6 - Accuracy $\\times$ epoch on the overlapping data ($\\eta = {ETA}$)")
ax.set_ylim(40, 80)
ax.grid(alpha=0.25, linestyle=":")
ax.legend(loc="lower right", framealpha=0.95)
fig.tight_layout()
fig.savefig(FIGURES / "fig6_accuracy.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# D - Analysis: where the final boundary sits, and why the loop leaves it there
# ---------------------------------------------------------------------------
norms = np.linalg.norm(X, axis=1)
centre = X.mean(axis=0)
updates = np.array(history.updates)                        # (epochs, 2): class 0, class 1
w_norms = np.linalg.norm(np.array(history.w), axis=1)

# Every update moves b by exactly +eta or -eta, so b_final = b0 + eta * (class-1 - class-0 updates).
b_walk = ETA * (updates[:, 1].sum() - updates[:, 0].sum())

# One mistake on x shifts the score w.x' + b of EVERY point x' by eta * e * (x . x' + 1).
# Compare the size of that shift with how spread out the scores of the data are.
score_jump = ETA * float(np.mean(X @ centre + 1))           # shift of a typical point, for a typical x
final_scores = X @ final_w + final_b

# Inside each epoch: the state right after the class-0 block against the state at its end.
replayed = replay_training(w0, 0.0, ETA, X, y, history)
mid_pred1 = np.array([np.mean(step(X @ w + b)) for _, (w, b), _ in replayed])
end_pred1 = np.array([np.mean(step(X @ w + b)) for _, _, (w, b) in replayed])
mid_acc = np.array([np.mean(step(X @ w + b) == y) for _, (w, b), _ in replayed])
mistake_positions = [i for mistakes, _, _ in replayed for i in mistakes]
n_on_block_start = sum(i in (0, N_PER_CLASS) for i in mistake_positions)


def run(eta, max_epochs, X_run=X, y_run=y):
    """Train a fresh copy of the same perceptron from the same w0; used by the checks below."""
    m = Perceptron(w0, 0.0, eta)
    h = m.fit(X_run, y_run, max_epochs=max_epochs)
    return m, h


# More epochs? A smaller eta? Both, from the same w0 and the same data.
checks = [(ETA, MAX_EPOCHS, model, history)]
for eta, max_epochs in ((ETA, 1000), (0.001, MAX_EPOCHS), (0.001, 1000)):
    checks.append((eta, max_epochs, *run(eta, max_epochs)))

# Is the ~50% an artefact of presenting the classes in two blocks? Shuffle and look again.
# (Drawn from the same rng AFTER everything above, so none of the reported numbers move.)
perm = rng.permutation(len(X))
shuffled_once, history_shuffled_once = run(ETA, MAX_EPOCHS, X[perm], y[perm])

reshuffled = Perceptron(w0, 0.0, ETA)            # a fresh permutation every epoch
reshuffled_best, reshuffled_stopped = -np.inf, False
for _ in range(MAX_EPOCHS):
    p = rng.permutation(len(X))
    h = reshuffled.fit(X[p], y[p], max_epochs=1)      # one epoch in the new order
    reshuffled_best = max(reshuffled_best, reshuffled.pocket_accuracy)
    if h.converged:
        reshuffled_stopped = True
        break

# Figure 6b: what the flat red curve of Figure 6 hides - the predictions flip every half epoch
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs[1:], end_pred1 * 100, color="#ff7f0e", marker="o", markersize=2.5, linewidth=1.3,
        label="At the end of the epoch (after the class-1 block)")
ax.plot(epochs[1:], mid_pred1 * 100, color="#1f77b4", marker="o", markersize=2.5, linewidth=1.3,
        label="Halfway through the epoch (after the class-0 block)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Points predicted as class 1 (%)")
ax.set_ylim(-5, 105)
ax.set_title("Figure 6b - Share of the 2000 points predicted as class 1, inside each epoch")
ax.grid(alpha=0.25, linestyle=":")
ax.legend(loc="center right", framealpha=0.95)
fig.tight_layout()
fig.savefig(FIGURES / "fig6b_flip.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Reported numbers
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 78)
    print("EXERCISE 2 - OVERLAPPING DATA")
    print(f"seed = {SEED} | 2 classes x {N_PER_CLASS} samples | order: all class 0, then all class 1")
    print("=" * 78)

    print("\n[A] generating parameters and sample statistics")
    for k, mean in enumerate((MEAN_0, MEAN_1)):
        cloud = X[y == k]
        print(f"  class {k}: mean={mean} cov={COV} | sample mean={cloud.mean(axis=0)}"
              f" sample std={cloud.std(axis=0)}")
    print(f"  ||x|| over the dataset: mean {norms.mean():.4f}, min {norms.min():.4f}, max {norms.max():.4f}")

    print("\n[B] training with the pocket")
    print(f"  w0 = {w0}, b0 = 0.0, accuracy of the initial weights = {initial_accuracy:.4f}")
    print(f"  epochs run = {history.epochs} (converged = {history.converged}); "
          f"updates in the last epoch = {sum(history.updates[-1])}, "
          f"fewest updates in any epoch = {int(updates.sum(axis=1).min())}")
    print(f"  FINAL  w = {final_w}, b = {final_b:.4f}, accuracy = {final_accuracy:.4f}"
          f" ({int(wrong_final.sum())} misclassified)")
    print(f"  POCKET w = {pocket_w}, b = {pocket_b:.4f}, accuracy = {pocket_accuracy:.4f}"
          f" ({int(wrong_pocket.sum())} misclassified), found in epoch {model.pocket_epoch}")
    print(f"  reference: Bayes line x1 + x2 = 7 -> accuracy {bayes_accuracy:.4f} on this sample, "
          f"{bayes_expected:.4f} expected")

    print("\n[C] accuracy per epoch")
    print(f"  current: min {min(history.accuracy):.4f}, max {max(history.accuracy):.4f}, "
          f"mean {np.mean(history.accuracy):.4f}")
    print(f"  current, first 10 epochs: {np.round(history.accuracy[:10], 4).tolist()}")
    print(f"  current, last 10 epochs:  {np.round(history.accuracy[-10:], 4).tolist()}")
    print(f"  best so far at the end of epochs 1, 10, 50, 100: "
          f"{[round(history.best_accuracy[e - 1], 4) for e in (1, 10, 50, 100)]}")

    print("\n[D] where the boundaries sit (offset = -b/||w||, distance from the origin along w)")
    for name, w, b in (("final ", final_w, final_b), ("pocket", pocket_w, pocket_b), ("bayes ", bayes_w, bayes_b)):
        print(f"  {name}: angle of w = {angle_deg(w):7.3f} deg, ||w|| = {np.linalg.norm(w):.4f}, "
              f"offset = {boundary_offset(w, b):.4f}, data centre along w = "
              f"{float(centre @ w / np.linalg.norm(w)):.4f}, predicted class 1 = {np.mean(step(X @ w + b)):.4f}")

    print("\n[D] how far one mistake moves b against w")
    print(f"  |delta b| = eta = {ETA};  ||delta w|| = eta * ||x|| = {ETA * norms.mean():.4f} on average "
          f"(ratio {norms.mean():.2f})")
    print(f"  bias walk: eta * (class-1 updates {int(updates[:, 1].sum())} - class-0 updates "
          f"{int(updates[:, 0].sum())}) = {b_walk:.4f} = final b ({final_b:.4f})")
    print(f"  ||w|| at the end of the epochs: min {w_norms.min():.4f}, max {w_norms.max():.4f} "
          f"(started at {np.linalg.norm(w0):.4f})")
    print(f"  bias needed to put a line with the final w through the data centre: "
          f"{-float(centre @ final_w):.4f} (actual b = {final_b:.4f})")
    print(f"  score shift of a typical point after ONE mistake: eta*(x.x'+1) = {score_jump:.4f}; "
          f"spread (std) of the final scores over the data = {final_scores.std():.4f}")

    print("\n[D] inside the epochs (replayed from the states recorded by fit)")
    print(f"  total updates = {len(mistake_positions)} in {history.epochs} epochs; "
          f"{n_on_block_start} of them on the FIRST sample of a class block (index 0 or {N_PER_CLASS})")
    print(f"  updates per epoch: min {int(updates.sum(axis=1).min())}, max {int(updates.sum(axis=1).max())}, "
          f"mean {updates.sum(axis=1).mean():.2f}")
    print(f"  right after the class-0 block: predicted class 1 = {mid_pred1.mean():.4f} on average "
          f"(max {mid_pred1.max():.4f}), accuracy {mid_acc.mean():.4f} on average")
    print(f"  at the end of the epoch:       predicted class 1 = {end_pred1.mean():.4f} on average "
          f"(min {end_pred1.min():.4f}), accuracy {np.mean(history.accuracy):.4f} on average")
    last_mistakes, (w_mid, b_mid), _ = replayed[-1]
    print(f"  last epoch: updates at samples {last_mistakes}; after the class-0 block w = {w_mid}, "
          f"b = {b_mid:.4f}, offset = {boundary_offset(w_mid, b_mid):.4f}, "
          f"predicted class 1 = {np.mean(step(X @ w_mid + b_mid)):.4f}")

    print("\n[D] more epochs / smaller eta (same data, same w0)")
    print("| eta | epochs | final accuracy | pocket accuracy | updates in the last epoch |")
    print("|---|---|---|---|---|")
    for eta, max_epochs, m, h in checks:
        print(f"| {eta} | {h.epochs} | {m.accuracy(X, y):.4f} | {m.pocket_accuracy:.4f} "
              f"| {sum(h.updates[-1])} |")

    print("\n[D] sample order (same data, same w0, eta = 0.01, 100 epochs)")
    print(f"  shuffled once:        final {shuffled_once.accuracy(X, y):.4f}, pocket "
          f"{shuffled_once.pocket_accuracy:.4f}, converged = {history_shuffled_once.converged}, "
          f"current accuracy min/max over epochs {min(history_shuffled_once.accuracy):.4f}/"
          f"{max(history_shuffled_once.accuracy):.4f}")
    print(f"  reshuffled per epoch: final {reshuffled.accuracy(X, y):.4f}, best seen "
          f"{reshuffled_best:.4f}, converged = {reshuffled_stopped}")

    print("\n[figures]")
    for name in ("fig4_data.png", "fig5_boundaries.png", "fig6_accuracy.png", "fig6b_flip.png"):
        print(f"  saved {FIGURES / name}")
