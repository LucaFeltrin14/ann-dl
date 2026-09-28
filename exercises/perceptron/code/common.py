"""Shared by both exercises: the seed, the data generator, the plotting helpers and an epoch replay.

The perceptron itself lives in perceptron.py; nothing here learns anything.
"""

from pathlib import Path

import numpy as np

from perceptron import Perceptron

# One seed for the whole report: every script builds rng = np.random.default_rng(SEED).
SEED = 42

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = ["#1f77b4", "#ff7f0e"]  # class 0, class 1


def generate_two_classes(rng, mean0, mean1, cov, n_per_class=1000):
    """Draw `n_per_class` 2D points per class from multivariate normals sharing the covariance `cov`.

    The data is returned in generation order - all of class 0, then all of class 1 - and is
    never shuffled, so the perceptron sees the samples in exactly this order in every epoch.
    """
    X0 = rng.multivariate_normal(mean0, cov, size=n_per_class)
    X1 = rng.multivariate_normal(mean1, cov, size=n_per_class)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    return X, y


def scatter_classes(ax, X, y, *, s=12, alpha=0.55):
    """Scatter the two classes on `ax`, one colour each."""
    for k in (0, 1):
        cloud = X[y == k]
        ax.scatter(cloud[:, 0], cloud[:, 1], s=s, alpha=alpha, color=COLORS[k],
                   label=f"Class {k} ({len(cloud)} points)")
    ax.set_xlabel("Feature 1 ($x_1$)")
    ax.set_ylabel("Feature 2 ($x_2$)")
    ax.grid(alpha=0.25, linestyle=":")


def mark_misclassified(ax, X, wrong, label="Misclassified"):
    """Circle the points flagged in the boolean mask `wrong` (the legend shows how many)."""
    ax.scatter(X[wrong, 0], X[wrong, 1], s=46, facecolors="none", edgecolors="black",
               linewidths=0.9, label=f"{label} ({int(wrong.sum())})")


def draw_boundary(ax, w, b, **kwargs):
    """Draw the line w . x + b = 0 across the current axis limits, without changing them."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if abs(w[1]) >= abs(w[0]):  # closer to horizontal: solve for x2
        x1 = np.linspace(*xlim, 400)
        ax.plot(x1, -(w[0] * x1 + b) / w[1], **kwargs)
    else:                       # closer to vertical: solve for x1
        x2 = np.linspace(*ylim, 400)
        ax.plot(-(w[1] * x2 + b) / w[0], x2, **kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def boundary_offset(w, b) -> float:
    """Signed distance from the origin to the line w . x + b = 0, measured along w."""
    return float(-b / np.linalg.norm(w))


def angle_deg(w) -> float:
    """Angle of the weight vector w with the x1 axis, in degrees."""
    return float(np.degrees(np.arctan2(w[1], w[0])))


def replay_epoch(w, b, eta, X, y):
    """Re-run one epoch from (w, b) with Perceptron.update, to see *where* in the epoch things happen.

    `fit` only records the end of each epoch; replaying the epoch from the (w, b) stored at the end of
    the previous one reproduces it exactly (the loop is deterministic) and exposes the inside of it.

    Returns the indices of the samples that triggered an update, the (w, b) held right after the
    class-0 block (first half of the epoch) and the (w, b) at the end of the epoch.
    """
    model = Perceptron(w, b, eta)
    n_class0 = int(np.sum(y == 0))
    mistakes, after_class0 = [], None
    for i, (x_i, y_i) in enumerate(zip(X, y)):
        if i == n_class0:
            after_class0 = (model.w.copy(), model.b)
        if model.update(x_i, y_i) != 0:
            mistakes.append(i)
    return mistakes, after_class0, (model.w.copy(), model.b)


def replay_training(w0, b0, eta, X, y, history):
    """Replay every epoch of a finished `fit`, starting each one from the state stored by `fit`.

    Also checks that each replayed epoch ends exactly where `fit` recorded it did.
    """
    starts = [(np.asarray(w0, dtype=float), float(b0))] + list(zip(history.w, history.b))[:-1]
    epochs = []
    for (w_start, b_start), w_end, b_end in zip(starts, history.w, history.b):
        mistakes, after_class0, end = replay_epoch(w_start, b_start, eta, X, y)
        assert np.allclose(end[0], w_end) and np.isclose(end[1], b_end), "replay diverged from fit"
        epochs.append((mistakes, after_class0, end))
    return epochs
