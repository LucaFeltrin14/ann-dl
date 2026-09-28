"""Single-layer perceptron, written from scratch with NumPy only.

This one implementation is imported, unchanged, by both exercises:

    exercise1_separable.py    separable data   -> the loop stops on its own
    exercise2_overlapping.py  overlapping data -> the loop never stops; the pocket keeps the best weights

Labels are 0/1, so the update rule is the error-driven one

    w <- w + eta * (y - y_hat) * x,        b <- b + eta * (y - y_hat),

whose error (y - y_hat) is 0 on a correct prediction and +1 / -1 on the two kinds of mistake.
(The textbook form w <- w + eta * y * x belongs to -1/+1 labels: with 0/1 labels it would never
update on class 0, so a false positive could never be corrected.)
"""

from dataclasses import dataclass, field

import numpy as np


def step(z):
    """Heaviside step: 1 where z >= 0, 0 otherwise. Works on scalars and arrays alike."""
    return np.where(z >= 0, 1, 0)


@dataclass
class History:
    """What `Perceptron.fit` records, one entry per epoch (a full pass over the data)."""

    accuracy: list = field(default_factory=list)        # accuracy of the current weights at the end of the epoch
    best_accuracy: list = field(default_factory=list)   # pocket (best-so-far) accuracy at the end of the epoch
    updates: list = field(default_factory=list)         # number of updates in the epoch, as (on class 0, on class 1)
    w: list = field(default_factory=list)               # weights at the end of the epoch
    b: list = field(default_factory=list)               # bias at the end of the epoch
    converged: bool = False                             # True if some epoch finished without a single update

    @property
    def epochs(self) -> int:
        """Number of epochs run, including the final update-free one when the loop converged."""
        return len(self.accuracy)


class Perceptron:
    """Perceptron y_hat = step(w . x + b), trained sample by sample with the 0/1 error-driven rule."""

    def __init__(self, w0, b0: float = 0.0, eta: float = 0.01):
        self.w = np.array(w0, dtype=float)  # copied, so the caller's w0 is never modified
        self.b = float(b0)
        self.eta = eta

        # Pocket: the best weights seen so far. Filled by `fit`, which starts it at the initial weights.
        self.pocket_w = self.w.copy()
        self.pocket_b = self.b
        self.pocket_accuracy = -np.inf
        self.pocket_epoch = 0

    # ------------------------------------------------------------------ prediction
    def net_input(self, X):
        """w . x + b for every row of X (or for a single sample)."""
        return X @ self.w + self.b

    def predict(self, X):
        """y_hat = step(w . x + b)."""
        return step(self.net_input(X))

    def accuracy(self, X, y) -> float:
        """Fraction of samples whose prediction matches the label."""
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------ learning
    def update(self, x, y) -> int:
        """Present one sample: predict it, apply the update rule and return the error (y - y_hat).

        A correct prediction gives error 0, so the weights are left untouched.
        """
        error = int(y - self.predict(x))
        if error != 0:
            self.w += self.eta * error * x
            self.b += self.eta * error
        return error

    def fit(self, X, y, max_epochs: int = 100) -> History:
        """Train until a full pass produces no update, or for `max_epochs` passes, whichever comes first.

        The samples are presented in the order they are stored in X, every epoch.
        """
        history = History()

        # The initial weights are the first ones "seen", so they seed the pocket.
        self.pocket_w, self.pocket_b = self.w.copy(), self.b
        self.pocket_accuracy, self.pocket_epoch = self.accuracy(X, y), 0

        for epoch in range(1, max_epochs + 1):
            updates = [0, 0]  # updates triggered by class-0 and by class-1 samples in this epoch
            for x_i, y_i in zip(X, y):
                if self.update(x_i, y_i) != 0:
                    updates[y_i] += 1
                    # Pocket algorithm - the only addition to the plain perceptron loop: after every
                    # update, if the new weights beat every accuracy seen so far, copy them aside.
                    acc = self.accuracy(X, y)
                    if acc > self.pocket_accuracy:
                        self.pocket_w, self.pocket_b = self.w.copy(), self.b
                        self.pocket_accuracy, self.pocket_epoch = acc, epoch

            history.accuracy.append(self.accuracy(X, y))
            history.best_accuracy.append(self.pocket_accuracy)
            history.updates.append(tuple(updates))
            history.w.append(self.w.copy())
            history.b.append(self.b)

            # Stopping rule: an epoch without a single update means every sample is classified
            # correctly, so the weights can no longer change.
            if sum(updates) == 0:
                history.converged = True
                break

        return history
