"""Exercise 4 - Does a second hidden layer help?

Same data, split, learning rate, number of epochs and initialization rule as Exercise 3; the only change
is one more hidden layer of the same width. Trains the deeper network with the seed used everywhere else
(item A), then trains both networks - and a one-layer control with the same parameter count as the
deeper one - from 10 seeds (item B), and writes Figure 6.

Every number quoted in the report is printed by this script. Run it from the repository root:

    python docs/exercises/mlp/code/exercise4_depth.py
"""

import matplotlib

matplotlib.use("Agg")  # no interactive display needed: we only save files
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from common import EPOCHS, FIGURES, LEARNING_RATE, MULTICLASS_HIDDEN, SEED, multiclass_data, split
from mlp import MLP

# ---------------------------------------------------------------------------
# Setup: one seed, one generator, used for everything below
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

SHALLOW = [4, *MULTICLASS_HIDDEN, 3]                      # Exercise 3: [4, 16, 3]
DEEP = [4, *MULTICLASS_HIDDEN, *MULTICLASS_HIDDEN, 3]     # Exercise 4: [4, 16, 16, 3], one more layer, same width
WIDE = [4, 50, 3]                                         # control, not asked: one layer, as many parameters as DEEP

# Item B asks for at least 3 seeds. Seed 42 - the one used for every other run in the report - comes
# first, so its runs are exactly the Exercise 3 network and the item-A network. The seed only changes
# the initial weights: the data and the split are the statement's, fixed by random_state=42.
SEEDS = list(range(42, 52))

X, y = multiclass_data()
X_train, X_test, y_train, y_test = split(X, y)

# ---------------------------------------------------------------------------
# A - Train the deeper network: same epochs, same learning rate, same seed as Exercise 3
# ---------------------------------------------------------------------------
deep = MLP(DEEP, output="softmax", rng=rng)
deep_history = deep.fit(X_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE)

print("=== A - deeper network (seed 42) ===")
print(f"architecture {DEEP} (tanh hidden layers, softmax output), {deep.n_parameters} parameters; "
      f"learning rate {LEARNING_RATE}; {EPOCHS} epochs of full-batch GD")
print(f"final training loss {deep_history[-1]:.4f}; train accuracy {deep.accuracy(X_train, y_train):.2%}; "
      f"TEST ACCURACY {deep.accuracy(X_test, y_test):.2%}")

# ---------------------------------------------------------------------------
# B - Controlled comparison over 10 seeds (Figure 6)
# ---------------------------------------------------------------------------
MODELS = {"Exercise 3": SHALLOW, "Exercise 4": DEEP, "Control": WIDE}
runs = {name: [] for name in MODELS}
for name, sizes in MODELS.items():
    for seed in SEEDS:
        model = MLP(sizes, output="softmax", rng=np.random.default_rng(seed))
        history = model.fit(X_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE)
        C = confusion_matrix(y_test, model.predict(X_test))
        confused = {(a, b): C[a, b] + C[b, a] for a, b in [(0, 1), (0, 2), (1, 2)]}
        runs[name].append(dict(history=history, parameters=model.n_parameters,
                               train_loss=history[-1], test_loss=model.loss(X_test, y_test),
                               train_acc=model.accuracy(X_train, y_train), test_acc=model.accuracy(X_test, y_test),
                               most_confused=max(confused, key=confused.get)))

# Seed 42 of the comparison is item A's run: same seed, same code, so the same numbers.
assert np.allclose(runs["Exercise 4"][0]["history"], deep_history)


def column(name, key):
    return np.array([run[key] for run in runs[name]])


print("\n=== B - comparison over seeds", SEEDS, "===")
print("per seed: test accuracy (Exercise 3 | Exercise 4 | control), final training loss (Exercise 3 | Exercise 4)")
for i, seed in enumerate(SEEDS):
    print(f"  seed {seed}: {column('Exercise 3', 'test_acc')[i]:.2%} | {column('Exercise 4', 'test_acc')[i]:.2%} | "
          f"{column('Control', 'test_acc')[i]:.2%}    loss {column('Exercise 3', 'train_loss')[i]:.4f} | "
          f"{column('Exercise 4', 'train_loss')[i]:.4f}")

print("\nmeans over the seeds (+- standard deviation):")
for name, sizes in MODELS.items():
    print(f"  {name:<10} {str(sizes):<14} hidden layers {len(sizes) - 2}, {runs[name][0]['parameters']:3d} parameters: "
          f"final train loss {column(name, 'train_loss').mean():.4f} +- {column(name, 'train_loss').std():.4f}, "
          f"train acc {column(name, 'train_acc').mean():.2%}, "
          f"TEST ACC {column(name, 'test_acc').mean():.2%} +- {column(name, 'test_acc').std():.2%}, "
          f"test loss {column(name, 'test_loss').mean():.4f}, "
          f"train-test gap {column(name, 'train_acc').mean() - column(name, 'test_acc').mean():.2%}")

difference = column("Exercise 4", "test_acc") - column("Exercise 3", "test_acc")
print(f"\ntest accuracy, Exercise 4 minus Exercise 3, paired by seed: mean {difference.mean():+.2%} "
      f"(std {difference.std():.2%}, standard error {difference.std(ddof=1) / np.sqrt(len(SEEDS)):.2%}); "
      f"deeper better in {np.sum(difference > 0)}, equal in {np.sum(difference == 0)}, worse in {np.sum(difference < 0)} "
      f"of {len(SEEDS)} seeds")
control_difference = column("Exercise 4", "test_acc") - column("Control", "test_acc")
print(f"test accuracy, Exercise 4 minus control (same parameter count): mean {control_difference.mean():+.2%}; "
      f"deeper better in {np.sum(control_difference > 0)} of {len(SEEDS)} seeds")
for name in ("Exercise 3", "Exercise 4"):
    curves = np.array([run["history"] for run in runs[name]])
    print(f"mean training loss, {name}: " + ", ".join(f"epoch {e} {curves[:, e].mean():.4f}" for e in (0, 10, 100, 500, EPOCHS)))
    # Full-batch gradient descent with a small enough step never increases the training loss; an epoch
    # where it goes up means the step overshot for the local curvature of the loss surface.
    rises = np.diff(curves, axis=1) > 0
    worst_jump = np.max(curves[:, 1:] / curves[:, :-1])
    print(f"  epochs where the training loss went UP: {rises.sum(axis=1).tolist()} per seed "
          f"({rises.any(axis=1).sum()} of {len(SEEDS)} seeds affected); largest one-epoch jump x{worst_jump:.2f}")
    pairs = [run["most_confused"] for run in runs[name]]
    print(f"  most confused class pair per seed: {pairs}; (0, 1) in {pairs.count((0, 1))} of {len(SEEDS)}")

fig, ax = plt.subplots(figsize=(8.5, 5.5))
epochs = np.arange(EPOCHS + 1)
for name, color in (("Exercise 3", "#1f77b4"), ("Exercise 4", "#d62728")):
    curves = np.array([run["history"] for run in runs[name]])
    for curve in curves:
        ax.plot(epochs, curve, color=color, alpha=0.18, linewidth=0.8)
    ax.plot(epochs, curves.mean(axis=0), color=color, linewidth=2.2,
            label=f"{name}: {MODELS[name]}, {len(MODELS[name]) - 2} hidden layer{'s' if len(MODELS[name]) > 3 else ''} "
                  f"(mean of {len(SEEDS)} seeds)")
ax.plot([], [], color="gray", alpha=0.4, linewidth=0.8, label="Individual seeds")
ax.set_yscale("log")
ax.set_title(f"Figure 6 - Training loss per epoch, one vs two hidden layers ($\\eta$ = {LEARNING_RATE}, same data)",
             fontsize=10.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss (categorical cross-entropy, log scale)")
ax.grid(alpha=0.3, linestyle=":", which="both")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "fig6_depth.png", dpi=150)
plt.close(fig)
