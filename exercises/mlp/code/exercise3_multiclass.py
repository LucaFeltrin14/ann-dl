"""Exercise 3 - Multi-class, same network.

Trains the MLP of mlp.py - the very same class as Exercise 2, now built with three output units and
output="softmax" - on the 4-feature, 3-class dataset of the statement. Fits the logistic-regression
baseline, writes Figures 4, 5 and 5b, and recovers the six generating clusters to explain the
confusion matrix (item D).

Every number quoted in the report is printed by this script. Run it from the repository root:

    python docs/exercises/mlp/code/exercise3_multiclass.py
"""

import matplotlib

matplotlib.use("Agg")  # no interactive display needed: we only save files
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from common import COLORS, EPOCHS, FIGURES, LEARNING_RATE, MULTICLASS_HIDDEN, SEED, multiclass_data, split
from mlp import MLP

# ---------------------------------------------------------------------------
# Setup: one seed, one generator, used for everything below
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

LAYER_SIZES = [4, *MULTICLASS_HIDDEN, 3]   # 4 inputs, one hidden layer of 16 tanh units, 3 softmax outputs

# ---------------------------------------------------------------------------
# A - Generate the data, with the statement's parameters unchanged
# ---------------------------------------------------------------------------
X, y = multiclass_data()

# make_classification does not say which of the 6 clusters each sample came from, and item D needs it.
# With shuffle=False the same generator returns the same samples, before the final shuffle, in cluster
# order: 250 samples per cluster, cluster k having label k % 3. The shuffle also permutes the columns,
# so each shuffled row is matched to its unshuffled copy by its sorted values.
X_ordered, y_ordered = multiclass_data(shuffle=False)
cluster_of = {tuple(np.sort(row).round(10)): i // 250 for i, row in enumerate(X_ordered)}
assert len(cluster_of) == len(X), "two samples share the same values"
cluster = np.array([cluster_of[tuple(np.sort(row).round(10))] for row in X])
assert np.all(np.bincount(cluster) == 250)
label_of = {tuple(np.sort(row).round(10)): label for row, label in zip(X_ordered, y_ordered)}
assert all(label_of[tuple(np.sort(row).round(10))] == label for row, label in zip(X, y))
flipped = y != cluster % 3  # make_classification flips ~1% of the labels at random (flip_y=0.01)

X_train, X_test, y_train, y_test, c_train, c_test = split(X, y, cluster)

print("=== A - data ===")
print(f"samples: {len(X)}, features: {X.shape[1]}, per class: {np.bincount(y)}; "
      f"train {len(X_train)} {np.bincount(y_train)}, test {len(X_test)} {np.bincount(y_test)}")

# ---------------------------------------------------------------------------
# B - Adapt the network: the same MLP class, three output units and output="softmax"
# ---------------------------------------------------------------------------
model = MLP(LAYER_SIZES, output="softmax", rng=rng)

# ---------------------------------------------------------------------------
# C - Train and evaluate (Figures 4 and 5)
# ---------------------------------------------------------------------------
history = model.fit(X_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE)
mlp_test = model.accuracy(X_test, y_test)
pred = model.predict(X_test)
C = confusion_matrix(y_test, pred)

print("\n=== C - MLP ===")
print(f"architecture {LAYER_SIZES} (tanh hidden, softmax output, categorical cross-entropy), "
      f"{model.n_parameters} parameters; learning rate {LEARNING_RATE}; {EPOCHS} epochs of full-batch GD")
print(f"training loss: {history[0]:.4f} at epoch 0 -> {history[100]:.4f} at epoch 100 -> "
      f"{history[500]:.4f} at epoch 500 -> {history[-1]:.4f} at epoch {EPOCHS}")
print(f"train accuracy {model.accuracy(X_train, y_train):.2%}, TEST ACCURACY {mlp_test:.2%} "
      f"({np.trace(C)} of {C.sum()})")
print("confusion matrix (rows = true class, columns = predicted class):")
print(C)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.arange(EPOCHS + 1), history, color="black", linewidth=1.6,
        label=f"MLP {LAYER_SIZES}, $\\eta$ = {LEARNING_RATE}")
ax.set_title("Figure 4 - Exercise 3: training loss (categorical cross-entropy) per epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss (categorical cross-entropy)")
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3, linestyle=":")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "fig4_loss.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 5.4))
image = ax.imshow(C, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax.text(j, i, C[i, j], ha="center", va="center", fontsize=14,
                color="white" if C[i, j] > C.max() / 2 else "black")
class_names = [f"Class {k}" for k in range(3)]
ax.set_xticks(range(3), class_names)
ax.set_yticks(range(3), class_names)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
fig.colorbar(image, ax=ax, label="Number of test samples")
ax.set_title(f"Figure 5 - Exercise 3: confusion matrix on the {C.sum()} test samples\n"
             f"(MLP {LAYER_SIZES}, test accuracy {mlp_test:.1%})", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIGURES / "fig5_confusion.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# D - Analysis: the most confused pair, the cluster structure behind it, and the linear baseline
# ---------------------------------------------------------------------------
pairs = {(a, b): int(C[a, b] + C[b, a]) for a, b in [(0, 1), (0, 2), (1, 2)]}
worst = max(pairs, key=pairs.get)

print("\n=== D - analysis ===")
print("test samples confused, per class pair (both directions):",
      ", ".join(f"{a}<->{b}: {n}" for (a, b), n in pairs.items()))
print(f"most confused pair: classes {worst[0]} and {worst[1]} ({pairs[worst]} test samples)")
print(f"recall per class: {(np.diag(C) / C.sum(axis=1)).round(3)}")

# The clusters: centroid (sample mean), the sign pattern of its hypercube vertex, and its class.
centroids = np.array([X[cluster == k].mean(axis=0) for k in range(6)])
vertices = np.sign(centroids).astype(int)
print(f"\nlabels flipped by the generator: {flipped.sum()} of {len(y)} ({np.sum(y_test != c_test % 3)} in the test set)")
print("clusters (k: class, vertex, sample centroid, spread = singular values of the centred cluster / sqrt(n)):")
for k in range(6):
    spread = np.linalg.svd(X[cluster == k] - centroids[k], compute_uv=False) / np.sqrt(250)
    print(f"  cluster {k}: class {k % 3}, vertex {vertices[k]}, centroid {centroids[k].round(2)}, "
          f"spread {spread.round(2)}")

print("pairs of clusters of different classes on adjacent vertices (they differ in one coordinate):")
for i in range(6):
    for j in range(i + 1, 6):
        if i % 3 != j % 3 and np.sum(vertices[i] != vertices[j]) == 1:
            d = centroids[j] - centroids[i]
            u = d / np.linalg.norm(d)
            print(f"  cluster {i} (class {i % 3}) - cluster {j} (class {j % 3}): centroids {np.linalg.norm(d):.3f} apart, "
                  f"differ in feature x{np.flatnonzero(vertices[i] != vertices[j])[0] + 1}; std along the joining line "
                  f"{np.std(X[cluster == i] @ u):.3f} and {np.std(X[cluster == j] @ u):.3f}")

print("test samples per cluster, MLP predictions (columns = predicted class 0, 1, 2):")
for k in range(6):
    s = c_test == k
    print(f"  cluster {k} (class {k % 3}): {s.sum():3d} samples, predicted {np.bincount(pred[s], minlength=3)}, "
          f"{np.mean(pred[s] == y_test[s]):.1%} correct")
confused = ((y_test == worst[0]) & (pred == worst[1])) | ((y_test == worst[1]) & (pred == worst[0]))
print(f"origin of the {confused.sum()} test samples confused between classes {worst[0]} and {worst[1]}: "
      f"clusters {np.bincount(c_test[confused], minlength=6)}")


def gaussian_logpdf(X, mean, cov):
    """log N(x; mean, cov) up to the shared constant, for every row of X."""
    d = X - mean
    return -0.5 * np.sum(d * np.linalg.solve(cov, d.T).T, axis=1) - 0.5 * np.linalg.slogdet(cov)[1]


# Reference ceiling: an "oracle" that knows the cluster of every TRAINING sample - which the network is
# never told - fits one Gaussian per cluster on the training set and predicts the class whose two
# clusters make a test sample most likely. The confusions it still makes come from the data itself.
log_lik = np.stack([gaussian_logpdf(X_test, X_train[c_train == k].mean(axis=0), np.cov(X_train[c_train == k].T))
                    for k in range(6)], axis=1)
oracle_pred = np.stack([np.logaddexp(log_lik[:, k], log_lik[:, k + 3]) for k in range(3)], axis=1).argmax(axis=1)
C_oracle = confusion_matrix(y_test, oracle_pred)
print(f"\noracle (one Gaussian per cluster, cluster ids known): test accuracy {np.mean(oracle_pred == y_test):.2%}; "
      "confused per pair: " + ", ".join(f"{a}<->{b}: {C_oracle[a, b] + C_oracle[b, a]}" for a, b in pairs))
print(C_oracle)

# Linear baseline: multinomial logistic regression, one linear score per class.
baseline = LogisticRegression(max_iter=1000).fit(X_train, y_train)
pred_baseline = baseline.predict(X_test)
print(f"\nlogistic-regression baseline: train {baseline.score(X_train, y_train):.2%}, "
      f"TEST ACCURACY {baseline.score(X_test, y_test):.2%}; MLP gain {mlp_test - baseline.score(X_test, y_test):+.2%}")
for k in range(6):
    s = c_test == k
    print(f"  cluster {k} (class {k % 3}): baseline {np.mean(pred_baseline[s] == y_test[s]):5.1%} correct, "
          f"MLP {np.mean(pred[s] == y_test[s]):5.1%}")

# A linear model gives each class one convex region, so a class whose two clusters are joined by a
# segment that passes close to another class's cluster cannot be covered without covering that one too.
print("midpoint between the two clusters of a class: distance to its own centroids vs to the nearest other-class centroid")
for k in range(3):
    midpoint = (centroids[k] + centroids[k + 3]) / 2
    others = [j for j in range(6) if j % 3 != k]
    nearest = min(others, key=lambda j: np.linalg.norm(centroids[j] - midpoint))
    print(f"  class {k} (clusters {k} and {k + 3}): own {np.linalg.norm(centroids[k] - midpoint):.3f}, "
          f"nearest other cluster {nearest} (class {nearest % 3}) at {np.linalg.norm(centroids[nearest] - midpoint):.3f}")

# Figure 5b: the chain of clusters behind the confusion. Cluster 1 (class 1) differs from cluster 0
# (class 0) in one feature and from cluster 2 (class 2) in another; plotting those two features shows
# the three clusters side by side, with the test samples the MLP gets wrong circled.
f_a = np.flatnonzero(vertices[0] != vertices[1])[0]
f_b = np.flatnonzero(vertices[1] != vertices[2])[0]
assert np.sum(vertices[0] != vertices[1]) == 1 and np.sum(vertices[1] != vertices[2]) == 1
fig, ax = plt.subplots(figsize=(7.5, 8))
for k in (0, 1, 2):
    s = cluster == k
    ax.scatter(X[s, f_a], X[s, f_b], s=10, alpha=0.35, color=COLORS[k % 3],
               label=f"Cluster {k} = class {k % 3}, vertex {vertices[k]}")
    ax.scatter(*centroids[k, [f_a, f_b]], s=180, marker="X", color=COLORS[k % 3], edgecolors="black")
in_chain = np.isin(c_test, [0, 1, 2]) & (pred != y_test)
ax.scatter(X_test[in_chain, f_a], X_test[in_chain, f_b], s=60, facecolors="none", edgecolors="black",
           linewidths=1.0, label=f"Test samples of these clusters misclassified by the MLP ({in_chain.sum()})")
ax.scatter([], [], s=120, marker="X", color="lightgray", edgecolors="black", label="Cluster centroid")
ax.set_xlabel(f"Feature $x_{f_a + 1}$ (the one coordinate where clusters 0 and 1 differ)")
ax.set_ylabel(f"Feature $x_{f_b + 1}$ (the one coordinate where clusters 1 and 2 differ)")
ax.set_title("Figure 5b - Clusters 0, 1 and 2: one hypercube edge apart, overlapping", fontsize=11)
ax.grid(alpha=0.25, linestyle=":")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5, framealpha=0.95)
fig.tight_layout()
fig.savefig(FIGURES / "fig5b_clusters.png", dpi=150)
plt.close(fig)
