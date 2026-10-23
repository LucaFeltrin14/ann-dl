"""Checks, not part of any exercise's method: they only confirm that the hand work and the code are right.

1. Exercise 1 was worked by hand in the report. This recomputes it with NumPy and asserts that every value
   typed in the report (8 decimals) matches.
2. The backward pass of mlp.py was derived by hand. This compares its gradients with central finite
   differences of the loss, for both output/loss pairs and for one and two hidden layers.

Run it from the repository root:

    python docs/exercises/mlp/code/gradient_check.py
"""

import numpy as np

from common import SEED
from mlp import MLP

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1 - Exercise 1, recomputed
# ---------------------------------------------------------------------------
x, y = np.array([0.5, -0.2]), 1.0
W1, b1 = np.array([[0.3, -0.1], [0.2, 0.4]]), np.array([0.1, -0.2])
W2, b2 = np.array([0.5, -0.3]), 0.2
eta = 0.3

z1 = W1 @ x + b1
h1 = np.tanh(z1)
u2 = W2 @ h1 + b2
y_hat = np.tanh(u2)
L = (y - y_hat) ** 2

dL_dyhat = -2 * (y - y_hat)
dL_du2 = dL_dyhat * (1 - y_hat ** 2)
dL_dW2, dL_db2 = dL_du2 * h1, dL_du2
dL_dh1 = dL_du2 * W2
dL_dz1 = dL_dh1 * (1 - h1 ** 2)
dL_dW1, dL_db1 = np.outer(dL_dz1, x), dL_dz1

W1_new, b1_new = W1 - eta * dL_dW1, b1 - eta * dL_db1
W2_new, b2_new = W2 - eta * dL_dW2, b2 - eta * dL_db2
y_hat_2 = np.tanh(W2_new @ np.tanh(W1_new @ x + b1_new) + b2_new)
L_2 = (y - y_hat_2) ** 2
squared_gradient_norm = sum(np.sum(g ** 2) for g in (dL_dW1, dL_db1, dL_dW2, dL_db2))

# The values written in the report, rounded to 8 decimals.
REPORT = {
    "z1": (z1, [0.27, -0.18]),
    "h1": (h1, [0.26362484, -0.17808087]),
    "u2": (u2, 0.38523668),
    "y_hat": (y_hat, 0.36724656),
    "L": (L, 0.40037691),
    "dL/dy_hat": (dL_dyhat, -1.26550687),
    "dL/du2": (dL_du2, -1.09482791),
    "dL/dW2": (dL_dW2, [-0.28862383, 0.19496791]),
    "dL/db2": (dL_db2, -1.09482791),
    "dL/dh1": (dL_dh1, [-0.54741396, 0.32844837]),
    "dL/dz1": (dL_dz1, [-0.50936975, 0.31803236]),
    "dL/dW1": (dL_dW1, [[-0.25468488, 0.10187395], [0.15901618, -0.06360647]]),
    "dL/db1": (dL_db1, [-0.50936975, 0.31803236]),
    "W1 updated": (W1_new, [[0.37640546, -0.13056219], [0.15229515, 0.41908194]]),
    "b1 updated": (b1_new, [0.25281093, -0.29540971]),
    "W2 updated": (W2_new, [0.58658715, -0.35849037]),
    "b2 updated": (b2_new, 0.52844837),
    "y_hat, 2nd pass": (y_hat_2, 0.71118053),
    "L, 2nd pass": (L_2, 0.08341669),
    "|grad|^2": (squared_gradient_norm, 1.78514110),
}
print("1 - Exercise 1: NumPy against the hand-worked values of the report")
for name, (computed, reported) in REPORT.items():
    assert np.allclose(computed, reported, rtol=0, atol=1e-8), f"{name}: computed {computed}, report {reported}"
    print(f"  {name:<16} {np.round(computed, 8)}  ok")

# ---------------------------------------------------------------------------
# 2 - mlp.py backward pass against central finite differences
# ---------------------------------------------------------------------------
print("\n2 - mlp.py: analytic gradients against central finite differences of the loss")
X = rng.normal(size=(20, 3))
STEP = 1e-6
for sizes, output in [([3, 5, 1], "sigmoid"), ([3, 5, 4, 1], "sigmoid"),
                      ([3, 5, 3], "softmax"), ([3, 5, 4, 3], "softmax")]:
    labels = rng.integers(0, sizes[-1] if output == "softmax" else 2, size=len(X))
    model = MLP(sizes, output=output, rng=rng)
    grads_W, grads_b = model.backward(model.forward(X), model.targets(labels))
    worst = 0.0
    for params, grads in ((model.W, grads_W), (model.b, grads_b)):
        for p, g in zip(params, grads):
            for i in np.ndindex(p.shape):
                original = p[i]
                p[i] = original + STEP
                loss_plus = model.loss(X, labels)
                p[i] = original - STEP
                loss_minus = model.loss(X, labels)
                p[i] = original
                numeric = (loss_plus - loss_minus) / (2 * STEP)
                worst = max(worst, abs(numeric - g[i]) / max(abs(numeric) + abs(g[i]), 1e-12))
    assert worst < 1e-6, f"{sizes} {output}: relative error {worst:.2e}"
    print(f"  {str(sizes):<13} {output:<8} {model.n_parameters:3d} parameters, largest relative error {worst:.1e}  ok")
