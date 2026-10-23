"""Multi-layer perceptron, written from scratch with NumPy only.

This one implementation is imported, unchanged, by Exercises 2, 3 and 4. What changes between them
is passed as an argument, never edited in here:

    exercise2_binary.py      MLP([2, 8, 1],      output="sigmoid")   binary cross-entropy
    exercise3_multiclass.py  MLP([4, 16, 3],     output="softmax")   categorical cross-entropy
    exercise4_depth.py       MLP([4, 16, 16, 3], output="softmax")   same, one more hidden layer

Notation, as in the report. Layer l = 1..L has weights W[l] of shape (n_out, n_in) and biases b[l].
A batch of N samples is a matrix with one sample per row, so for every layer

    U^(l) = H^(l-1) W^(l)T + b^(l)            pre-activation (the statement's z, u)
    H^(l) = tanh(U^(l))                        hidden layers
    P     = output(U^(L))                      sigmoid or softmax, the network's probabilities

with H^(0) = X. The backward pass is the chain rule written out by hand; nothing is autodiffed.
"""

import numpy as np


# ---------------------------------------------------------------------------- activations
def tanh_derivative(h):
    """d tanh(u) / du = 1 - tanh(u)^2, written in terms of the activation h = tanh(u) itself."""
    return 1.0 - h ** 2


def sigmoid(u):
    """Logistic function 1 / (1 + e^-u), written as 0.5 (1 + tanh(u/2)) so that large |u| cannot overflow."""
    return 0.5 * (1.0 + np.tanh(0.5 * u))


def softmax(u):
    """Row-wise softmax. Subtracting the row maximum leaves the result unchanged and avoids overflow."""
    e = np.exp(u - u.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------- losses (mean over the batch)
EPS = 1e-12  # keeps log() finite when a probability rounds to exactly 0 or 1


def binary_cross_entropy(P, Y):
    """L = -(1/N) sum_i [ y_i log p_i + (1 - y_i) log(1 - p_i) ], for P and Y of shape (N, 1)."""
    P = np.clip(P, EPS, 1 - EPS)
    return float(-np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P)))


def categorical_cross_entropy(P, Y):
    """L = -(1/N) sum_i sum_k y_ik log p_ik, for P and one-hot Y of shape (N, K)."""
    return float(-np.mean(np.sum(Y * np.log(np.clip(P, EPS, 1.0)), axis=1)))


# Each output activation comes with the loss it is paired with. For both pairs the derivative of the
# loss with respect to the output pre-activation is the same, dL/dU^(L) = (P - Y) / N (derived in the
# report, items 2C and 3B), which is why backward() below works for both without a single change.
OUTPUTS = {
    "sigmoid": (sigmoid, binary_cross_entropy),
    "softmax": (softmax, categorical_cross_entropy),
}


class MLP:
    """Fully connected network: tanh hidden layers, a sigmoid or softmax output, trained by gradient descent.

    layer_sizes  [n_inputs, hidden_1, ..., hidden_k, n_outputs], e.g. [2, 8, 1]; any number of hidden layers
    output       "sigmoid" (1 output unit, binary cross-entropy) or "softmax" (K units, categorical cross-entropy)
    rng          the numpy Generator used for the initial weights
    """

    def __init__(self, layer_sizes, output, rng):
        self.layer_sizes = list(layer_sizes)
        self.output_activation, self.loss_function = OUTPUTS[output]
        # Initialization: W ~ N(0, 1/n_in), b = 0. With inputs of unit scale the pre-activations then start
        # with variance ~1, inside tanh's responsive range, so no unit starts saturated.
        self.W = [rng.normal(0.0, np.sqrt(1.0 / n_in), size=(n_out, n_in))
                  for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        self.b = [np.zeros(n_out) for n_out in self.layer_sizes[1:]]

    @property
    def n_parameters(self) -> int:
        """Number of trainable numbers: every weight and every bias."""
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    def targets(self, y):
        """Integer labels -> the matrix Y the loss compares with P: a column of 0/1, or one-hot rows."""
        n_outputs = self.layer_sizes[-1]
        return y.reshape(-1, 1).astype(float) if n_outputs == 1 else np.eye(n_outputs)[y]

    # ------------------------------------------------------------------------ forward pass
    def forward(self, X):
        """Return the activations of every layer, [H^(0) = X, H^(1), ..., H^(L) = P].

        All of them are kept because the backward pass needs each layer's input and output.
        """
        H = [X]
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            U = H[-1] @ W.T + b
            is_output = l == len(self.W) - 1
            H.append(self.output_activation(U) if is_output else np.tanh(U))
        return H

    # ------------------------------------------------------------------------ backward pass
    def backward(self, H, Y):
        """Gradients of the mean loss with respect to every W^(l) and b^(l), by the chain rule.

        delta^(l) = dL/dU^(l), one row per sample:
            output layer:  delta^(L) = (P - Y) / N                     (sigmoid+BCE and softmax+CCE alike)
            hidden layer:  delta^(l) = (delta^(l+1) W^(l+1)) * (1 - H^(l)^2)
        and for every layer
            dL/dW^(l) = delta^(l)T H^(l-1)        dL/db^(l) = sum over the samples of delta^(l)
        """
        N = Y.shape[0]
        delta = (H[-1] - Y) / N
        grads_W, grads_b = [], []
        for l in reversed(range(len(self.W))):
            grads_W.insert(0, delta.T @ H[l])          # H[l] is this layer's input, H^(l-1) in the report
            grads_b.insert(0, delta.sum(axis=0))
            if l > 0:                                    # propagate to the layer below, through its tanh,
                delta = (delta @ self.W[l]) * tanh_derivative(H[l])  # using W before it is updated
        return grads_W, grads_b

    # ------------------------------------------------------------------------ update
    def update(self, grads_W, grads_b, learning_rate):
        """One gradient-descent step on every parameter: theta <- theta - eta * dL/dtheta."""
        for l in range(len(self.W)):
            self.W[l] -= learning_rate * grads_W[l]
            self.b[l] -= learning_rate * grads_b[l]

    # ------------------------------------------------------------------------ training and prediction
    def fit(self, X, y, epochs, learning_rate):
        """Full-batch gradient descent: every epoch is one forward pass, one backward pass and one update
        computed on the whole training set.

        Returns the training loss after 0, 1, ..., `epochs` epochs (entry 0 is the initial weights).
        The loss of each epoch comes from the same forward pass that feeds the backward pass, so recording
        it costs nothing.
        """
        Y = self.targets(y)
        history = []
        for _ in range(epochs):
            H = self.forward(X)
            history.append(self.loss_function(H[-1], Y))
            self.update(*self.backward(H, Y), learning_rate)
        history.append(self.loss(X, y))
        return np.array(history)

    def loss(self, X, y) -> float:
        """Mean loss of the current weights on (X, y)."""
        return self.loss_function(self.predict_proba(X), self.targets(y))

    def predict_proba(self, X):
        """The network's output P: P(class 1) as a column for sigmoid, one row of K probabilities for softmax."""
        return self.forward(X)[-1]

    def predict(self, X):
        """Predicted class: P >= 0.5 for a sigmoid output, the most probable class for a softmax."""
        P = self.predict_proba(X)
        return (P[:, 0] >= 0.5).astype(int) if P.shape[1] == 1 else P.argmax(axis=1)

    def accuracy(self, X, y) -> float:
        """Fraction of samples whose predicted class matches the label."""
        return float(np.mean(self.predict(X) == y))
