"""
Additional Task: Multi-Layer Perceptron Learning Algorithm for the XOR Gate
Shows weights after every update, plots the decision boundary after every
update, and studies why the network fails to converge.

Architecture: 2 inputs -> 2 hidden neurons (step activation) -> 1 output
neuron (step activation). All neurons use the plain perceptron delta rule
(error = target - output), applied directly to both the output layer and
the hidden layer. There is no backpropagation: the hidden layer weights
are updated using the same output error signal as the output layer,
instead of an error term that is specific to each hidden neuron.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

np.random.seed(42)

# Font & export settings: Times New Roman, size 13, 600 DPI, PDF output.
# Times New Roman itself is a proprietary Microsoft font and is not installed
# on this system, so Liberation Serif is used instead, a metrically
# compatible, open-source substitute. If real Times New Roman is installed
# locally, change FONT_NAME below and re-run.
FONT_NAME = "Liberation Serif"  # swap for "Times New Roman" if installed locally
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = [FONT_NAME, "Times New Roman", "DejaVu Serif"]
mpl.rcParams["font.size"] = 13
mpl.rcParams["savefig.dpi"] = 600
mpl.rcParams["figure.dpi"] = 600


class NaiveMLP:
    def __init__(self, n_inputs, n_hidden, learning_rate=0.1):
        self.lr = learning_rate
        # small random initial weights, so the two hidden neurons do not
        # start out perfectly identical
        self.W_hidden = np.random.uniform(-0.1, 0.1, size=(n_hidden, n_inputs))
        self.b_hidden = np.random.uniform(-0.1, 0.1, size=n_hidden)
        self.W_out = np.random.uniform(-0.1, 0.1, size=n_hidden)
        self.b_out = np.random.uniform(-0.1, 0.1)
        self.history = []  # snapshot after every update

    @staticmethod
    def step(z):
        return np.where(z >= 0, 1, 0)

    def forward(self, x):
        hidden_z = self.W_hidden @ x + self.b_hidden
        hidden_a = self.step(hidden_z)
        out_z = self.W_out @ hidden_a + self.b_out
        out_a = self.step(out_z)
        return hidden_a, out_a

    def snapshot(self):
        self.history.append((
            self.W_hidden.copy(), self.b_hidden.copy(),
            self.W_out.copy(), self.b_out
        ))

    def fit(self, X, y, epochs=30):
        self.snapshot()
        for epoch in range(1, epochs + 1):
            errors = 0
            for xi, target in zip(X, y):
                hidden_a, pred = self.forward(xi)
                error = target - pred
                if error != 0:
                    # naive rule: the SAME output error is used to update
                    # both the output layer and the hidden layer, with no
                    # backpropagated, neuron-specific error term
                    self.W_out += self.lr * error * hidden_a
                    self.b_out += self.lr * error
                    self.W_hidden += self.lr * error * np.outer(np.ones_like(self.b_hidden), xi)
                    self.b_hidden += self.lr * error * np.ones_like(self.b_hidden)

                    errors += 1
                    self.snapshot()
                    print(f"Epoch {epoch}, sample {xi}, target {target}, pred {pred} -> "
                          f"W_hidden={np.round(self.W_hidden, 3).tolist()}, "
                          f"b_hidden={np.round(self.b_hidden, 3).tolist()}, "
                          f"W_out={np.round(self.W_out, 3)}, b_out={self.b_out:.3f}")
            print(f"  epoch {epoch} misclassified: {errors}")
            if errors == 0:
                print(f"Converged after epoch {epoch}, no errors left.\n")
                break
        return self.history


# ----------------------------------------------------------------------
# Plot the decision boundary (as a filled contour of the predicted class)
# after every weight update
# ----------------------------------------------------------------------
def plot_decision_boundaries(X, y, history, max_panels=16):
    n_updates = min(len(history), max_panels)
    n_cols = 4
    n_rows = int(np.ceil(n_updates / n_cols))

    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i in range(n_updates):
        W_hidden, b_hidden, W_out, b_out = history[i]
        ax = axes[i]

        preds = np.zeros(grid.shape[0])
        for idx, point in enumerate(grid):
            hidden_z = W_hidden @ point + b_hidden
            hidden_a = np.where(hidden_z >= 0, 1, 0)
            out_z = W_out @ hidden_a + b_out
            preds[idx] = 1 if out_z >= 0 else 0
        preds = preds.reshape(xx.shape)

        ax.contourf(xx, yy, preds, levels=[-0.5, 0.5, 1.5],
                    colors=["#f7c9c2", "#bfe3c8"], alpha=0.8)

        for xi, target in zip(X, y):
            color = "seagreen" if target == 1 else "crimson"
            ax.scatter(xi[0], xi[1], c=color, s=120, edgecolor="black", zorder=3)

        title = "Initial" if i == 0 else f"Update {i}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    for j in range(n_updates, len(axes)):
        axes[j].axis("off")

    fig.suptitle("XOR: Decision Boundary After Each Weight Update (Naive MLP, No Backprop)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("xor_mlp_decision_boundaries.pdf", dpi=600)
    plt.show()


# ----------------------------------------------------------------------
# Error curve across epochs
# ----------------------------------------------------------------------
def plot_error_curve(X, y, model_class, n_inputs, n_hidden, epochs=30, lr=0.1):
    np.random.seed(42)
    model = model_class(n_inputs, n_hidden, learning_rate=lr)
    errors_per_epoch = []
    for epoch in range(1, epochs + 1):
        errors = 0
        for xi, target in zip(X, y):
            hidden_a, pred = model.forward(xi)
            error = target - pred
            if error != 0:
                model.W_out += model.lr * error * hidden_a
                model.b_out += model.lr * error
                model.W_hidden += model.lr * error * np.outer(np.ones_like(model.b_hidden), xi)
                model.b_hidden += model.lr * error * np.ones_like(model.b_hidden)
                errors += 1
        errors_per_epoch.append(errors)

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, epochs + 1), errors_per_epoch, marker="o", color="darkorange")
    plt.xlabel("Epoch")
    plt.ylabel("Misclassified Samples")
    plt.title("Naive MLP on XOR: Training Error vs Epoch")
    plt.savefig("xor_mlp_error_vs_epoch.pdf", dpi=600)
    plt.show()
    return errors_per_epoch


# ----------------------------------------------------------------------
# Train on XOR
# ----------------------------------------------------------------------
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

print("=" * 60)
print("XOR GATE, NAIVE MULTI-LAYER PERCEPTRON (NO BACKPROPAGATION)")
print("=" * 60)

mlp = NaiveMLP(n_inputs=2, n_hidden=2, learning_rate=0.1)
history_xor = mlp.fit(X_xor, y_xor, epochs=30)

plot_decision_boundaries(X_xor, y_xor, history_xor, max_panels=16)
errors_per_epoch = plot_error_curve(X_xor, y_xor, NaiveMLP, n_inputs=2, n_hidden=2, epochs=30, lr=0.1)

print("Misclassified samples per epoch:", errors_per_epoch)
