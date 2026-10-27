"""Shared by the exercise scripts: the seed, the split, the Exercise 3 data and its training settings.

The network itself lives in mlp.py; nothing here learns anything.
"""

from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# One seed for the whole report: every script builds rng = np.random.default_rng(SEED).
SEED = 42

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # class 0, class 1, class 2

# Exercise 3 data, with the parameters given in the statement. Exercise 4 uses the same data.
MULTICLASS_PARAMS = dict(n_samples=1500, n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
                         n_classes=3, n_clusters_per_class=2, class_sep=1.0, random_state=42)

# Exercise 3 training settings. Exercise 4 imports them, so the two can only differ in depth.
MULTICLASS_HIDDEN = [16]
LEARNING_RATE = 0.5
EPOCHS = 2000


def split(X, y, *others):
    """80/20 stratified split with the statement's seed. Extra arrays (e.g. cluster ids) are split alongside."""
    return train_test_split(X, y, *others, test_size=0.2, random_state=SEED, stratify=y)


def multiclass_data(shuffle=True):
    """Exercise 3 data. shuffle=False returns the same samples in generation order (used only in item 3D)."""
    return make_classification(**MULTICLASS_PARAMS, shuffle=shuffle)
