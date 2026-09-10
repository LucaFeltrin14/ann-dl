"""Exercise 3 - Preparing Real-World Data for a Neural Network.

Describes the Kaggle Spaceship Titanic training set, splits it before touching any
statistic, and preprocesses it for a network whose hidden layers use tanh.
Writes Figure 6.

Every number quoted in the report is printed by this script. Run it with:

    python docs/exercises/data/code/exercise3_spaceship_titanic.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.2

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
DATA_PATH = HERE / "data" / "train.csv"

TARGET = "Transported"
SPENDING = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERICAL = ["Age"] + SPENDING
CATEGORICAL = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
IDENTIFIERS = ["PassengerId", "Cabin", "Name"]  # dropped: identifiers, not features
ENGINEERED = ["TotalSpend"]
HEAVY_TAILED = SPENDING + ENGINEERED  # the columns that get log(1 + x)

if not DATA_PATH.exists():  # pragma: no cover - guard for a fresh checkout
    raise SystemExit(
        f"Missing dataset: {DATA_PATH}\n"
        "Download train.csv from https://www.kaggle.com/competitions/spaceship-titanic "
        "and place it there."
    )

df = pd.read_csv(DATA_PATH)


def missing_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Missing values per column, in absolute count and in percentage."""
    return pd.DataFrame({
        "missing": frame.isna().sum(),
        "missing_pct": (frame.isna().mean() * 100).round(2),
    }).sort_values("missing", ascending=False)


def as_markdown(frame: pd.DataFrame, index_name: str) -> str:
    """Render a small DataFrame as a markdown table, ready to paste into the report."""
    header = f"| {index_name} | " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * (len(frame.columns) + 1)
    def cell(value: object) -> str:
        if isinstance(value, float) and value.is_integer():
            return f"{int(value)}"
        return f"{value}"

    rows = [
        f"| {index} | " + " | ".join(cell(value) for value in row) + " |"
        for index, row in zip(frame.index, frame.to_numpy())
    ]
    return "\n".join([header, rule, *rows])


# ---------------------------------------------------------------------------
# B - Split BEFORE any statistic is computed
# ---------------------------------------------------------------------------
y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
)

# ---------------------------------------------------------------------------
# C - Preprocess: everything below is FITTED ON THE TRAINING SET ONLY
# ---------------------------------------------------------------------------
# 1. Missing values.
#    Numerical  -> median: the spending columns are extremely right-skewed (mean far
#                  above the median), so the mean would be dragged up by a handful of
#                  big spenders; the median is unaffected by them. For the spending
#                  columns the training median is 0, which also matches the most
#                  plausible reading of a missing value there ("did not spend").
#    Categorical -> most frequent: only ~2% of the rows are missing per column, so the
#                  mode barely shifts the observed distribution, and it keeps the
#                  one-hot block to the categories that really exist.
numerical_imputer = SimpleImputer(strategy="median").fit(X_train[NUMERICAL])
categorical_imputer = SimpleImputer(strategy="most_frequent").fit(X_train[CATEGORICAL])


def impute(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.drop(columns=IDENTIFIERS).copy()
    out[NUMERICAL] = numerical_imputer.transform(out[NUMERICAL])
    out[CATEGORICAL] = categorical_imputer.transform(out[CATEGORICAL])
    # 2. Feature engineering: total amount spent across the five amenities.
    out["TotalSpend"] = out[SPENDING].sum(axis=1)
    return out


train_imputed = impute(X_train)
test_imputed = impute(X_test)

# 3. Heavy tails: log(1 + x) compresses the 0 - 29 813 range of the spending columns
#    into roughly 0 - 10 before scaling, so a single big spender stops deciding the
#    whole scale of the feature.
train_logged = train_imputed.copy()
test_logged = test_imputed.copy()
for column in HEAVY_TAILED:
    train_logged[column] = np.log1p(train_logged[column])
    test_logged[column] = np.log1p(test_logged[column])

# 4. Categorical encoding. handle_unknown="ignore" is the answer to "a category that
#    appears in the test set but not in the training set": the encoder only knows the
#    training categories, and an unseen one is encoded as an all-zeros block instead of
#    raising or silently creating a column the network was never built for.
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train_logged[CATEGORICAL])

# 5. Scaling to [-1, 1]. tanh maps into exactly that interval, so inputs on the same
#    scale keep the units in the informative part of the curve instead of saturating
#    them. Fitted on the training set: the test set is only transformed.
NUMERIC_FEATURES = NUMERICAL + ENGINEERED
scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_logged[NUMERIC_FEATURES])

FEATURE_NAMES = NUMERIC_FEATURES + list(encoder.get_feature_names_out(CATEGORICAL))


def transform(frame: pd.DataFrame) -> np.ndarray:
    """Apply the fitted scaler and encoder, returning the final feature matrix."""
    return np.hstack([scaler.transform(frame[NUMERIC_FEATURES]), encoder.transform(frame[CATEGORICAL])])


X_train_final = transform(train_logged)
X_test_final = transform(test_logged)

# ---------------------------------------------------------------------------
# D - Figure 6: FoodCourt through the pipeline
# ---------------------------------------------------------------------------
food_raw = X_train["FoodCourt"].dropna()
food_log = train_logged["FoodCourt"]
food_scaled = X_train_final[:, NUMERIC_FEATURES.index("FoodCourt")]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
panels = [
    (food_raw, "Before: raw FoodCourt", "Amount spent", "#1f77b4"),
    (food_log, r"After $\log(1 + x)$", r"$\log(1 + \mathrm{FoodCourt})$", "#ff7f0e"),
    (food_scaled, "After scaling to $[-1, 1]$", "Scaled value", "#2ca02c"),
]
for ax, (values, title, xlabel, color) in zip(axes, panels):
    ax.hist(values, bins=50, color=color, edgecolor="white", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of passengers")
    ax.set_yscale("log")  # without it the zero bar hides every other bar
    ax.grid(alpha=0.25, linestyle=":")
fig.suptitle("Figure 6 - FoodCourt (training set) before and after preprocessing "
             "(log-scaled counts)", fontsize=13)
fig.tight_layout()
fig.savefig(FIGURES / "fig6_foodcourt.png", dpi=150)
plt.close(fig)


if __name__ == "__main__":
    print("=" * 78)
    print("EXERCISE 3 - PREPARING REAL-WORLD DATA FOR A NEURAL NETWORK")
    print(f"seed = {SEED} | source = {DATA_PATH.relative_to(HERE.parents[3])}")
    print("=" * 78)

    print(f"\n[A] dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    balance = df[TARGET].value_counts(normalize=True).sort_index()
    counts = df[TARGET].value_counts().sort_index()
    print(f"[A] target balance: False = {balance[False]:.4f} ({counts[False]} rows) | "
          f"True = {balance[True]:.4f} ({counts[True]} rows)")
    print(f"[A] numerical features  ({len(NUMERICAL)}): {NUMERICAL}")
    print(f"[A] categorical features ({len(CATEGORICAL)}): {CATEGORICAL}")
    print(f"[A] identifiers dropped  ({len(IDENTIFIERS)}): {IDENTIFIERS}")

    print("\n[A] missing values per column (markdown table)")
    print(as_markdown(missing_table(df), "Column"))

    print("\n[A] spending columns - mean, median and maximum (markdown table)")
    spending_stats = df[SPENDING].agg(["mean", "median", "max"]).T.round(2)
    spending_stats["mean - median"] = (spending_stats["mean"] - spending_stats["median"]).round(2)
    print(as_markdown(spending_stats, "Column"))

    print("\n[B] stratified 80/20 split, computed before any statistic")
    print(f"  train: {X_train.shape}  positive share {y_train.mean():.4f}")
    print(f"  test : {X_test.shape}  positive share {y_test.mean():.4f}")
    print(f"  FoodCourt on the TRAINING set before transforming: "
          f"mean = {X_train['FoodCourt'].mean():.2f}, median = {X_train['FoodCourt'].median():.2f}")

    print("\n[C] fitted statistics (training set only)")
    medians = {name: float(value) for name, value in zip(NUMERICAL, numerical_imputer.statistics_)}
    modes = {name: str(value) for name, value in zip(CATEGORICAL, categorical_imputer.statistics_)}
    print(f"  numerical medians used for imputation: {medians}")
    print(f"  categorical modes used for imputation: {modes}")
    for name, categories in zip(CATEGORICAL, encoder.categories_):
        print(f"  one-hot {name}: {list(categories)}")
    print(f"  final feature list ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

    print("\n[D] final checks")
    print(f"  training feature matrix shape: {X_train_final.shape}")
    print(f"  test feature matrix shape:     {X_test_final.shape}")
    print(f"  remaining NaN - train: {int(np.isnan(X_train_final).sum())} | "
          f"test: {int(np.isnan(X_test_final).sum())}")
    n_numeric = len(NUMERIC_FEATURES)
    print(f"  scaled numeric block  - train: [{X_train_final[:, :n_numeric].min():.4f}, "
          f"{X_train_final[:, :n_numeric].max():.4f}] | "
          f"test: [{X_test_final[:, :n_numeric].min():.4f}, {X_test_final[:, :n_numeric].max():.4f}]")
    print(f"  full feature matrix   - train: [{X_train_final.min():.4f}, {X_train_final.max():.4f}] | "
          f"test: [{X_test_final.min():.4f}, {X_test_final.max():.4f}]")
    outside_mask = np.abs(X_test_final) > 1
    outside = int(outside_mask.sum())
    culprits = [FEATURE_NAMES[column] for column in np.unique(np.nonzero(outside_mask)[1])]
    print(f"  test values outside [-1, 1]: {outside} of {X_test_final.size} "
          f"({outside / X_test_final.size:.4%}) in {culprits} "
          f"- expected, the scaler never saw them")
    print(f"  tanh-compatible (|x| <= 1) on the training set: {bool((np.abs(X_train_final) <= 1).all())}")

    print("\n[figures]")
    print(f"  saved {FIGURES / 'fig6_foodcourt.png'}")
