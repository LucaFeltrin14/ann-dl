"""Importable, unfitted preprocessing for the Stroke Prediction classification task.

No estimator is fitted during import. Call build_preprocessor(), fit it on X_train,
and only transform X_test. Binary clinical flags are nominal features, not counts.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

SEED = 42
NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL_FEATURES = [
    "gender", "hypertension", "heart_disease", "ever_married", "work_type",
    "Residence_type", "smoking_status",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> Pipeline:
    """Return a fresh Pipeline; output is a dense, finite numeric matrix.

    The explicit BMI indicator is never scaled (its values stay 0/1). Missing
    categories use a distinct sentinel; the literal smoking category Unknown
    remains an observed category. Unknown categories at transform time produce
    an all-zero one-hot block and never trigger fitting on the held-out rows.
    """
    columns = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", RobustScaler(quantile_range=(25, 75))),
        ]), NUMERIC_FEATURES),
        ("bmi_missing", MissingIndicator(features="all", error_on_new=False), ["bmi"]),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                     dtype=float, drop=None)),
        ]), CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=True)
    return Pipeline([("columns", columns)])
