"""Reproduce the complete EDA from the bundled CSV; no predictive model is trained.

Usage (repository root): python docs/projects/eda/code/eda.py
All post-split EDA and learned statistics use training rows exclusively. The test
set is used only for the mandated split counts and pipeline contract checks.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import hashlib
import json
import platform
import importlib.metadata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import train_test_split
from umap import UMAP

from preprocessing import FEATURES, NUMERIC_FEATURES as NUM, CATEGORICAL_FEATURES as CAT
from preprocessing import SEED, build_preprocessor

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "code/data/healthcare-dataset-stroke-data.csv"
FIG = BASE / "figures"
OUT = BASE / "results"
EXPECTED_SHA = "644d473b05d2797006bd94865e4f8bb057f0c721617911613c82c8fcfc707420"
COLORS = {0: "#346b89", 1: "#d04a2c"}
LABELS = {0: "No stroke (0)", 1: "Stroke (1)"}
UNITS = {"age": "Age (years)", "avg_glucose_level": "Average glucose (source unit unspecified)",
         "bmi": "BMI (kg/m²)"}


def save(fig, number, slug, title):
    fig.suptitle(f"Figure {number}. {title}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(FIG / f"fig{number:02d}_{slug}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def export(frame, name):
    frame.to_csv(OUT / f"{name}.csv", index=False, float_format="%.10g")


def grouped_numeric(train, by):
    rows = []
    for feature in NUM:
        for category, group in train.groupby(by, observed=True, sort=True):
            x = group[feature].dropna()
            rows.append(dict(feature=feature, group=str(category), count=len(x),
                             median=x.median(), q1=x.quantile(.25), q3=x.quantile(.75),
                             iqr=x.quantile(.75)-x.quantile(.25)))
    return pd.DataFrame(rows)


def projection_plot(ax, xy, y, title, names=("Dimension 1 (arbitrary units)", "Dimension 2 (arbitrary units)")):
    for value in [0, 1]:
        mask = y == value
        ax.scatter(xy[mask, 0], xy[mask, 1], s=9 if value == 0 else 19,
                   color=COLORS[value], alpha=.28 if value == 0 else .85,
                   label=f"{LABELS[value]} · n={mask.sum():,}", rasterized=True,
                   edgecolors="none")
    ax.set(title=title, xlabel=names[0], ylabel=names[1])
    ax.legend(fontsize=8)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=.95)
    sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA:
        raise ValueError("Bundled CSV changed: verify the source and update its documented checksum.")
    raw = pd.read_csv(DATA)
    assert raw.shape == (5110, 12) and raw.id.is_unique
    assert raw.stroke.isin([0, 1]).all()
    # No data-dependent preprocessing/feature selection occurs before this split.
    train, test = train_test_split(raw, test_size=.2, stratify=raw.stroke, random_state=SEED)
    assert not set(train.id) & set(test.id)
    export(pd.concat([train[["id"]].assign(partition="train"),
                      test[["id"]].assign(partition="test")]), "split_ids")

    # Full-file inventory is mandated in stage 1; subsequent EDA is train-only.
    quality = pd.DataFrame({"column": raw.columns,
                            "missing_count": raw.isna().sum().to_numpy(),
                            "missing_pct": raw.isna().mean().to_numpy()*100,
                            "train_missing_count": train.isna().sum().to_numpy(),
                            "train_missing_pct": train.isna().mean().to_numpy()*100,
                            "unique_nonmissing": raw.nunique().to_numpy()})
    export(quality, "quality")
    checks = {
        "duplicate_full_rows": int(raw.duplicated().sum()),
        "duplicate_rows_without_id": int(raw.drop(columns="id").duplicated().sum()),
        "duplicate_ids": int(raw.id.duplicated().sum()),
        "constant_features": [c for c in FEATURES if raw[c].nunique(dropna=False) == 1],
        "age_outside_0_120": int((~raw.age.between(0, 120)).sum()),
        "nonpositive_glucose": int((raw.avg_glucose_level <= 0).sum()),
        "nonpositive_observed_bmi": int((raw.bmi.dropna() <= 0).sum()),
        "invalid_binary_flags": int(sum((~raw[c].isin([0, 1])).sum() for c in ["hypertension", "heart_disease", "stroke"])),
        "invalid_categories": int(sum((~raw[c].isin(values)).sum() for c, values in {
            "gender": ["Female", "Male", "Other"], "ever_married": ["Yes", "No"],
            "work_type": ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
            "Residence_type": ["Urban", "Rural"],
            "smoking_status": ["formerly smoked", "never smoked", "smokes", "Unknown"],
        }.items())),
        # Contextual checks identify records needing clarification, not diagnoses.
        "children_work_type_age_18_or_more": int(((raw.work_type == "children") & (raw.age >= 18)).sum()),
        "married_under_18": int(((raw.ever_married == "Yes") & (raw.age < 18)).sum()),
        "smokes_under_18": int(((raw.smoking_status == "smokes") & (raw.age < 18)).sum()),
    }
    targets = pd.DataFrame([dict(partition=part, label=label, count=int((data.stroke == label).sum()),
                                pct=float((data.stroke == label).mean()*100))
                            for part, data in [("full", raw), ("train", train), ("test", test)]
                            for label in [0, 1]])
    export(targets, "target")
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = raw.stroke.value_counts().sort_index()
    ax.bar([LABELS[0], LABELS[1]], counts, color=[COLORS[0], COLORS[1]])
    for i, n in enumerate(counts):
        ax.text(i, n+70, f"{n:,} ({n/len(raw):.2%})", ha="center")
    ax.legend(handles=[Patch(color=COLORS[0], label=LABELS[0]), Patch(color=COLORS[1], label=LABELS[1])], fontsize=8)
    ax.set(xlabel="Observed target class", ylabel="Records (full-file inventory)", ylim=(0, 5500))
    save(fig, 1, "target", "The target is severely imbalanced")

    stats = train[NUM].describe().T.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})
    stats["missing"] = train[NUM].isna().sum()
    stats["skewness"] = train[NUM].skew()
    stats = stats.reset_index(names="feature")
    export(stats, "numeric_stats")
    q1, q3 = train[NUM].quantile(.25), train[NUM].quantile(.75)
    lower, upper = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
    flags = train[NUM].lt(lower) | train[NUM].gt(upper)
    outliers = pd.DataFrame({"feature": NUM, "lower_fence": lower.to_numpy(),
                             "upper_fence": upper.to_numpy(), "flagged": flags.sum().to_numpy(),
                             "flagged_pct": flags.mean().to_numpy()*100})
    export(outliers, "outliers")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for j, c in enumerate(NUM):
        sns.histplot(train[c].dropna(), bins=30, ax=axes[0, j], color=COLORS[0])
        axes[0, j].axvline(train[c].median(), color=COLORS[1], label="Median")
        axes[0, j].set(title=c, xlabel=UNITS[c], ylabel="Training records")
        axes[0, j].legend(fontsize=8)
        sns.boxplot(x=train[c], ax=axes[1, j], color="#87b6ca", flierprops={"markersize": 2})
        axes[1, j].set(xlabel=UNITS[c], ylabel="Training distribution")
    fig.legend(handles=[Patch(color="#87b6ca", label="Boxes: Q1–Q3; line: median; whiskers: 1.5×IQR; dots: extremes")], loc="lower center", fontsize=8)
    fig.subplots_adjust(bottom=.13)
    save(fig, 2, "numerical", "Numerical distributions and IQR extremes · train only")

    categories, rates = [], []
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    for ax, c in zip(axes.flat, CAT):
        vc = train[c].value_counts().sort_index()
        for k, n in vc.items():
            categories.append(dict(feature=c, category=str(k), count=int(n), pct=float(n/len(train)*100),
                                   cardinality=len(vc), rare=bool(n/len(train) < .01)))
        ax.barh([str(v) for v in vc.index], vc.values, color=COLORS[0])
        ax.set(title=f"{c} · {len(vc)} categories", xlabel="Training records", ylabel="Category")
        for i, n in enumerate(vc):
            ax.text(n+max(vc)*.02, i, str(n), va="center", fontsize=8)
        ax.set_xlim(0, max(vc)*1.23)
    axes.flat[-1].axis("off")
    fig.legend(handles=[Patch(color=COLORS[0], label="Observed training counts")], loc="lower right", fontsize=9)
    save(fig, 3, "categorical", "Category frequencies · train only")
    export(pd.DataFrame(categories), "categories")

    corr = train[NUM].corr(method="spearman")
    export(corr.reset_index(names="feature"), "correlations")
    correlation_pairs = []
    for i, c1 in enumerate(NUM):
        for c2 in NUM[i+1:]:
            pair = train[[c1, c2]].dropna()
            correlation_pairs.append(dict(feature1=c1, feature2=c2, n=len(pair),
                                          spearman=float(spearmanr(pair[c1], pair[c2]).statistic)))
    export(pd.DataFrame(correlation_pairs), "correlation_pairs")
    strongest = max(correlation_pairs, key=lambda r: abs(r["spearman"]))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".3f", vmin=-1, vmax=1, cmap="vlag", square=True,
                cbar_kws={"label": "Spearman rank correlation"}, ax=ax)
    ax.set(xlabel="Numerical feature", ylabel="Numerical feature")
    save(fig, 4, "correlation", "Moderate association does not imply redundancy")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, pair in zip(axes, correlation_pairs):
        c1, c2 = pair["feature1"], pair["feature2"]
        ax.scatter(train[c1], train[c2], s=7, alpha=.25, color=COLORS[0], label="Observed train pairs")
        ax.legend(fontsize=8)
        ax.set(xlabel=UNITS[c1], ylabel=UNITS[c2], title=f"ρ = {pair['spearman']:.3f} · n = {pair['n']:,}")
    save(fig, 5, "scatter", "All numerical pairs · un-imputed training observations")

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    for ax, c in zip(axes.flat, CAT):
        groups = train.groupby(c, observed=True, sort=True).stroke.agg(["count", "sum", "mean"])
        labels = [f"{k} (n={int(row['count'])})" for k, row in groups.iterrows()]
        ax.barh(labels, groups["mean"]*100, color=COLORS[1])
        ax.axvline(train.stroke.mean()*100, linestyle="--", color=COLORS[0], label="Train prevalence")
        ax.set(title=c, xlabel="Stroke within category (%)", ylabel="Category and denominator")
        ax.set_xlim(0, max(22, float(groups["mean"].max()*120)))
        ax.legend(fontsize=7, loc="lower right")
        for k, row in groups.iterrows():
            rates.append(dict(feature=c, category=str(k), n=int(row["count"]),
                              stroke=int(row["sum"]), no_stroke=int(row["count"]-row["sum"]),
                              stroke_pct=float(row["mean"]*100)))
    axes.flat[-1].axis("off")
    save(fig, 6, "category_target", "Conditional stroke proportions · denominators matter")
    export(pd.DataFrame(rates), "category_target")

    for number, group in [(7, "hypertension"), (8, "work_type"), (9, "stroke")]:
        grouped = grouped_numeric(train, group)
        export(grouped, f"grouped_{group}")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        for ax, c in zip(axes, NUM):
            sns.boxplot(data=train, x=group, y=c, ax=ax, color="#87b6ca",
                        flierprops={"markersize": 2})
            ax.set(xlabel=group, ylabel=UNITS[c], title=c)
            ax.tick_params(axis="x", labelrotation=20)
        fig.legend(handles=[Patch(color="#87b6ca", label="Boxes: Q1–Q3; line: median; whiskers: 1.5×IQR; dots: extremes")], loc="lower center", fontsize=8)
        save(fig, number, f"box_{group}", f"Numerical features grouped by {group} · train only")

    missing = train.groupby("stroke").bmi.apply(lambda x: float(x.isna().mean()*100)).to_dict()
    Xtrain, Xtest = train[FEATURES], test[FEATURES]
    preprocessor = build_preprocessor()
    Xt = preprocessor.fit_transform(Xtrain)
    Xv = preprocessor.transform(Xtest)
    names = preprocessor.get_feature_names_out()
    assert np.isfinite(Xt).all() and np.isfinite(Xv).all()
    assert Xt.shape[1] == len(names) == Xv.shape[1]
    export(pd.DataFrame({"feature": names}), "feature_names")
    columns = preprocessor.named_steps["columns"]
    num_pipe = columns.named_transformers_["num"]
    export(pd.DataFrame({"feature": NUM, "imputation_median": num_pipe["impute"].statistics_,
                         "scaler_center": num_pipe["scale"].center_, "scaler_iqr": num_pipe["scale"].scale_}),
           "fitted_numeric")
    # Full PCA retains all components for cumulative variance and loading analysis.
    pca = PCA(svd_solver="full", random_state=SEED)
    P = pca.fit_transform(Xt)
    export(pd.DataFrame({"component": np.arange(1, len(names)+1),
                         "explained_ratio": pca.explained_variance_ratio_,
                         "cumulative_ratio": np.cumsum(pca.explained_variance_ratio_)}), "pca_variance")
    loadings = pd.DataFrame({"feature": names, "PC1": pca.components_[0], "PC2": pca.components_[1]})
    export(loadings, "pca_loadings")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    y = train.stroke.to_numpy()
    projection_plot(axes[0], P[:, :2], y, "PCA projection", ("PC1 score", "PC2 score"))
    axes[1].plot(np.arange(1, len(names)+1), np.cumsum(pca.explained_variance_ratio_)*100, marker="o", ms=3)
    axes[1].axhline(90, color=COLORS[1], ls="--", label="90% variance")
    axes[1].set(xlabel="Number of principal components", ylabel="Cumulative explained variance (%)",
                title="Variance retained", ylim=(0, 103))
    axes[1].legend(fontsize=8)
    top_idx = np.argsort(np.max(np.abs(pca.components_[:2]), axis=0))[-8:]
    short = [names[i].replace("num__", "").replace("cat__", "").replace("bmi_missing__missingindicator_", "missing_") for i in top_idx]
    pos = np.arange(len(top_idx))
    axes[2].barh(pos-.18, pca.components_[0, top_idx], height=.36, label="PC1")
    axes[2].barh(pos+.18, pca.components_[1, top_idx], height=.36, label="PC2")
    axes[2].set(yticks=pos, yticklabels=short, xlabel="Eigenvector coefficient (signed)", title="Largest coefficients")
    axes[2].legend(fontsize=8)
    save(fig, 10, "pca", "PCA on the training pipeline output")

    # Same entire training set for all projections; targets are coloring only.
    embeddings = {"pca": P[:, :2]}
    diagnostics = []
    eval_idx = np.sort(np.random.default_rng(SEED).choice(len(Xt), size=1000, replace=False))
    for method, params, number in [("tsne", [30, 50], 11), ("umap", [15, 50], 12)]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, value in zip(axes, params):
            if method == "tsne":
                reducer = TSNE(n_components=2, perplexity=value, init="pca", learning_rate="auto",
                               max_iter=1000, metric="euclidean", random_state=SEED, n_jobs=1)
                label = f"perplexity = {value}"
            else:
                reducer = UMAP(n_components=2, n_neighbors=value, min_dist=.1, n_epochs=500,
                               metric="euclidean", init="spectral", random_state=SEED,
                               transform_seed=SEED, n_jobs=1)
                label = f"n_neighbors = {value}"
            print(f"Fitting {method}: {label} on {len(Xt):,} training rows", flush=True)
            embedding = reducer.fit_transform(Xt)
            assert np.isfinite(embedding).all()
            embeddings[f"{method}_{value}"] = embedding
            projection_plot(ax, embedding, y, label)
            diagnostic = dict(method=method, parameter=value,
                              trustworthiness=float(trustworthiness(Xt[eval_idx], embedding[eval_idx], n_neighbors=10)))
            if method == "tsne":
                diagnostic["kl_divergence"] = float(reducer.kl_divergence_)
            diagnostics.append(diagnostic)
        save(fig, number, method, f"{method.upper()} parameter sensitivity · same training rows")
    diagnostics.insert(0, dict(method="pca", parameter=2,
                              trustworthiness=float(trustworthiness(Xt[eval_idx], P[eval_idx, :2], n_neighbors=10))))
    export(pd.DataFrame(diagnostics), "projection_diagnostics")
    for name, xy in embeddings.items():
        export(pd.DataFrame({"id": train.id.to_numpy(), "stroke": y,
                             "dimension1": xy[:, 0], "dimension2": xy[:, 1]}), f"embedding_{name}")
    versions = {package: importlib.metadata.version(package) for package in [
        "numpy", "pandas", "matplotlib", "seaborn", "scipy", "scikit-learn", "umap-learn", "numba",
    ]}
    summary = dict(dataset_sha256=sha, seed=SEED, raw_shape=list(raw.shape),
                   train_shape=list(Xt.shape), test_shape=list(Xv.shape), checks=checks,
                   missing_bmi_by_target_train=missing, outlier_flagged_rows=int(flags.any(axis=1).sum()),
                   outlier_rows_removed=0, outlier_values_clipped=0,
                   strongest_pair=strongest, pca_pc1_pc2=float(pca.explained_variance_ratio_[:2].sum()),
                   pca_components_90=int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), .9)+1),
                   projection_diagnostics=diagnostics, python=platform.python_version(), versions=versions,
                   numeric_feature_count=len(NUM), categorical_feature_count=len(CAT),
                   transformed_nan_train=int(np.isnan(Xt).sum()), transformed_nan_test=int(np.isnan(Xv).sum()))
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    from report import write_report
    write_report()
    print("Report, figures, tables and split IDs regenerated successfully.", flush=True)


if __name__ == "__main__":
    main()
