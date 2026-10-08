"""Independent audit of saved EDA artifacts and preprocessing isolation.

Run after eda.py. Uses raw-file counts, NumPy ranks/SVD and independently rebuilt
training partitions. The projection arrays are inspected, not treated as models.
"""
from pathlib import Path
import hashlib
import json
import pickle
import re

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.manifold import trustworthiness
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from preprocessing import FEATURES, NUMERIC_FEATURES as NUM, CATEGORICAL_FEATURES as CAT
from preprocessing import build_preprocessor

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"


def csv(name):
    return pd.read_csv(RESULTS / f"{name}.csv")


def main():
    raw_path = BASE / "code/data/healthcare-dataset-stroke-data.csv"
    raw = pd.read_csv(raw_path)
    m = json.loads((RESULTS / "metrics.json").read_text())
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == m["dataset_sha256"]
    assert raw.shape == (5110, 12) and raw.stroke.sum() == 249
    assert raw.bmi.isna().sum() == 201
    train, test = train_test_split(raw, test_size=.2, stratify=raw.stroke, random_state=42)
    ids = csv("split_ids")
    assert set(ids[ids.partition == "train"].id) == set(train.id)
    assert set(ids[ids.partition == "test"].id) == set(test.id)
    assert len(ids) == len(raw) == ids.id.nunique()
    assert not set(train.id).intersection(test.id)
    assert len(train) == 4088 and len(test) == 1022
    assert train.stroke.sum() == 199 and test.stroke.sum() == 50

    quality = csv("quality").set_index("column")
    for c in raw:
        assert quality.loc[c, "missing_count"] == raw[c].isna().sum()
        assert_allclose(quality.loc[c, "missing_pct"], 100*raw[c].isna().mean(), rtol=1e-8)
        assert quality.loc[c, "train_missing_count"] == train[c].isna().sum()
        assert_allclose(quality.loc[c, "train_missing_pct"], 100*train[c].isna().mean(), rtol=1e-8)
        assert quality.loc[c, "unique_nonmissing"] == raw[c].nunique()
    expected_checks = {
        "duplicate_full_rows": int(raw.duplicated().sum()),
        "duplicate_rows_without_id": int(raw.drop(columns="id").duplicated().sum()),
        "duplicate_ids": int(raw.id.duplicated().sum()), "constant_features": [],
        "age_outside_0_120": int(((raw.age<0)|(raw.age>120)).sum()),
        "nonpositive_glucose": int((raw.avg_glucose_level<=0).sum()),
        "nonpositive_observed_bmi": int((raw.bmi<=0).sum()),
        "invalid_binary_flags": sum(int((~raw[c].isin([0,1])).sum()) for c in ["hypertension","heart_disease","stroke"]),
        "invalid_categories": 0,
        "children_work_type_age_18_or_more": int(((raw.work_type=="children")&(raw.age>=18)).sum()),
        "married_under_18": int(((raw.ever_married=="Yes")&(raw.age<18)).sum()),
        "smokes_under_18": int(((raw.smoking_status=="smokes")&(raw.age<18)).sum()),
    }
    assert m["checks"] == expected_checks
    for part, data in [("full",raw),("train",train),("test",test)]:
        counts = csv("target").query("partition == @part").set_index("label")
        for label in [0,1]:
            assert counts.loc[label,"count"] == int((data.stroke==label).sum())
            assert_allclose(counts.loc[label,"pct"], 100*(data.stroke==label).mean(),rtol=1e-8)

    num = csv("numeric_stats").set_index("feature")
    for c in NUM:
        x = train[c].dropna().to_numpy()
        assert num.loc[c,"count"] == len(x)
        assert num.loc[c,"missing"] == train[c].isna().sum()
        for field, value in dict(mean=np.mean(x), median=np.median(x), std=np.std(x,ddof=1),
                                 q1=np.quantile(x,.25),q3=np.quantile(x,.75),min=np.min(x),max=np.max(x),
                                 skewness=train[c].skew()).items():
            assert_allclose(num.loc[c,field],value,rtol=1e-8,atol=1e-9)
    categories = csv("categories")
    conditional = csv("category_target")
    for c in CAT:
        observed = categories[categories.feature==c]
        assert observed["count"].sum() == len(train)
        assert_allclose(observed.pct.sum(),100,atol=1e-7)
        for _,r in observed.iterrows():
            group = train[train[c].astype(str)==str(r.category)]
            assert r["count"] == len(group) and r.cardinality == train[c].nunique()
            assert bool(r.rare) == (len(group)/len(train)<.01)
            assert_allclose(r.pct,100*len(group)/len(train),rtol=1e-8)
            out = conditional[(conditional.feature==c)&(conditional.category.astype(str)==str(r.category))].iloc[0]
            assert out.n == len(group) and out.stroke == group.stroke.sum()
            assert out.no_stroke == len(group)-group.stroke.sum()
            assert_allclose(out.stroke_pct,100*group.stroke.mean(),rtol=1e-8)
    correlation_matrix = csv("correlations").set_index("feature")
    pairs = csv("correlation_pairs")
    for _,r in pairs.iterrows():
        pair = train[[r.feature1,r.feature2]].dropna()
        # Pearson correlation of average ranks is an independent Spearman calculation.
        ranked = pair.rank(method="average").to_numpy().T
        rho = np.corrcoef(ranked)[0,1]
        assert r.n == len(pair)
        assert_allclose(r.spearman,rho,rtol=1e-8)
        assert_allclose(correlation_matrix.loc[r.feature1,r.feature2],rho,rtol=1e-8)
    best = pairs.loc[pairs.spearman.abs().idxmax()]
    assert m["strongest_pair"]["feature1"] == best.feature1
    assert m["strongest_pair"]["feature2"] == best.feature2
    assert_allclose(m["strongest_pair"]["spearman"],best.spearman,rtol=1e-8)

    flags = []
    outlier_table = csv("outliers").set_index("feature")
    for c in NUM:
        x = train[c].to_numpy()
        q1,q3 = np.nanquantile(x,[.25,.75])
        low,high = q1-1.5*(q3-q1),q3+1.5*(q3-q1)
        flag = (x<low)|(x>high)
        flags.append(flag)
        assert outlier_table.loc[c,"flagged"] == flag.sum()
        assert_allclose(outlier_table.loc[c,["lower_fence","upper_fence"]].to_numpy(dtype=float),[low,high])
    assert m["outlier_flagged_rows"] == np.any(flags,axis=0).sum() == 569
    assert m["outlier_rows_removed"] == m["outlier_values_clipped"] == 0
    for by in ["hypertension","work_type","stroke"]:
        grouped = csv(f"grouped_{by}")
        for _,r in grouped.iterrows():
            x = train.loc[train[by].astype(str)==str(r.group),r.feature].dropna().to_numpy()
            q1,median,q3 = np.quantile(x,[.25,.5,.75])
            assert r["count"] == len(x)
            assert_allclose(r[["q1","median","q3","iqr"]].to_numpy(dtype=float),[q1,median,q3,q3-q1])
    for label in [0,1]:
        assert_allclose(m["missing_bmi_by_target_train"][str(label)],100*train.loc[train.stroke==label,"bmi"].isna().mean())

    Xtrain, Xtest = train[FEATURES].copy(), test[FEATURES].copy()
    original_train, original_test = Xtrain.copy(deep=True), Xtest.copy(deep=True)
    unfitted = build_preprocessor()
    assert not hasattr(unfitted.named_steps["columns"],"transformers_")
    assert "id" not in FEATURES and "stroke" not in FEATURES
    pipeline = clone(unfitted)
    Xt = pipeline.fit_transform(Xtrain)
    state = pickle.dumps(pipeline)
    Xv = pipeline.transform(Xtest)
    assert state == pickle.dumps(pipeline), "transform must not alter fitted state"
    assert_frame_equal(Xtrain,original_train)
    assert_frame_equal(Xtest,original_test)
    assert list(Xt.shape) == m["train_shape"] == [4088,24]
    assert list(Xv.shape) == m["test_shape"] == [1022,24]
    assert np.isfinite(Xt).all() and np.isfinite(Xv).all()
    assert_array_equal(pipeline.get_feature_names_out(),csv("feature_names").feature.to_numpy())
    columns = pipeline.named_steps["columns"]
    medians = train[NUM].median().to_numpy()
    assert_allclose(columns.named_transformers_["num"]["impute"].statistics_,medians)
    imputed = train[NUM].fillna(dict(zip(NUM,medians))).to_numpy()
    center = np.median(imputed,axis=0)
    q1,q3 = np.quantile(imputed,[.25,.75],axis=0)
    assert_allclose(Xt[:,:3],(imputed-center)/(q3-q1))
    assert_array_equal(Xt[:,3],train.bmi.isna().to_numpy().astype(float))
    fitted_num = csv("fitted_numeric")
    assert_allclose(fitted_num.imputation_median,medians)
    assert_allclose(fitted_num.scaler_center,center)
    assert_allclose(fitted_num.scaler_iqr,q3-q1)
    encoder = columns.named_transformers_["cat"]["encode"]
    assert sum(map(len,encoder.categories_)) == 20
    for c,categories_ in zip(CAT,encoder.categories_):
        assert set(categories_) == set(train[c].unique())
    # Deliberately shift only held-out inputs; training state and output must stay fixed.
    changed = Xtest.copy()
    changed[NUM] = changed[NUM].fillna(0)+10000
    changed["gender"] = "UNSEEN_GENDER"
    transformed = pipeline.transform(changed)
    assert np.isfinite(transformed).all()
    assert state == pickle.dumps(pipeline)
    assert_allclose(pipeline.transform(Xtrain),Xt)
    gender_indices = [i for i,n in enumerate(pipeline.get_feature_names_out()) if n.startswith("cat__gender_")]
    assert_allclose(transformed[:,gender_indices],0)
    probe = Xtest.iloc[:4].copy()
    probe[NUM] = np.nan
    probe["smoking_status"] = np.nan
    probe["gender"] = "UNSEEN_GENDER"
    out = pipeline.transform(probe)
    assert out.shape == (4,24) and np.isfinite(out).all()
    assert_allclose(out[:,gender_indices],0)
    assert_array_equal(out[:,3],np.ones(4))
    assert state == pickle.dumps(pipeline)

    # Independent NumPy SVD verifies PCA explained variance and coefficients.
    _, singular, right = np.linalg.svd(Xt-Xt.mean(axis=0),full_matrices=False)
    variance_ratio = singular**2 / np.sum(singular**2)
    pca = csv("pca_variance")
    assert_allclose(pca.explained_ratio,variance_ratio,atol=1e-9,rtol=1e-8)
    assert_allclose(pca.cumulative_ratio,np.cumsum(variance_ratio),atol=1e-9)
    assert_allclose(m["pca_pc1_pc2"],variance_ratio[:2].sum())
    assert m["pca_components_90"] == np.searchsorted(np.cumsum(variance_ratio),.9)+1
    loads = csv("pca_loadings")
    for i,pc in enumerate(["PC1","PC2"]):
        assert_allclose(np.abs(loads[pc]),np.abs(right[i]),atol=1e-8)
    diagnostics = csv("projection_diagnostics")
    eval_idx = np.sort(np.random.default_rng(42).choice(len(Xt),size=1000,replace=False))
    for method in ["pca","tsne_30","tsne_50","umap_15","umap_50"]:
        embedding = csv(f"embedding_{method}")
        assert_array_equal(embedding.id,train.id)
        assert_array_equal(embedding.stroke,train.stroke)
        assert embedding.shape == (4088,4)
        assert np.isfinite(embedding[["dimension1","dimension2"]]).all().all()
        xy = embedding[["dimension1","dimension2"]].to_numpy()
        method_name,parameter = ("pca",2) if method=="pca" else (method.split("_")[0],int(method.split("_")[1]))
        diagnostic = diagnostics[(diagnostics.method==method_name)&(diagnostics.parameter==parameter)].iloc[0]
        score = trustworthiness(Xt[eval_idx],xy[eval_idx],n_neighbors=10)
        assert_allclose(diagnostic.trustworthiness,score,atol=1e-6)
        if method=="pca":
            assert_allclose(xy,(Xt-Xt.mean(axis=0)) @ loads[["PC1","PC2"]].to_numpy(),atol=1e-8,rtol=1e-8)
    assert len(csv("projection_diagnostics")) == 5
    assert csv("projection_diagnostics").trustworthiness.between(0,1).all()

    report = (BASE / "index.md").read_text()
    for key in ["project: eda","task: classification","ai_use:","Luca Santana Feltrin",
                "Eduardo Takei Yaginuma","Gabriel Fernando Missaka Mendes"]:
        assert key in report
    assert [int(x) for x in re.findall(r"^## ([0-5])\.",report,re.M)] == list(range(6))
    sections = re.split(r"^## ",report,flags=re.M)
    for number in [2,3]:
        section = next(s for s in sections if s.startswith(f"{number}."))
        for item in re.split(r"^### ",section,flags=re.M)[1:]:
            assert len(re.findall(r"!\[Figure",item)) <= 3
    figures = re.findall(r"!\[Figure (\d+)\.[^\]]*\]\((figures/[^)]+)\)",report)
    assert [int(n) for n,_ in figures] == list(range(1,13))
    for number,path in figures:
        assert (BASE / path).is_file()
        assert f"Conclusion — Figure {number}:" in report
    summary = report.split("### Results summary\n",1)[1]
    rows = re.findall(r"^\| (\d+) \|",summary,re.M)
    assert list(map(int,rows)) == list(range(1,11))
    assert summary.strip().endswith("|")
    assert f"{m['pca_pc1_pc2']*100:.2f}%" in summary
    assert f"ρ={m['strongest_pair']['spearman']:.4f}" in summary
    for name in pipeline.get_feature_names_out():
        assert name in report
    print("PASS: source/checksum; all split, quality, target, numerical, category, correlation,")
    print("outlier and grouped statistics; train-only fitted parameters; held-out perturbation;")
    print("missing/unseen-category contract; independent PCA SVD; embedding IDs; report structure,")
    print("all 12 figures and conclusions; feature names; all 10 results-summary rows.")


if __name__ == "__main__":
    main()
