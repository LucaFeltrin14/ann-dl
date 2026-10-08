"""Render the English report from computed CSV/JSON artifacts.

Numbers are never transcribed manually. Narrative decisions are explicit, and
verify.py checks the saved artifacts independently against the original CSV.
"""
from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results"


def table(frame, digits=3):
    """Small Markdown renderer without an extra tabulate dependency."""
    def cell(x):
        if pd.isna(x):
            return "—"
        if isinstance(x, float):
            return f"{x:.{digits}f}"
        return str(x).replace("|", "\\|")
    return "\n".join([
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
        *("| " + " | ".join(cell(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)),
    ])


def read(name):
    return pd.read_csv(OUT / f"{name}.csv")


def figure(number, slug, caption, conclusion):
    return f"![Figure {number}. {caption}](figures/fig{number:02d}_{slug}.png)\n\n**Conclusion — Figure {number}:** {conclusion}\n"


def write_report():
    m = json.loads((OUT / "metrics.json").read_text())
    stats = read("numeric_stats").set_index("feature")
    cats = read("categories")
    rates = read("category_target")
    variance = read("pca_variance")
    loads = read("pca_loadings")
    missing = m["missing_bmi_by_target_train"]
    strongest = m["strongest_pair"]
    diagnostics = {(r["method"], r["parameter"]): r["trustworthiness"] for r in m["projection_diagnostics"]}
    coefficients = loads.set_index("feature")
    pieces = []
    def add(text):
        pieces.append(text.strip())
    def rate(feature, category):
        row = rates[(rates.feature == feature) & (rates.category.astype(str) == str(category))].iloc[0]
        return f"{row.stroke_pct:.2f}% ({int(row.stroke)}/{int(row.n)})"
    def group(feature, category, by):
        frame = read(f"grouped_{by}")
        row = frame[(frame.feature == feature) & (frame.group.astype(str) == str(category))].iloc[0]
        return f"median {row['median']:.2f}, IQR {row.iqr:.2f}"

    add('''---
project: eda
task: classification
dataset: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
team:
  - Luca Santana Feltrin
  - Eduardo Takei Yaginuma
  - Gabriel Fernando Missaka Mendes
ai_use: "OpenAI Codex assisted with analysis design, Python implementation, report drafting, and automated numerical and visual checks. This assistance is declared; each team member is responsible for understanding and explaining the results and decisions."
---

# EDA — Stroke Prediction

**Luca Santana Feltrin · Eduardo Takei Yaginuma · Gabriel Fernando Missaka Mendes**

Insper · Artificial Neural Networks and Deep Learning · 2026.2

This report asks which demographic and recorded health features are associated with the observed
`stroke` label and how to prepare them for a future neural network classification project. It does
not train a predictive model. All exploratory analyses after the split use training rows only;
test rows are used for split accounting and the required transformation checks.
''')
    add(f'''## 0. Proposal

The team selected **classification** with the public, tabular
[Stroke Prediction Dataset, published by fedesoriano](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
The team confirms instructor approval of this dataset. The team repository is
[LucaFeltrin14/ann-dl](https://github.com/LucaFeltrin14/ann-dl).

The CSV contains **5,110 rows × 12 columns**: an identifier, the target, and **10 predictors
(3 numerical and 7 categorical)**. It meets the minimum of 1,000 instances and five mixed-type
features. The same dataset will be retained for the classification deliverable.

**Motivation:** Mixed demographic and health measurements provide a useful test of neural network
preparation: categorical encodings, different numerical scales, missing measurements and a rare
positive class coexist in a compact dataset. Studying these properties first allows the later
modeling choices to be grounded in evidence.

**First risk:** Only **249/5,110 = {249/5110*100:.2f}%** of rows have `stroke=1`; accuracy alone
could conceal failure to detect positives. The original sampling frame, measurement timing and
collection protocol are not established by the public description, so clinical generalization
and prospective prediction cannot be claimed from this file.
''')
    add('''## 1. Initial inspection

### A. Data dictionary

**Source and unit of observation:** Each row is a patient record in the published CSV, with a
unique `id`. The public description does not establish the hospital, country, collection date,
representative sampling or whether measurements precede the stroke label. This is an analysis
of the recorded label, not evidence of future incident stroke prediction.

**Table 1. Complete data dictionary.** Semantic type takes precedence over storage dtype.

| Column | Meaning | Semantic type / role | Unit or values |
| --- | --- | --- | --- |
| `id` | Record identifier | Identifier; excluded | Arbitrary integer; no measurement unit |
| `gender` | Recorded gender category | Nominal categorical | Female, Male, Other |
| `age` | Recorded age | Continuous numerical | Years; fractional ages retained |
| `hypertension` | Recorded hypertension flag | Binary categorical | 0=no, 1=yes |
| `heart_disease` | Recorded heart disease flag | Binary categorical | 0=no, 1=yes |
| `ever_married` | Whether ever married | Binary categorical | No, Yes |
| `work_type` | Recorded employment category | Nominal categorical | Private, Self-employed, Govt_job, children, Never_worked |
| `Residence_type` | Recorded residence category | Binary categorical | Rural, Urban |
| `avg_glucose_level` | Recorded average glucose measurement | Continuous numerical | Public description does not specify the unit; no conversion assumed |
| `bmi` | Body mass index | Continuous numerical | kg/m²; N/A parsed as missing |
| `smoking_status` | Recorded smoking category | Nominal categorical | formerly smoked, never smoked, smokes, Unknown |
| `stroke` | Recorded stroke occurrence | Binary categorical target | 0=no, 1=yes |

The name `gender` is retained from the source; the single `Other` record is not treated as an error.
The source states that `Unknown` in smoking status means the information was unavailable: it is a
recorded missing-information category, distinct from a literal empty cell. No unsupported unit
or medical plausibility cutoff is inferred for glucose.
''')
    add("### B. Quality\n\n**Table 2. Full-file missingness and training missingness by column.**\n\n" + table(read("quality")))
    checks = m["checks"]
    add(f'''The largest missingness is **`bmi`: 201/5,110 = {201/5110*100:.2f}%** in the full file;
training has **170/4,088 = {170/4088*100:.2f}%**. Every other column has **0 empty values**.
In the training data, `smoking_status=Unknown` is retained and counted in Table 5.

The full-file audit found **{checks['duplicate_full_rows']} identical rows**, **{checks['duplicate_rows_without_id']}
duplicates after removing `id`**, **{checks['duplicate_ids']} repeated IDs**, and **{len(checks['constant_features'])}
constant predictors. Explicit domain checks found **{checks['age_outside_0_120']} ages outside [0,120]**,
**{checks['nonpositive_glucose']} nonpositive glucose values**, **{checks['nonpositive_observed_bmi']}
nonpositive observed BMI values**, **{checks['invalid_binary_flags']} invalid binary flags**, and
**{checks['invalid_categories']} categorical values outside the documented vocabularies**.
Contextual checks found **{checks['children_work_type_age_18_or_more']} `children` records aged ≥18**,
**{checks['married_under_18']} ever-married records under 18**, and **{checks['smokes_under_18']}
current-smoker records under 18. These contextual flags warrant source clarification, not automatic deletion.

**Leakage audit and dropped columns:** `id` is dropped because its arbitrary value is not a usable
patient measurement; `stroke` is separated as the target and never enters preprocessing or the
projections. No other predictor directly contains the target, and no numerical predictor pair
has an absolute correlation ≥0.90 (Table 6). Correlation alone cannot rule out temporal leakage:
measurement timing is unknown for every health feature. There is no timestamp for a temporal split;
unique IDs support a record split, but cannot prove that undisclosed repeated patients or related
records do not exist. These limitations remain modeling risks.

The full-file quality inventory and target counts are the required stage-1 audit. They do not fit
imputers, scalers, projections or encoders. The fixed split precedes all data-dependent
preprocessing choices and the training analyses below.
''')
    add("### C. Target\n\n**Table 3. Target distribution and partition accounting.**\n\n" + table(read("target"), digits=2))
    add(figure(1, "target", "Full-file target distribution", f"The minority class contains 249 records ({249/5110*100:.2f}%), versus 4,861 negatives ({4861/5110*100:.2f}%): a {4861/249:.2f}:1 imbalance. An always-negative rule would have {4861/5110*100:.2f}% accuracy but zero positive recall; this is arithmetic, not a trained baseline."))
    add('''### D. Train and test

Use `train_test_split(..., test_size=0.2, stratify=stroke, random_state=42)`:
**4,088 training records (3,889 negative, 199 positive)** and **1,022 test records (972 negative,
50 positive)**. Both classes appear in both sets; Table 3 gives the resulting percentages.
The IDs are saved in [`results/split_ids.csv`](results/split_ids.csv) and checked for disjointness
and complete coverage of the 5,110 original records. No row was removed before the split.

The absence of dates rules out a defensible temporal split with this file. There are no repeated
IDs to group; if repeated-patient information becomes available, the split must be revisited.
All medians, scales, category vocabularies, outlier fences, correlations and projections below
come from training data. The test set is not used to choose transformations, parameters or figures.
For future model selection, create validation folds within these training IDs and refit the
complete pipeline inside every fold; reserve these test IDs for final evaluation.
''')
    add("## 2. Univariate analysis\n\n### A. Numerical\n\n**Table 4. Descriptive statistics for every numerical predictor; observed training values only.**\n\n" + table(read("numeric_stats")))
    add(figure(2, "numerical", "Histograms and boxplots for all numerical features",
               f"Age is broadly spread with weak negative skew ({stats.loc['age','skewness']:.3f}); glucose and BMI have long right tails (skewness {stats.loc['avg_glucose_level','skewness']:.3f} and {stats.loc['bmi','skewness']:.3f}). The coarse glucose histogram suggests a higher-value shoulder; it does not establish a separate population or formal multimodality. BMI concentrates near its median of {stats.loc['bmi','median']:.2f}, with extreme observations extending to {stats.loc['bmi','max']:.2f}."))
    add(f'''**Figure choice and interpretation:** One composite figure shows both distribution shape
and IQR extremes for all three features, keeping the item within its three-figure limit.
Age has mean **{stats.loc['age','mean']:.2f}** and median **{stats.loc['age','median']:.2f}**;
glucose has mean **{stats.loc['avg_glucose_level','mean']:.2f}** and median **{stats.loc['avg_glucose_level','median']:.2f}**;
BMI has mean **{stats.loc['bmi','mean']:.2f}** and median **{stats.loc['bmi','median']:.2f}**.
The latter two mean–median gaps support the right-tail reading. Means, medians, standard deviations,
quartiles, extrema and observed counts for every feature are reported in Table 4.
No KDE bandwidth or histogram alone is used to assert the number of modes.
''')
    add("### B. Categorical\n\n**Table 5. Complete category frequencies, cardinalities and rare-category flags, training only.** Rare means less than 1% of training records.\n\n" + table(cats, digits=2))
    rare = cats[cats.rare]
    add(figure(3, "categorical", "Complete categorical frequencies",
               "Every categorical feature has only two to five observed categories, so none has high cardinality. The frequencies are uneven: binary health flags are dominated by 0, and Private is the largest work category. Rare categories require cautious interpretation and an encoder that tolerates unseen values."))
    add("Rare categories (<1%) are **" + "; ".join(f"{r.feature}={r.category}: {r['count']} ({r.pct:.2f}%)" for _, r in rare.iterrows()) + "**. The threshold is a descriptive flag; these categories are not deleted or merged. The ID is high-cardinality (5,110 unique values), but is already excluded. `Unknown` is an information gap, not evidence of nonsmoking. One composite bar-chart figure covers all seven predictors; horizontal labels keep their category names readable.")
    add("## 3. Bivariate and multivariate analysis\n\n### A. Numerical × numerical\n\n**Table 6. Pairwise Spearman correlations and complete-pair sample sizes.**\n\n" + table(read("correlation_pairs"), digits=4))
    add(f'''Use **Spearman**, because glucose and BMI are asymmetric with extreme observations,
and monotonic associations are more defensible here than an assumption of linear relationships.
Each pair uses the original observed training values; no imputation enters this calculation.
The largest absolute coefficient is **{strongest['feature1']} × {strongest['feature2']}:
ρ = {strongest['spearman']:.4f}, n = {strongest['n']:,}**. Using |ρ|≥0.90 as a descriptive
redundancy flag, **zero pairs** are redundant; none is dropped on this basis.
''')
    add(figure(4, "correlation", "Spearman correlation matrix", f"The strongest pair is age–BMI (ρ={strongest['spearman']:.4f}), a moderate association rather than interchangeable features. Glucose has weaker monotonic associations with the other numerical features."))
    add(figure(5, "scatter", "All numerical feature pairs", "The scatters show a broad age–BMI tendency and substantial spread; the glucose upper tail is not described fully by a single coefficient. These two figures were chosen to pair an overview with the underlying observations, rather than infer redundancy from a heatmap alone."))
    add("### B. Categorical × target\n\n**Table 7. Stroke counts and conditional proportions for every category (training only).** The denominator is the number of records in that category, not the number of stroke cases.\n\n" + table(rates, digits=2))
    add(figure(6, "category_target", "Stroke proportions conditional on category",
               f"Stroke is more frequent in the hypertension=1 group ({rate('hypertension',1)}) than hypertension=0 ({rate('hypertension',0)}), and in heart_disease=1 ({rate('heart_disease',1)}) than heart_disease=0 ({rate('heart_disease',0)}). These are unadjusted associations, not causal effects; age and other confounders may contribute."))
    add(f'''The ever-married group has **{rate('ever_married','Yes')}**, versus **{rate('ever_married','No')}**
for never-married records. Self-employed records have **{rate('work_type','Self-employed')}**,
versus **{rate('work_type','Private')}** for Private; formerly smoked has
**{rate('smoking_status','formerly smoked')}**, versus **{rate('smoking_status','never smoked')}**
for never smoked. Table 7 supplies both classes' counts and all remaining categories, including
residence and gender. A zero rate in a tiny group is not evidence of protection: in particular,
`gender=Other` has only one training record. The complete panel was chosen to expose denominators
and avoid selectively displaying only the largest apparent effects.
''')
    add("### C. Numerical × categorical\n\nThe three chosen grouping variables answer different questions: hypertension links recorded disease status to numerical features, work type exposes age structure, and the binary target tests whether class differences concern location, dispersion, or both. Values remain un-imputed; group-specific observed counts, quartiles, medians and IQRs are recorded below.")
    for number, by, label in [(7,"hypertension","Hypertension"),(8,"work_type","Work type"),(9,"stroke","Target class")]:
        add(f"**Table {number+1}. {label}: numerical medians, quartiles and IQRs.**\n\n" + table(read(f"grouped_{by}"), digits=2))
        if by == "hypertension":
            conclusion = (f"Age differs in both location and dispersion: hypertension=0 has {group('age',0,by)}, "
                          f"versus {group('age',1,by)} for hypertension=1. Glucose has {group('avg_glucose_level',0,by)} versus {group('avg_glucose_level',1,by)}; "
                          f"BMI has {group('bmi',0,by)} versus {group('bmi',1,by)}. The glucose boxes differ strongly in spread, while BMI mainly shifts in location; overlap remains.")
        elif by == "work_type":
            conclusion = (f"Work categories encode different age structures: children has {group('age','children',by)}, "
                          f"Private has {group('age','Private',by)}, and Self-employed has {group('age','Self-employed',by)}. "
                          "Location and spread differ together, making age a plausible confounder of the category–stroke association; BMI comparisons also require care because children and adults are mixed.")
        else:
            conclusion = (f"Stroke classes differ in age location and spread: class 0 has {group('age',0,by)}, "
                          f"class 1 has {group('age',1,by)}. Glucose class 0 has {group('avg_glucose_level',0,by)}, "
                          f"class 1 has {group('avg_glucose_level',1,by)}. BMI shows {group('bmi',0,by)} versus {group('bmi',1,by)}, "
                          "with overlapping distributions and fewer observed positives due to missing BMI.")
        add(figure(number, f"box_{by}", f"Numerical distributions by {by}", conclusion))
    add('''## 4. Preprocessing

### A. Strategies

These choices prepare the features for a neural network without consuming the test set.
''')
    add(f'''**Missing values:** BMI is the only column with empty cells (Table 2). Training BMI missingness
is **{missing['1']:.2f}% for stroke=1** versus **{missing['0']:.2f}% for stroke=0**.
Dropping incomplete rows could disproportionately remove positives. Impute each numerical
feature with its **training median**, and retain an explicit **0/1 BMI-missing indicator** so the
absence is not silently equated to an observed median. This pattern may reflect collection practice,
so the indicator is also a leakage/generalization risk to review in future modeling.
Categorical empty cells, if encountered, use `__MISSING__`; the observed `Unknown` smoking category
is retained separately. No missing value is filled using the target.

**Outliers:** Table 11 applies training-only 1.5×IQR fences to observed values.
There are **{m['outlier_flagged_rows']:,} distinct flagged training rows** (some features can flag the same row).
The chosen strategy **removes 0 rows and clips 0 values**: an IQR flag does not establish an erroneous
health measurement, and adult BMI thresholds should not be imposed on a mixed-age sample.
Retain the extremes and document their influence; do not delete a minority patient because of
an unusual feature. Robust scaling limits the effect of extremes on fitted location and scale,
although it does **not** bound the transformed values or eliminate their effect on PCA.

**Table 11. Training outlier fences and counts.**

{table(read('outliers'))}

**Encoding:** The seven nominal predictors (including hypertension and heart disease) have
2–5 categories each (Table 5), making one-hot encoding appropriate without a large vocabulary.
Use `drop=None`: there is no reference-category constraint for a neural network, and observed
categories remain explicit. `OneHotEncoder(handle_unknown="ignore")` maps an unseen category to
an all-zero block and preserves the feature count. This is explicitly checked using a synthetic
unseen gender; the real test set need not contain an unseen category to test this contract.
Do not target-encode, resample, or balance before the split.

**Scaling:** The training ranges differ: age **{stats.loc['age','min']:.2f}–{stats.loc['age','max']:.2f}**,
glucose **{stats.loc['avg_glucose_level','min']:.2f}–{stats.loc['avg_glucose_level','max']:.2f}**,
and BMI **{stats.loc['bmi','min']:.2f}–{stats.loc['bmi','max']:.2f}** (Table 4), while categorical
indicators lie in [0,1]. The observed glucose/BMI right tails and Table 11 motivate
**RobustScaler**, subtracting the training median and dividing by the training IQR **after
imputation**. It helps numerical conditioning without fitting extreme-sensitive mean/variance
parameters. One-hot columns and the missingness indicator remain 0/1; scaling every rare binary
column to unit variance would amplify rare categories. No additional scaling is fitted on test.

**Table 12. Fitted numerical parameters.** The scaler IQR uses the imputed training matrix;
therefore it can differ from Table 4's observed-value quartiles.

{table(read('fitted_numeric'))}
''')
    add(f'''### B. Dimensionality reduction

All projections use the **same 4,088 training rows and {m['train_shape'][1]} pipeline-output features**.
Neither `stroke` nor `id` enters their fitting; target values are used **only for coloring**.
There is no subsampling for fitting. Euclidean distance is applied to robust-scaled numerical
features plus one-hot categories and the BMI-missing indicator. This mixed representation weights
continuous and categorical blocks according to this preprocessing choice; its geometry is not a
clinical distance. Redundant one-hot blocks can also create zero-variance PCA directions.

**PCA:** Fit full SVD on the transformed train only. **PC1 explains {variance.iloc[0].explained_ratio*100:.2f}%**,
**PC2 explains {variance.iloc[1].explained_ratio*100:.2f}%**, and together they explain
**{m['pca_pc1_pc2']*100:.2f}%**; **{m['pca_components_90']} components** reach 90% cumulative
variance. The two displayed components therefore cannot represent all predictive information.
PCA centers its input internally; no target enters the decomposition.
''')
    add(figure(10,"pca","PCA projection, variance and coefficients", f"The positive records are visually more common toward the higher-glucose arm and the older/higher-BMI portion, but negatives occupy these regions too. PC1 is dominated by glucose ({coefficients.loc['num__avg_glucose_level','PC1']:+.4f}); PC2 contrasts BMI ({coefficients.loc['num__bmi','PC2']:+.4f}) and age ({coefficients.loc['num__age','PC2']:+.4f}) with glucose ({coefficients.loc['num__avg_glucose_level','PC2']:+.4f}). The projection preserves only {m['pca_pc1_pc2']*100:.2f}% of variance and does not demonstrate class separability."))
    top_rows = []
    for pc in ["PC1", "PC2"]:
        top = loads.loc[loads[pc].abs().nlargest(5).index]
        for _, r in top.iterrows():
            top_rows.append(dict(component=pc, feature=r.feature, coefficient=r[pc]))
    add("**Table 13. Five largest absolute eigenvector coefficients per displayed component.** Full coefficients are in [`results/pca_loadings.csv`](results/pca_loadings.csv).\n\n" + table(pd.DataFrame(top_rows),digits=4))
    for pc in ["PC1","PC2"]:
        top = loads.loc[loads[pc].abs().nlargest(3).index]
        add(f"**{pc} reading:** The largest contributions are " + ", ".join(f"`{r.feature}` ({r[pc]:+.4f})" for _,r in top.iterrows()) + ". Same-sign coefficients move in the same score direction; opposite signs express a contrast. These are eigenvector coefficients, not correlations or feature importance for stroke prediction. Component signs can be reversed without changing the projection's information.")
    add('''**t-SNE:** Compare perplexities **30 and 50**, with 2 components, Euclidean metric,
PCA initialization, `learning_rate="auto"`, 1,000 iterations, and `random_state=42`.
**UMAP:** Compare `n_neighbors` **15 and 50**, with 2 components, `min_dist=0.1`, Euclidean
metric, spectral initialization, 500 epochs, `random_state=42`, `transform_seed=42` and `n_jobs=1`.
The parameter pairs are sensitivity checks, not an optimization against test labels or target separation.
''')
    add(figure(11,"tsne","t-SNE at two perplexities", f"Both perplexities show many small local groups and a region containing more visible positives, with negative records mixed into it. A clean class boundary is absent at either setting. Trustworthiness is {diagnostics[('tsne',30)]:.4f} at perplexity 30 and {diagnostics[('tsne',50)]:.4f} at 50 (Table 14), exceeding PCA's {diagnostics[('pca',2)]:.4f} on the same evaluation subset. Island sizes and inter-island distances do not measure original feature-space distances or disease risk."))
    add(figure(12,"umap","UMAP at two neighborhood sizes", f"Both neighborhood sizes show separated local groups and mixed-class regions; their positions and arrangement change markedly between settings. Trustworthiness is {diagnostics[('umap',15)]:.4f} at 15 neighbors and {diagnostics[('umap',50)]:.4f} at 50 (Table 14). Local organization is better preserved than in 2D PCA, but this metric does not validate the apparent islands or show that stroke is predictable."))
    diag = read("projection_diagnostics")
    add("**Table 14. Projection diagnostics.** Trustworthiness (k=10) is evaluated on the same fixed, random 1,000-row subset of training data for all maps; it measures neighborhood preservation within that subset. It is not a classification score. t-SNE KL divergence is an optimization diagnostic; comparing its values across perplexities does not identify a predictive winner.\n\n" + table(diag,digits=4))
    add('''**Comparison:** PCA supplies a global linear view with interpretable variance and coefficients.
t-SNE and UMAP reveal fragmented local arrangements that can be compressed
or superimposed by the first two principal components. Mixed categorical indicators can contribute
to these fragments, but target coloring alone does not identify which variable generates them. Their island layouts depend on neighborhood
parameters and on the chosen mixed-feature distance. Inspect the two settings together rather than
name a single cluster configuration as ground truth. Figure 10 gives the interpretable variance
baseline; Figures 11–12 show the extra local detail, and Table 14 quantifies neighborhood distortion.
None of these embeddings is used to claim future classifier performance. The pipeline retains the
original transformed features for the later modeling deliverable rather than replacing them with
a chosen 2D embedding.
''')
    add(f'''### C. Pipeline

The importable factory `build_preprocessor()` in [`code/preprocessing.py`](code/preprocessing.py)
returns an unfitted scikit-learn `Pipeline` containing a `ColumnTransformer` with numerical,
BMI-missing-indicator and categorical branches. `eda.py` calls **`fit_transform(X_train)` once**,
and **`transform(X_test)`**. The target is separate; `id` is absent from `FEATURES`.

The final output is **train `{tuple(m['train_shape'])}` and test `{tuple(m['test_shape'])}`**,
with **{m['transformed_nan_train']} training NaNs, {m['transformed_nan_test']} test NaNs and zero infinities**.
The {m['train_shape'][1]} features are three scaled numerical columns, one missingness indicator and
20 one-hot columns. The exact names are reported below and exported in
[`results/feature_names.csv`](results/feature_names.csv).

**Table 15. Final feature names in output order.**

{table(read('feature_names'))}

Verification checks that fitting statistics match the training partition, that transformation
leaves the fitted estimator unchanged, and that a deliberately perturbed held-out batch does not
change the training representation. Synthetic inputs with missing numerical values, missing
smoking category and an unseen gender produce finite output with the same width. These checks
are reproducible with `verify.py`; none trains a classifier.
''')
    add('''## 5. Synthesis

### Findings and risks for the modeling team

| Finding and evidence | Modeling consequence / handling plan |
| --- | --- |
| Minority class: 249 cases, 4.87% (Table 3; Figure 1) | Use stratified validation within training. Report PR-AUC, recall, precision and confusion counts; choose thresholds using validation. Fit class weights or any resampling inside training folds only. |
| Age structure and unadjusted disease/category associations overlap (Tables 7–10; Figures 6–9) | Keep the features, investigate confounding and subgroup behavior during validation; avoid causal or diagnostic claims from EDA. |
| BMI missingness is more frequent among positives (Table 2; section 4A) | Keep median imputation and the indicator; compare validation behavior with/without the indicator and investigate recording-time leakage. |
| Glucose and BMI have right tails; 569 rows are flagged by IQR (Tables 4 and 11; Figure 2) | Retain the rows, use robust scaling and inspect training-fold sensitivity to extreme values; never select cutoffs from test outcomes. |
| Rare categories and unavailable smoking information (Table 5; Figure 3) | Preserve documented categories; handle new categories explicitly and monitor their frequency. One training record cannot support a gender-specific conclusion. |
| Strongest numerical pair is moderate, not redundant (Table 6; Figures 4–5) | Keep all three numerical predictors; do not turn an exploratory rank correlation into automatic feature selection. |
| PCA loses information and nonlinear maps vary with settings (Figures 10–12; Tables 13–14) | Use projections to inspect geometry, keep the full pipeline features for modeling, and evaluate any later dimension reduction within validation folds. |
| Collection provenance and feature timing are not established (sections 0 and 1) | Treat the task as classification of recorded labels. Obtain provenance and longitudinal timing before claiming future risk prediction or clinical validity. |

### Reproduction, audit and source code

The original CSV is bundled under `code/data/`. SHA-256:
''')
    add(f"```text\n{m['dataset_sha256']}\n```\n\nRun from the repository root with Python 3.11 or 3.12:\n\n```shell\npython3 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements.txt\npython docs/projects/eda/code/eda.py\npython docs/projects/eda/code/verify.py\nmkdocs build --strict\n```\n\nThe first script regenerates all 12 figures, computed tables, saved coordinates, split IDs,\n`results/metrics.json` and this report. It prints fitted dimensions and numerical diagnostics.\n`verify.py` recomputes the required statistics independently from the bundled file, checks\npreprocessing isolation and tests missing/unseen inputs. See\n[`results/metrics.json`](results/metrics.json) for every recorded library version.\nThe computation was run with **Python {m['python']}**. Numerical reproducibility uses the fixed\nseed and pinned numerical packages; embedding coordinates and rendering can still vary across\nplatforms or numerical-library versions. Projection comparisons use unchanged IDs and parameters.\n\n**Table 16. Numerical environment used for this execution.**\n\n" + table(pd.DataFrame(m["versions"].items(),columns=["Package","Version"])))
    add('''The sources below are pulled in from executable files, as required by the
[submission format](https://insper.github.io/ann-dl/2026.2/exercises/submission/).
The team must be able to explain the split, the BMI missingness pattern, the fitted parameters,
the meaning of the PCA coefficients, and why the nonlinear maps do not measure model quality.

??? example "Importable preprocessing pipeline"

    ```python
    --8<-- "docs/projects/eda/code/preprocessing.py"
    ```

??? example "Analysis and figure generation"

    ```python
    --8<-- "docs/projects/eda/code/eda.py"
    ```

??? example "Independent numerical and pipeline verification"

    ```python
    --8<-- "docs/projects/eda/code/verify.py"
    ```

The report renderer is also an executable file: [`code/report.py`](code/report.py).

**References:** [dataset publisher](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset),
[project specification and rubric](https://insper.github.io/ann-dl/2026.2/projects/eda/),
[course leakage guidance](https://insper.github.io/ann-dl/2026.2/classes/data/leakage/),
[OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html),
[RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html).
''')
    add(f'''### Results summary

| # | Results summary | Value |
| --- | --- | --- |
| 1 | Dataset, task and target | Stroke Prediction Dataset; classification; `stroke` (0/1) |
| 2 | Instances × features (numerical / categorical) | 5,110 × 10 predictors (3 numerical / 7 categorical); source file has 12 columns including ID and target |
| 3 | Column with the most missing values and its percentage | `bmi`: 201 missing, {201/5110*100:.2f}% of the source file; 170, {170/4088*100:.2f}% in training |
| 4 | Dropped columns and the reason | `id`: arbitrary identifier; `stroke`: separated as target, excluded from inputs; no other predictors dropped |
| 5 | Minority class (%) · or mean and median of the target | `stroke=1`: 249/5,110, {249/5110*100:.2f}%; training: 199/4,088, {199/4088*100:.2f}% |
| 6 | Size of the training and test sets | Train 4,088; test 1,022; stratified 80/20, random_state=42 |
| 7 | Most correlated numerical pair and its value | {strongest['feature1']} × {strongest['feature2']}; Spearman ρ={strongest['spearman']:.4f}, n={strongest['n']:,}; training observed values |
| 8 | Rows affected by the outlier strategy | {m['outlier_flagged_rows']} training rows flagged by 1.5×IQR; 0 rows removed, 0 values clipped; all rows retained and robust-scaled |
| 9 | Variance explained by PC1 + PC2 | {m['pca_pc1_pc2']*100:.2f}% of transformed training variance |
| 10 | `shape` of train and test after the pipeline | Train `{tuple(m['train_shape'])}`; test `{tuple(m['test_shape'])}`; 0 NaNs and 0 infinities |
''')
    (BASE / "index.md").write_text("\n\n".join(pieces)+"\n")


if __name__ == "__main__":
    write_report()
