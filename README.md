# Artificial Neural Networks and Deep Learning — Insper, 2026.2

Portfolio repository for the **Artificial Neural Networks and Deep Learning** course at Insper
(2026.2), by **Luca Santana Feltrin**.

📄 **Published site:** <https://lucafeltrin14.github.io/ann-dl/>

Every deliverable of the course lives in this single repository and is published as a GitHub Pages
site by the workflow in [`.github/workflows/main.yaml`](.github/workflows/main.yaml) on every push
to `main`.

## Layout

```
docs/
  index.md                          landing page
  exercises/
    data/
      index.md                      the report
      code/                         the scripts that were actually run
      figures/                      the figures the report shows
    perceptron/
      index.md                      the report
      code/                         the perceptron and the scripts that were actually run
      figures/                      the figures the report shows
    mlp/
      index.md                      the report
      code/                         the MLP, the scripts that were actually run and the checks
      figures/                      the figures the report shows
  projects/
    eda/
      index.md                      the English EDA report
      code/                         importable pipeline, analysis and verification
      code/data/                    original Stroke Prediction CSV
      figures/                      12 numbered figures
      results/                      computed tables, split IDs and projection coordinates
mkdocs.yml
requirements.txt
```

## Running the code

```shell
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt --upgrade

python docs/exercises/data/code/exercise1_point_clouds.py
python docs/exercises/data/code/exercise2_nonlinearity.py
python docs/exercises/data/code/exercise3_spaceship_titanic.py

python docs/exercises/perceptron/code/exercise1_separable.py
python docs/exercises/perceptron/code/exercise2_overlapping.py

python docs/exercises/mlp/code/gradient_check.py
python docs/exercises/mlp/code/exercise2_binary.py
python docs/exercises/mlp/code/exercise3_multiclass.py
python docs/exercises/mlp/code/exercise4_depth.py
```

Every script fixes `numpy.random.default_rng(42)`, prints all reported numbers to stdout and writes
its figures into the matching `figures/` folder, so every value in a report can be reproduced from a
clean checkout.

## EDA project

Team: **Luca Santana Feltrin, Eduardo Takei Yaginuma, Gabriel Fernando Missaka Mendes**.
The approved dataset is [Stroke Prediction](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
The [report](docs/projects/eda/index.md) includes every rubric item, all 10 summary rows and an explicit AI-use declaration.
No predictive model is trained in this deliverable.

```shell
python docs/projects/eda/code/eda.py
python docs/projects/eda/code/verify.py
mkdocs build --strict
```

The EDA command regenerates the report, figures and numeric artifacts using only training data
after the fixed stratified split. The audit independently checks the numbers and pipeline isolation.
The source CSV is bundled, so these commands require no Kaggle login or dataset download.

## Serving the site locally

```shell
mkdocs serve -o
```

---

Based on the course [documentation template](https://github.com/hsandmann/documentation.template)
by Sandmann, H.
