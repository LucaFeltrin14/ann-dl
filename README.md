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
```

Every script fixes `numpy.random.default_rng(42)`, prints all reported numbers to stdout and writes
its figures into the matching `figures/` folder, so every value in a report can be reproduced from a
clean checkout.

## Serving the site locally

```shell
mkdocs serve -o
```

---

Based on the course [documentation template](https://github.com/hsandmann/documentation.template)
by Sandmann, H.
