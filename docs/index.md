# Artificial Neural Networks and Deep Learning

**Luca Santana Feltrin** — Insper, 2026.2
Instructor: Humberto Sandmann

---

## About this site

This is my portfolio for the *Artificial Neural Networks and Deep Learning* course. Each deliverable
gets its own folder, and it grows one folder at a time throughout the semester: the report, the code
that produced every number in it, and the figures the report shows.

The repository behind this site is public: [github.com/LucaFeltrin14/ann-dl](https://github.com/LucaFeltrin14/ann-dl).

## Exercises

| Exercise | Due | Status |
|---|---|---|
| [Data](exercises/data/index.md) — data preparation and analysis for neural networks | Sep 10 | ✅ delivered |
| Perceptron | Sep 22 | — |
| MLP | Oct 13 | — |
| VAE | Oct 22 | — |

## How to reproduce

Everything here is reproducible from a clean checkout. Each exercise ships its scripts under
`code/`; they fix the random seed (`np.random.default_rng(42)`), print every reported number to
stdout, and regenerate the figures in `figures/`.

```shell
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --upgrade
python docs/exercises/data/code/exercise1_point_clouds.py
```

## Course links

- [Course site](https://insper.github.io/ann-dl/2026.2/)
- [Submission format](https://insper.github.io/ann-dl/2026.2/exercises/submission/)
