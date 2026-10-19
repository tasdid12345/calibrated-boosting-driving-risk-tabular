# Calibrated LightGBM vs Logistic Regression for Road-Hazard Risk Probabilities

Motivation: "VISION-RISK: Vision-Language Model for Risk Assessment in Autonomous Driving" (2025) proposes a vision-language model that scores risk in driving scenes. That model needs image and language inputs and cannot be reproduced in a small, self-contained experiment. This repository therefore asks a narrower question about the probability quality of risk scores on tabular data.

Question: Can a class-balanced, calibrated LightGBM model give more reliable risk probabilities than a standard logistic-regression baseline on a public tabular driving-hazard stand-in dataset?

Dataset: a public binary road-accident/hazard table loaded from OpenML with the `openml` Python package. The data is a stand-in and is not the paper's own dataset (`same_as_paper: false`). The split is stratified 75/25 with seed 42, and rows are subsampled to at most 200,000.

Baseline: standardised scikit-learn logistic regression.

Improved model: class-weighted LightGBM with a small cross-validated hyperparameter search, followed by isotonic calibration. Both models use the same loading, preprocessing, split, and evaluation code.

Metrics: PR-AUC on the held-out test set is the main metric. Brier score and macro-F1 (threshold chosen by cross-validation on the training split) are secondary.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to `results.json` (and a chart to `figures/pr_auc_bar.png`).

## Results (latest run)

On the public road-safety dataset (OpenML 44038, held-out test split), a standard logistic-regression risk classifier reached PR-AUC 0.736, while my class-weighted, isotonically calibrated LightGBM reached 0.898 on the same split.

| Metric | Value |
|---|---|
| baseline_score | 0.7359 |
| improved_score | 0.8983 |
| improvement | 0.1624 |
| baseline_score_std | 1.11e-16 |
| improved_score_std | 0.001511 |
| baseline_f1 | 0.6974 |
| improved_f1 | 0.7943 |
| f1_improvement | 0.09692 |
| baseline_f1_std | 1.11e-16 |
| improved_f1_std | 0.0007365 |
| baseline_brier | 0.2038 |
| improved_brier | 0.1355 |
| brier_improvement | 0.06836 |
| baseline_brier_std | 0 |
| improved_brier_std | 0.0009466 |
| elapsed_seconds | 164.7 |

Run on 2026-10-19. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in dataset chosen by name from OpenML, not the paper's data; the paper's vision-language scorer is not reproduced. Positive class is the minority label. Rows subsampled to at most 200,000 stratified. Only LightGBM is randomised; the logistic-regression baseline is deterministic, so its std is zero. Test set used once per seed for both models; no test-set selection.

## Reference

This project is motivated by the following paper:

> AB Constantin, V Negru, C Lemnaru (2025). *VISION-RISK: Vision-Language Model for Risk Assessment in Autonomous Driving*. 2025 IEEE 21st. https://ieeexplore.ieee.org/abstract/document/11427156/

Paper author: Prof. Camelia Lemnaru, Technical University of Cluj-Napoca (camelia.lemnaru@cs.utcluj.ro)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
