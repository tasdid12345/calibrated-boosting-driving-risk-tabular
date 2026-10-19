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

## Reference

This project is motivated by the following paper:

> AB Constantin, V Negru, C Lemnaru (2025). *VISION-RISK: Vision-Language Model for Risk Assessment in Autonomous Driving*. 2025 IEEE 21st. https://ieeexplore.ieee.org/abstract/document/11427156/

Paper author: Prof. Camelia Lemnaru, Technical University of Cluj-Napoca (camelia.lemnaru@cs.utcluj.ro)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
