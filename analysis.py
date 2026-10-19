"""Baseline (standardised logistic regression) vs improved (class-weighted,
isotonically calibrated LightGBM) on a public binary road-accident table from OpenML."""
import json, os, socket, sys, time
import numpy as np, pandas as pd
import openml, lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SEEDS = [42, 43, 44]          # seeds for the randomised (LightGBM) model; baseline run with the same seeds
MAX_ROWS = 200_000            # subsample budget for the 20-minute limit
TEST_SIZE = 0.25
CANDIDATES = ["road-safety", "road-accidents", "road_accidents", "traffic-accidents", "accidents"]
socket.setdefaulttimeout(60)  # 60 s timeout on all network I/O used by openml
HERE = os.path.dirname(os.path.abspath(__file__))


def with_retries(fn, tries=5):
    """Retry on network errors, HTTP 429 and HTTP 5xx with exponential backoff."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            retriable = code is None or code == 429 or code >= 500
            if i == tries - 1 or not retriable:
                raise
            print(f"  download error ({e}); retrying in {2 ** i}s")
            time.sleep(2 ** i)


def load_dataset():
    """Find the first candidate binary dataset on OpenML; positive class = minority (high-risk) class."""
    for name in CANDIDATES:
        lst = with_retries(lambda: openml.datasets.list_datasets(data_name=name, output_format="dataframe"))
        if len(lst) == 0:
            continue
        ok = lst[(lst.NumberOfClasses == 2) & (lst.NumberOfInstances >= 500)]
        if len(ok) == 0:
            continue
        did = int(ok.sort_values("NumberOfInstances", ascending=False).did.iloc[0])
        ds = with_retries(lambda: openml.datasets.get_dataset(did))
        if not ds.default_target_attribute:
            continue
        X, y, _, _ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)
        y = pd.Series(np.asarray(y).astype(str))
        minority = y.value_counts().idxmin()
        X = pd.get_dummies(X).astype(float)          # one-hot categoricals, shared by both models
        print(f"Loaded OpenML dataset '{ds.name}' (id {did}): {X.shape[0]} rows, {X.shape[1]} features")
        return X, (y == minority).astype(int).values, ds.name, did
    sys.exit("No suitable binary OpenML dataset found among candidates; cannot produce metrics.")


def pick_threshold(oof_proba, y):
    """Threshold maximising macro-F1 on out-of-fold training predictions (never the test set)."""
    grid = np.linspace(0.02, 0.98, 97)
    scores = [f1_score(y, oof_proba >= t, average="macro") for t in grid]
    return float(grid[int(np.argmax(scores))])


def fit_eval(model, Xtr, ytr, Xte, yte, seed):
    """Shared evaluation: CV threshold on train, then one test evaluation."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    oof = cross_val_predict(model, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
    thr = pick_threshold(oof, ytr)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    return {"pr_auc": average_precision_score(yte, p), "brier": brier_score_loss(yte, p),
            "f1": f1_score(yte, p >= thr, average="macro"), "threshold": thr}


def tune_lgbm(Xtr, ytr):
    """Small randomised search with 3-fold stratified CV on the training split only."""
    base = lgb.LGBMClassifier(class_weight="balanced", n_estimators=300, random_state=SEED, verbose=-1)
    space = {"num_leaves": randint(15, 64), "learning_rate": uniform(0.03, 0.12),
             "min_child_samples": randint(20, 100), "colsample_bytree": uniform(0.6, 0.4)}
    search = RandomizedSearchCV(base, space, n_iter=8, scoring="average_precision",
                                cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                                random_state=SEED, n_jobs=1)
    search.fit(Xtr, ytr)
    print(f"  best CV PR-AUC {search.best_score_:.4f} with {search.best_params_}")
    return search.best_estimator_


def main():
    t0 = time.time()
    X, y, name, did = load_dataset()
    if len(X) > MAX_ROWS:  # stratified subsample for runtime
        X, _, y, _ = train_test_split(X, y, train_size=MAX_ROWS, stratify=y, random_state=SEED)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    imp = SimpleImputer(strategy="median").fit(Xtr)  # preprocessing fitted on train only
    Xtr, Xte = imp.transform(Xtr), imp.transform(Xte)
    print(f"Split: {len(ytr)} train / {len(yte)} test, positive rate {ytr.mean():.3f}")

    print("Tuning LightGBM on training CV only...")
    best_lgb = tune_lgbm(Xtr, ytr)

    base_runs, imp_runs = [], []
    for seed in SEEDS:
        print(f"Seed {seed}: baseline logistic regression")
        base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        base_runs.append(fit_eval(base, Xtr, ytr, Xte, yte, seed))
        print(f"Seed {seed}: improved calibrated LightGBM")
        improved = CalibratedClassifierCV(clone(best_lgb).set_params(random_state=seed),
                                          method="isotonic", cv=3)
        imp_runs.append(fit_eval(improved, Xtr, ytr, Xte, yte, seed))

    def agg(runs, key):
        v = np.array([r[key] for r in runs])
        return float(v.mean()), float(v.std())

    b_pr, b_pr_sd = agg(base_runs, "pr_auc")
    i_pr, i_pr_sd = agg(imp_runs, "pr_auc")
    b_f1, b_f1_sd = agg(base_runs, "f1")
    i_f1, i_f1_sd = agg(imp_runs, "f1")
    b_br, b_br_sd = agg(base_runs, "brier")
    i_br, i_br_sd = agg(imp_runs, "brier")
    if not np.isfinite([b_pr, i_pr, b_f1, i_f1, b_br, i_br]).all():
        sys.exit("Non-finite metrics; failing.")

    results = {
        "question": "Can a class-balanced, calibrated LightGBM model give more reliable risk probabilities "
                    "than a standard logistic-regression baseline on a public tabular driving-hazard stand-in dataset?",
        "method": "Stratified 75/25 train/test split (seed 42). Baseline: standardised logistic regression. "
                  "Improved: class-weighted LightGBM with CV-tuned hyperparameters plus isotonic calibration. "
                  "Thresholds for macro-F1 chosen by out-of-fold CV on the training split. Seeds 42-44.",
        "dataset": {"name": name, "source_url": f"https://www.openml.org/d/{did}",
                    "n_rows": int(len(y)), "n_train": int(len(ytr)), "n_test": int(len(yte)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "PR-AUC (average precision) on held-out test set",
        "higher_is_better": True,
        "baseline": "Standardised logistic regression (scikit-learn, default L2, no class weighting)",
        "improved": "Class-weighted LightGBM with 8-draw CV hyperparameter search and isotonic calibration (3-fold)",
        "metrics": {
            "baseline_score": b_pr, "improved_score": i_pr, "improvement": i_pr - b_pr,
            "baseline_score_std": b_pr_sd, "improved_score_std": i_pr_sd,
            "baseline_f1": b_f1, "improved_f1": i_f1, "f1_improvement": i_f1 - b_f1,
            "baseline_f1_std": b_f1_sd, "improved_f1_std": i_f1_sd,
            "baseline_brier": b_br, "improved_brier": i_br,
            "brier_improvement": b_br - i_br,  # positive = improved has lower Brier score
            "baseline_brier_std": b_br_sd, "improved_brier_std": i_br_sd,
            "elapsed_seconds": time.time() - t0,
        },
        "notes": "Stand-in dataset chosen by name from OpenML, not the paper's data; the paper's vision-language "
                 "scorer is not reproduced. Positive class is the minority label. Rows subsampled to at most "
                 "200,000 stratified. Only LightGBM is randomised; the logistic-regression baseline is deterministic, "
                 "so its std is zero. Test set used once per seed for both models; no test-set selection.",
    }

    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["baseline LR", "improved LGBM"], [b_pr, i_pr], yerr=[b_pr_sd, i_pr_sd], capsize=5,
           color=["grey", "steelblue"])
    ax.set_ylabel("Test PR-AUC (higher is better)")
    ax.set_title("Baseline vs improved (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "pr_auc_bar.png"), dpi=120)

    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Baseline PR-AUC {b_pr:.4f} | Improved PR-AUC {i_pr:.4f} | results.json written")


if __name__ == "__main__":
    main()
