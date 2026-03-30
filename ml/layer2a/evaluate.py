"""
ml/layer2a/evaluate.py

Evaluation helpers for Layer 2A one-class models.
Shared by train.py and notebook 03.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    roc_curve,
)


def evaluate_candidate(model, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    """
    Evaluate a Layer 2A model on a mixed test set.

    Parameters
    ----------
    model  : object with .anomaly_scores(X) -> np.ndarray and .predict(X) -> np.ndarray
    X_test : (N, 20) float32 — already normalised
    y_test : (N,)   int     — 0=normal, 1=attack
    name   : str label for the result dict

    Returns
    -------
    dict with keys: model, auc, avg_precision, fpr, tpr, tp, fp, tn, fn
    """
    scores = model.anomaly_scores(X_test)   # higher = more anomalous
    preds  = model.predict(X_test)          # 1=anomaly, 0=normal

    auc = roc_auc_score(y_test, scores)
    ap  = average_precision_score(y_test, scores)

    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    result = {
        "model":         name,
        "auc":           round(float(auc), 4),
        "avg_precision": round(float(ap),  4),
        "fpr":           round(float(fpr), 4),
        "tpr":           round(float(tpr), 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }

    print(f"\n[evaluate] {name}")
    for k, v in result.items():
        if k != "model":
            print(f"  {k:<22}: {v}")

    return result


def pick_best(results: list, models: dict, max_fpr: float = 0.05) -> tuple:
    """
    Select the best L2A model.
    Primary:   FPR <= max_fpr
    Secondary: highest TPR
    Tiebreak:  highest AUC

    Returns (winner_name, winner_model_object)
    """
    qualifying = [r for r in results if r["fpr"] <= max_fpr]
    if not qualifying:
        print(f"[pick_best] No model under FPR={max_fpr}. Taking lowest FPR.")
        qualifying = sorted(results, key=lambda r: r["fpr"])[:1]

    best = sorted(qualifying, key=lambda r: (-r["tpr"], -r["auc"]))[0]
    name = best["model"]

    print(f"\n[pick_best] Winner: {name}")
    print(f"  FPR={best['fpr']}  TPR={best['tpr']}  AUC={best['auc']}")

    return name, models[name]


def threshold_sweep(model, X_val: np.ndarray, y_val: np.ndarray,
                    target_fpr: float = 0.05, n_steps: int = 200) -> float:
    """
    Find the score threshold that achieves target_fpr with max TPR on val set.
    Use on validation set only — never on test set.
    """
    scores = model.anomaly_scores(X_val)
    best_thr, best_tpr = None, 0.0

    for thr in np.linspace(scores.min(), scores.max(), n_steps):
        preds = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if fpr <= target_fpr and tpr > best_tpr:
            best_tpr, best_thr = tpr, float(thr)

    print(f"[threshold_sweep] Best thr={best_thr:.5f} → FPR≤{target_fpr}, TPR={best_tpr:.4f}")
    return best_thr