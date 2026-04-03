"""
ml/layer2a/evaluate.py

Evaluation helpers for Layer 2A one-class models.
Shared by train.py and notebook 03.
"""

import numpy as np
import pandas as pd
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
    X_test : (N, 25) float32 — already normalised
    y_test : (N,)   int     — 0=normal, 1=attack
    name   : str label for the result dict

    Returns
    -------
    dict with keys: model, auc, avg_precision, fpr, tpr, tp, fp, tn, fn
    """
    scores = model.anomaly_scores(X_test)
    preds  = model.predict(X_test)

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


def compare_l2a(results: list) -> "pd.DataFrame":
    """
    Build a comparison DataFrame from a list of evaluate_candidate result dicts.
    Sorted by AUC descending.

    Parameters
    ----------
    results : list of dicts returned by evaluate_candidate()

    Returns
    -------
    pd.DataFrame with columns: Model, AUC, Avg Precision, TPR (recall), FPR, TP, FP, TN, FN
    """
    rows = []
    for r in results:
        rows.append({
            "Model":         r["model"],
            "AUC":           r["auc"],
            "Avg Precision": r["avg_precision"],
            "TPR (recall)":  r["tpr"],
            "FPR":           r["fpr"],
            "TP":            r["tp"],
            "FP":            r["fp"],
            "TN":            r["tn"],
            "FN":            r["fn"],
        })
    df = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
    return df


def pick_best_l2a(results: list, models: dict, target_fpr: float = 0.05) -> tuple:
    print(">>> USING NEW pick_best_l2a() FROM evaluate.py <<<")
    """
    Selects the OVERALL best model.
    
    Priority:
    1. Highest AUC (Overall separation power)
    2. Lowest FPR (Production stability/Silence)
    3. Highest TPR (Detection rate)
    """
    # Sort by AUC first (Primary quality metric)
    # Then by FPR (Secondary: lower is better for WAF noise)
    # Then by TPR (Tertiary: more detections)
    sorted_results = sorted(
        results, 
        key=lambda r: (-r["auc"], r["fpr"], -r["tpr"])
    )

    best = sorted_results[0]
    name = best["model"]

    # Special check: If the top AUC model has a catastrophic FPR (> 20%), 
    # we fallback to the safest model under target_fpr.
    if best["fpr"] > 0.20:
        print(f"[compare] Top model {name} has extreme FPR ({best['fpr']}). Falling back to safe models...")
        safe_models = [r for r in results if r["fpr"] <= target_fpr]
        if safe_models:
            best = sorted(safe_models, key=lambda r: (-r["auc"], -r["tpr"]))[0]
            name = best["model"]

    print(f"\n[compare] OVERALL WINNER: {name}")
    print(f"  Final Decision Metrics -> AUC: {best['auc']} | FPR: {best['fpr']} | TPR: {best['tpr']}")

    return name, models[name]
# ── Aliases kept for backward compatibility with train.py ─────────────────────
pick_best = pick_best_l2a


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