"""
ml/layer2a/evaluate.py

Shared evaluation utilities for Layer 2A candidates.
Called by layer2a/train.py and notebook 03.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve


# ── Comparison table ──────────────────────────────────────────────────────────

def compare_candidates(results: list[dict]) -> pd.DataFrame:
    """
    Build a side-by-side comparison DataFrame from evaluate() dicts.

    Sorted by FPR ascending (lower FPR is primary criterion).
    """
    rows = []
    for r in results:
        rows.append({
            "Model":          r["model"],
            "AUC":            r.get("auc", "-"),
            "Avg Precision":  r.get("avg_precision", "-"),
            "TPR (recall)":   r.get("tpr", "-"),
            "FPR":            r.get("fpr", "-"),
            "TP":             r.get("tp", "-"),
            "FP":             r.get("fp", "-"),
            "TN":             r.get("tn", "-"),
            "FN":             r.get("fn", "-"),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("FPR")
    return df


def select_best(results: list[dict], max_fpr: float = 0.05) -> dict:
    """
    Select winner from L2A candidates.

    Primary   : FPR <= max_fpr
    Secondary : highest TPR
    Tiebreak  : highest AUC
    """
    qualifying = [r for r in results if r.get("fpr", 1.0) <= max_fpr]
    if not qualifying:
        print(f"[L2A] No model meets FPR ≤ {max_fpr}. Selecting lowest FPR.")
        qualifying = sorted(results, key=lambda r: r.get("fpr", 1.0))

    winner = sorted(qualifying, key=lambda r: (-r.get("tpr", 0), -r.get("auc", 0)))[0]
    print(f"[L2A] Winner: {winner['model']}  "
          f"FPR={winner.get('fpr')}  TPR={winner.get('tpr')}  AUC={winner.get('auc')}")
    return winner


# ── ROC / PR plots ────────────────────────────────────────────────────────────

def plot_roc(scores_dict: dict, y_true: np.ndarray, save_path: str = None):
    """
    Plot ROC curves for multiple models.

    Parameters
    ----------
    scores_dict : {model_name: anomaly_score_array}
    y_true      : ground truth (0=normal, 1=attack)
    save_path   : path to save PNG (optional)
    """
    from sklearn.metrics import roc_auc_score

    plt.figure(figsize=(7, 5))
    for name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.axvline(x=0.05, color="red", linestyle=":", linewidth=1,
                label="FPR=0.05 target")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Layer 2A — ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[L2A] ROC plot saved → {save_path}")
    plt.show()


def plot_pr(scores_dict: dict, y_true: np.ndarray, save_path: str = None):
    """Plot Precision-Recall curves for multiple models."""
    from sklearn.metrics import average_precision_score

    plt.figure(figsize=(7, 5))
    for name, scores in scores_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})", linewidth=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Layer 2A — Precision-Recall Curves")
    plt.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[L2A] PR plot saved → {save_path}")
    plt.show()


def plot_reconstruction_errors(normal_errors: np.ndarray,
                               attack_errors: np.ndarray,
                               threshold: float,
                               save_path: str = None):
    """
    Histogram of reconstruction errors for normal vs attack samples.
    Shows how well the autoencoder separates the two distributions.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(normal_errors, bins=60, alpha=0.6, color="steelblue",
             label="Normal", density=True)
    plt.hist(attack_errors, bins=60, alpha=0.6, color="tomato",
             label="Attack", density=True)
    plt.axvline(x=threshold, color="black", linestyle="--", linewidth=1.5,
                label=f"Threshold={threshold:.4f}")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Density")
    plt.title("Layer 2A Autoencoder — Reconstruction Error Distribution")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()