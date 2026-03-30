"""
ml/evaluation/compare_models.py

Builds comparison tables for both layers.
Run from notebook 05 after all candidates are trained.

Usage:
    from evaluation.compare_models import compare_l2a, compare_l2b
    df_l2a = compare_l2a(results_list)
    df_l2b = compare_l2b(results_list)
"""

import pandas as pd
import numpy as np
import json


# ── Layer 2A ──────────────────────────────────────────────────────────────────

def compare_l2a(results: list) -> pd.DataFrame:
    """
    Build L2A comparison table from a list of result dicts.
    Each dict: { model, auc, avg_precision, fpr, tpr, tp, fp, tn, fn }

    Sorted by FPR ascending (lower is better primary metric).
    """
    rows = [{
        "Model":         r["model"],
        "AUC":           r["auc"],
        "Avg Precision": r.get("avg_precision", "-"),
        "TPR (recall)":  r["tpr"],
        "FPR":           r["fpr"],
        "TP": r.get("tp", "-"),
        "FP": r.get("fp", "-"),
        "TN": r.get("tn", "-"),
        "FN": r.get("fn", "-"),
    } for r in results]

    df = pd.DataFrame(rows).sort_values("FPR")
    print("\n=== Layer 2A Model Comparison ===")
    print(df.to_string(index=False))
    return df


def pick_best_l2a(results: list, models: dict, max_fpr: float = 0.05) -> tuple:
    """
    Select best L2A model: lowest FPR <= max_fpr, then highest TPR, then AUC.
    Returns (winner_name, winner_model).
    """
    qualifying = [r for r in results if r["fpr"] <= max_fpr]
    if not qualifying:
        print(f"[compare] No model meets FPR<={max_fpr}. Selecting lowest FPR.")
        qualifying = sorted(results, key=lambda r: r["fpr"])[:1]

    best = sorted(qualifying, key=lambda r: (-r["tpr"], -r["auc"]))[0]
    name = best["model"]
    print(f"\n[compare] L2A Winner: {name}  "
          f"FPR={best['fpr']}  TPR={best['tpr']}  AUC={best['auc']}")
    return name, models[name]


# ── Layer 2B ──────────────────────────────────────────────────────────────────

def compare_l2b(results: list) -> pd.DataFrame:
    """
    Build L2B comparison table.
    Each dict: { model, macro_f1, accuracy, per_class_f1 }

    Sorted by macro_f1 descending.
    """
    rows = []
    for r in results:
        row = {
            "Model":    r["model"],
            "Macro F1": r["macro_f1"],
            "Accuracy": r["accuracy"],
        }
        for cls, f1 in r.get("per_class_f1", {}).items():
            row[f"F1 ({cls})"] = f1
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
    print("\n=== Layer 2B Model Comparison ===")
    print(df.to_string(index=False))
    return df


def pick_best_l2b(results: list, models: dict,
                  min_attack_f1: float = 0.90) -> tuple:
    """
    Select best L2B model.
    Constraint: per-class F1 >= min_attack_f1 for all attack classes.
    Primary:    highest macro F1.
    Returns (winner_name, winner_model).
    """
    attack_cls = ["sqli", "xss", "lfi", "other_attack"]

    def all_ok(r):
        return all(r["per_class_f1"].get(c, 0) >= min_attack_f1
                   for c in attack_cls)

    qualifying = [r for r in results if all_ok(r)]
    if not qualifying:
        print(f"[compare] No model meets per-class F1>={min_attack_f1}. "
              "Selecting highest macro F1.")
        qualifying = results

    best = sorted(qualifying, key=lambda r: -r["macro_f1"])[0]
    name = best["model"]
    print(f"\n[compare] L2B Winner: {name}  "
          f"Macro F1={best['macro_f1']}  Accuracy={best['accuracy']}")
    return name, models[name]


# ── Combined summary ──────────────────────────────────────────────────────────

def save_full_report(l2a_results: list, l2b_results: list,
                     latency: list, output_path: str) -> None:
    """
    Save a single JSON report with all comparison data.
    Include in docs/model_selection.md.
    """
    report = {
        "layer2a": l2a_results,
        "layer2b": l2b_results,
        "latency": latency,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[compare] Full report saved → {output_path}")