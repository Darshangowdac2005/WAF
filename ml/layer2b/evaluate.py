"""
ml/layer2b/evaluate.py

Evaluation helpers for Layer 2B multi-class classifiers.
Shared by train.py and notebook 04.
"""

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

CLASS_NAMES = ["normal", "sqli", "xss", "lfi", "other_attack"]


def evaluate_candidate(model, X_test, y_test: np.ndarray, name: str) -> dict:
    """
    Evaluate a Layer 2B model on a labeled test set.

    Parameters
    ----------
    model  : object with .predict(X) -> np.ndarray of class indices
    X_test : feature matrix or token array depending on model type
    y_test : (N,) int — class labels 0-4
    name   : str label

    Returns
    -------
    dict with model, macro_f1, accuracy, per_class_f1, confusion_matrix
    """
    preds    = model.predict(X_test)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    accuracy = accuracy_score(y_test, preds)
    cm       = confusion_matrix(y_test, preds, labels=list(range(5))).tolist()

    report = classification_report(
        y_test, preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    per_class = {cls: round(report[cls]["f1-score"], 4) for cls in CLASS_NAMES}

    result = {
        "model":            name,
        "macro_f1":         round(float(macro_f1), 4),
        "accuracy":         round(float(accuracy), 4),
        "per_class_f1":     per_class,
        "confusion_matrix": cm,
    }

    print(f"\n[evaluate] {name}")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES, zero_division=0))

    return result


CLASS_NAMES = ["normal", "sqli", "xss", "lfi", "other_attack", "cmdi"]


def pick_best(results: list, models: dict,
              min_attack_f1: float = 0.90,
              max_p99_ms: float = 20.0) -> tuple:
    """
    Select best L2B model with BOTH accuracy AND latency constraints.

    Primary:   highest macro F1
    Constraint 1: all attack-class F1 >= min_attack_f1
    Constraint 2: ONNX P99 inference latency <= max_p99_ms
                  (disqualifies GRU if it exceeds the SLA even at higher F1)

    Returns (winner_name, winner_model_object)
    """
    attack_classes = ["sqli", "xss", "lfi", "other_attack", "cmdi"]

    def f1_ok(r):
        return all(r["per_class_f1"].get(c, 0) >= min_attack_f1 for c in attack_classes
                   if c in r["per_class_f1"])  # skip classes not yet in training set

    def latency_ok(r):
        # p99_ms is populated by export_onnx() benchmark — default 0 means not measured
        return r.get("p99_ms", 0) <= max_p99_ms or r.get("p99_ms", 0) == 0

    qualifying = [r for r in results if f1_ok(r) and latency_ok(r)]

    if not qualifying:
        print(f"[pick_best] No model meets F1>={min_attack_f1} AND p99<={max_p99_ms}ms. "
              f"Relaxing latency constraint.")
        qualifying = [r for r in results if f1_ok(r)]

    if not qualifying:
        print(f"[pick_best] No model meets per-class F1>={min_attack_f1}. "
              f"Selecting highest macro F1.")
        qualifying = results

    best = sorted(qualifying, key=lambda r: -r["macro_f1"])[0]
    name = best["model"]

    print(f"\n[pick_best] Winner: {name}")
    print(f"  Macro F1={best['macro_f1']}  Accuracy={best['accuracy']}"
          f"  P99={best.get('p99_ms', 'N/A')}ms")
    for cls in CLASS_NAMES:
        if cls in best["per_class_f1"]:
            print(f"  F1[{cls}]={best['per_class_f1'][cls]}")

    return name, models[name]