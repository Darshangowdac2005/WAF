"""
ml/layer2a/candidates/isolation_forest.py

Layer 2A Candidate 1 — Isolation Forest
----------------------------------------
One-class anomaly detector using the ensemble isolation approach.
Trained ONLY on normal traffic feature vectors (25 numeric features).
No attack labels required during training.

How it works
------------
Each tree isolates observations by randomly selecting a feature
and a random split value. Anomalies — being rare and structurally
different — are isolated in fewer steps (shorter path length).
The anomaly score is derived from the average path length across trees.

Strengths : fastest training, no GPU, scales to large datasets,
            robust to irrelevant features, easy ONNX export.
Weakness  : linear-ish decision boundaries, may miss subtle
            anomalies that look "normal" in aggregate statistics.

Output
------
decision_function() → float (higher = more normal, lower = anomalous)
predict()           → -1 (anomaly) or +1 (normal)
Anomaly score used by threat_scorer = -decision_function() normalised to 0-50
"""

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)


# ── Hyperparameters ────────────────────────────────────────────────────────────

PARAMS = {
    "n_estimators":  200,
    "contamination": 0.05,   # assumed anomaly fraction in training set
                             # keep low: training should be clean normal traffic
    "max_features":  1.0,    # fraction of features per tree
    "max_samples":   "auto", # min(256, n_samples)
    "bootstrap":     False,
    "random_state":  42,
    "n_jobs":        -1,
}


# ── Build ──────────────────────────────────────────────────────────────────────

def build() -> Pipeline:
    """
    StandardScaler → IsolationForest pipeline.
    Scaling ensures no single feature dominates the split selection.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  IsolationForest(**PARAMS)),
    ])


# ── Train ──────────────────────────────────────────────────────────────────────

def train(
    X_train: np.ndarray,
    run_name: str = "iforest",
) -> Pipeline:
    """
    Train IsolationForest on normal-only feature vectors.

    Parameters
    ----------
    X_train : (N, 25) float32 — normal traffic ONLY, no attacks
    run_name : MLflow run label

    Returns
    -------
    Fitted sklearn Pipeline
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(PARAMS)
        mlflow.log_param("n_train_samples", len(X_train))

        pipe = build()
        pipe.fit(X_train)

        mlflow.sklearn.log_model(pipe, "isolation_forest")
        print(f"[IForest] Trained on {len(X_train)} normal samples.")

    return pipe


# ── Evaluate ───────────────────────────────────────────────────────────────────

def evaluate(
    pipe: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = None,
) -> dict:
    """
    Evaluate on a MIXED test set.

    Parameters
    ----------
    pipe      : fitted Pipeline from train()
    X_test    : (N, 25) — mix of normal + attack samples
    y_test    : (N,) int — 0=normal, 1=anomaly/attack
    threshold : float or None
        Anomaly score cutoff. None = use IForest's built-in contamination
        threshold. Pass a float (from find_threshold()) to tune FPR/TPR.

    Returns
    -------
    dict with keys: model, auc, avg_precision, fpr, tpr, tp, fp, tn, fn
    """
    # decision_function: higher = more normal
    scores        = pipe.decision_function(X_test)
    anomaly_scores = -scores     # flip: higher = more anomalous

    auc = roc_auc_score(y_test, anomaly_scores)
    ap  = average_precision_score(y_test, anomaly_scores)

    if threshold is None:
        preds = (pipe.predict(X_test) == -1).astype(int)
        thr_used = "contamination_default"
    else:
        preds    = (anomaly_scores > threshold).astype(int)
        thr_used = round(float(threshold), 6)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    results = {
        "model":         "isolation_forest",
        "auc":           round(float(auc), 4),
        "avg_precision": round(float(ap),  4),
        "fpr":           round(fpr, 4),
        "tpr":           round(tpr, 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "threshold_used": thr_used,
    }

    print("\n[IForest] Evaluation:")
    for k, v in results.items():
        print(f"  {k:<22}: {v}")

    return results


# ── Threshold tuning ───────────────────────────────────────────────────────────

def find_threshold(
    pipe: Pipeline,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_fpr: float = 0.05,
) -> float:
    """
    Find the anomaly score threshold that achieves target_fpr
    while maximising TPR on the VALIDATION set.

    Never call this on the test set.
    """
    scores = -pipe.decision_function(X_val)

    best_thr = None
    best_tpr = 0.0

    for thr in np.linspace(float(scores.min()), float(scores.max()), 300):
        preds = (scores > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if fpr <= target_fpr and tpr > best_tpr:
            best_tpr = tpr
            best_thr = float(thr)

    if best_thr is None:
        # no threshold meets target — pick the one with lowest FPR
        best_thr = float(scores.max())

    print(f"[IForest] Best threshold for FPR≤{target_fpr}: "
          f"{best_thr:.5f}  →  TPR={best_tpr:.4f}")
    return best_thr


# ── ONNX export ────────────────────────────────────────────────────────────────

def export_onnx(pipe: Pipeline, output_path: str, input_dim: int = 25):
    """
    Export fitted sklearn Pipeline to ONNX.
    Requires: pip install skl2onnx
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as ort

    initial_types = [("features", FloatTensorType([None, input_dim]))]
    proto = convert_sklearn(
        pipe,
        name="isolation_forest_l2a",
        initial_types=initial_types,
        target_opset=17,
    )
    with open(output_path, "wb") as f:
        f.write(proto.SerializeToString())

    # validate
    sess  = ort.InferenceSession(output_path)
    dummy = np.random.randn(1, input_dim).astype(np.float32)
    out   = sess.run(None, {"features": dummy})
    print(f"[IForest] ONNX exported → {output_path}")
    print(f"[IForest] ONNX validation OK. Output: {[o.shape for o in out]}")