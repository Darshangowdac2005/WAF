"""
ml/layer2a/candidates/isolation_forest.py

Layer 2A Candidate 1 — Isolation Forest
----------------------------------------
One-class anomaly detector using the ensemble isolation approach.
Trained ONLY on normal traffic feature vectors (20 numeric features).
No attack labels required during training.

How it works
------------
Each tree isolates observations by randomly selecting a feature
and a random split value. Anomalies — being rare and structurally
different — are isolated in fewer steps (shorter average path length).
The anomaly score is derived from the average path length across trees.

Strengths : fastest training (~seconds), no GPU needed, scales to
            large datasets, robust to irrelevant features, easy ONNX
            export via skl2onnx.
Weakness  : linear-ish decision boundaries, may miss subtle anomalies
            that look "normal" in aggregate numeric statistics.
            Use ShallowAutoencoder as the second candidate to cover
            this gap.

Interface (same as autoencoder_shallow.py)
------------------------------------------
    model = IsolationForestModel()
    model.train(X_normal)
    model.tune_threshold(X_val)
    result = evaluate_candidate(model, X_test, y_test, name="isolation_forest")
    model.export_onnx("exported_models/layer2a_best.onnx")

Anomaly score used by threat_scorer.py
---------------------------------------
    score = model.anomaly_scores(x)   # higher = more anomalous
    # mapped to 0-50 range in threat_scorer:
    #   l2a_contrib = min(50, score * 15)
"""

import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from pathlib import Path
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
# Tune contamination first if FPR is too high.
# Increasing n_estimators improves stability at the cost of training time.

PARAMS = {
    "n_estimators":  200,
    "contamination": 0.05,   # assumed anomaly fraction in training set
                             # keep low: training should be clean normal traffic
    "max_features":  1.0,    # fraction of features used per tree
    "max_samples":   "auto", # min(256, n_samples) per tree
    "bootstrap":     False,
    "random_state":  42,
    "n_jobs":        -1,     # use all CPU cores
}

INPUT_DIM = 20               # must match len(FEATURE_NAMES) in extractor.py


# ── Model class ────────────────────────────────────────────────────────────────

class IsolationForestModel:
    """
    Wrapper around sklearn IsolationForest that exposes the standard
    Layer 2A interface expected by layer2a/evaluate.py and train.py:

        .train(X_normal)
        .tune_threshold(X_val, y_val=None, target_fpr=0.05)
        .anomaly_scores(X)   -> np.ndarray  (higher = more anomalous)
        .predict(X)          -> np.ndarray  (1=anomaly, 0=normal)
        .predict_single(x)   -> (is_anomaly: bool, score: float)
        .export_onnx(path)
        .save(path) / .load(path)
    """

    def __init__(self):
        self.pipeline  = self._build()
        self.threshold = None   # set by tune_threshold()
        self._fitted   = False

    # ── Build pipeline ─────────────────────────────────────────────────────────

    @staticmethod
    def _build() -> Pipeline:
        """
        StandardScaler → IsolationForest pipeline.
        Scaling ensures no single feature dominates the split selection
        even though IsolationForest is scale-invariant in theory — in
        practice, scaled features produce more consistent path lengths.
        """
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  IsolationForest(**PARAMS)),
        ])

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self,
        X_normal: np.ndarray,
        run_name: str = "iforest",
    ) -> None:
        """
        Fit on normal-only feature vectors.
        X_normal should NOT contain any attack samples.

        Parameters
        ----------
        X_normal : (N, 20) float32 — normal traffic feature vectors only.
                   Do NOT normalise before passing — the pipeline's
                   StandardScaler handles it internally.
        run_name : MLflow run label for experiment tracking.
        """
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_params(PARAMS)
            mlflow.log_param("input_dim",       INPUT_DIM)
            mlflow.log_param("n_train_samples", len(X_normal))

            self.pipeline.fit(X_normal.astype(np.float32))
            self._fitted = True

            mlflow.sklearn.log_model(self.pipeline, "isolation_forest")

        print(f"[IForest] Trained on {len(X_normal)} normal samples.")

    # ── Threshold tuning ───────────────────────────────────────────────────────

    def tune_threshold(
        self,
        X_val:      np.ndarray,
        y_val:      np.ndarray = None,
        target_fpr: float = 0.05,
    ) -> float:
        """
        Set the anomaly score threshold.

        Two modes:
        - Supervised (y_val provided): sweep thresholds to find the one
          that achieves target_fpr with maximum TPR on the validation set.
          Use this when you have a small labelled validation set.

        - Unsupervised (y_val=None): set threshold at the 95th percentile
          of anomaly scores on the validation set. Use this for the
          purely one-class case where no labels are available.

        Never call with the test set — always use validation data.

        Parameters
        ----------
        X_val      : (N, 20) float32
        y_val      : (N,) int — 0=normal, 1=attack  OR  None
        target_fpr : target false positive rate (supervised mode only)

        Returns
        -------
        float — the chosen threshold (also stored as self.threshold)
        """
        scores = self.anomaly_scores(X_val)

        if y_val is None:
            # unsupervised: 95th percentile of normal val set scores
            # (assuming X_val is mostly normal)
            self.threshold = float(np.percentile(scores, 95))
            print(f"[IForest] Threshold (95th pct, unsupervised): "
                  f"{self.threshold:.5f}")
        else:
            self.threshold = self._find_threshold(scores, y_val, target_fpr)

        return self.threshold

    def _find_threshold(
        self,
        anomaly_scores: np.ndarray,
        y_val:          np.ndarray,
        target_fpr:     float,
        n_steps:        int = 300,
    ) -> float:
        """
        Grid-search the anomaly score threshold.
        Maximises TPR subject to FPR <= target_fpr.
        Falls back to the score that minimises FPR if none qualify.

        Parameters
        ----------
        anomaly_scores : (N,) float — output of self.anomaly_scores()
        y_val          : (N,) int   — 0=normal, 1=attack
        target_fpr     : float      — e.g. 0.05
        n_steps        : int        — number of threshold candidates

        Returns
        -------
        float — best threshold
        """
        best_thr = None
        best_tpr = 0.0

        lo = float(anomaly_scores.min())
        hi = float(anomaly_scores.max())

        for thr in np.linspace(lo, hi, n_steps):
            preds = (anomaly_scores > thr).astype(int)

            # guard against all-one or all-zero predictions
            if len(np.unique(preds)) < 2:
                continue

            tn, fp, fn, tp = confusion_matrix(
                y_val, preds, labels=[0, 1]
            ).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if fpr <= target_fpr and tpr > best_tpr:
                best_tpr = tpr
                best_thr = float(thr)

        if best_thr is None:
            # no threshold meets FPR target — pick max anomaly score
            # (aggressive: blocks everything; human should lower threshold)
            best_thr = hi
            print(f"[IForest] WARNING: no threshold achieves FPR≤{target_fpr}. "
                  f"Using max score ({best_thr:.5f}). "
                  f"Consider raising contamination parameter.")
        else:
            print(f"[IForest] Best threshold for FPR≤{target_fpr}: "
                  f"{best_thr:.5f}  →  TPR={best_tpr:.4f}")

        return best_thr

    # ── Inference ──────────────────────────────────────────────────────────────

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for a batch of samples.

        IsolationForest.decision_function() returns higher values for
        normal samples and lower values for anomalies. We negate it so
        that higher score = more anomalous, which is the convention used
        by threat_scorer.py and evaluate.py.

        Parameters
        ----------
        X : (N, 20) float32 — raw (un-normalised) feature vectors.
            The pipeline's StandardScaler handles normalisation internally.

        Returns
        -------
        (N,) float32 — anomaly scores, higher = more anomalous
        """
        return -self.pipeline.decision_function(X.astype(np.float32))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Binary predictions: 1 = anomaly, 0 = normal.

        Uses self.threshold if set by tune_threshold(), otherwise falls
        back to IsolationForest's built-in contamination-based boundary.

        Parameters
        ----------
        X : (N, 20) float32

        Returns
        -------
        (N,) int — 1=anomaly, 0=normal
        """
        if self.threshold is not None:
            return (self.anomaly_scores(X) >= self.threshold).astype(int)
        # built-in boundary: predict() returns -1 (anomaly) or +1 (normal)
        return (self.pipeline.predict(X.astype(np.float32)) == -1).astype(int)

    def predict_single(self, x: np.ndarray) -> tuple:
        """
        Predict a single request.
        Called by layer2a_anomaly.py in the FastAPI app.

        Parameters
        ----------
        x : (1, 20) float32

        Returns
        -------
        (is_anomaly: bool, score: float)
        """
        score    = float(self.anomaly_scores(x)[0])
        thr      = self.threshold if self.threshold is not None else 0.0
        is_anom  = score >= thr
        return is_anom, score

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save pipeline + threshold to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "pipeline":  self.pipeline,
                "threshold": self.threshold,
            }, f)
        print(f"[IForest] Saved → {path}")

    def load(self, path: str) -> None:
        """Load pipeline + threshold from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.pipeline  = data["pipeline"]
        self.threshold = data["threshold"]
        self._fitted   = True
        print(f"[IForest] Loaded ← {path}")

    # ── ONNX export ────────────────────────────────────────────────────────────

    def export_onnx(self, output_path: str) -> None:
        """
        Export the fitted sklearn pipeline to ONNX format.
        Also saves the anomaly threshold to a companion .txt file so the
        FastAPI app can reload it without the sklearn pipeline.

        Requires: pip install skl2onnx onnxruntime

        Parameters
        ----------
        output_path : str — e.g. "exported_models/layer2a_best.onnx"
        """
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnxruntime as ort
        import time

        if not self._fitted:
            raise RuntimeError("Model must be trained before export.")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # ── Export ────────────────────────────────────────────────────────────
        initial_types = [("features", FloatTensorType([None, INPUT_DIM]))]
        proto = convert_sklearn(
            self.pipeline,
            name="isolation_forest_l2a",
            initial_types=initial_types,
            target_opset=17,
        )
        with open(output_path, "wb") as f:
            f.write(proto.SerializeToString())

        # ── Save threshold alongside ──────────────────────────────────────────
        thr_path = output_path.replace(".onnx", "_threshold.txt")
        with open(thr_path, "w") as f:
            f.write(str(self.threshold if self.threshold is not None else 0.0))

        # ── Validate + benchmark ──────────────────────────────────────────────
        sess     = ort.InferenceSession(output_path)
        in_name  = sess.get_inputs()[0].name
        dummy    = np.random.randn(1, INPUT_DIM).astype(np.float32)

        # warmup
        for _ in range(10):
            sess.run(None, {in_name: dummy})

        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            out = sess.run(None, {in_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = round(np.mean(times), 3)
        p99_ms = round(np.percentile(times, 99), 3)
        status = "PASS" if p99_ms <= 2.0 else f"WARN (p99={p99_ms}ms > 2ms target)"

        print(f"[IForest] ONNX exported     → {output_path}")
        print(f"[IForest] Threshold saved   → {thr_path}")
        print(f"[IForest] Output shape:       {[o.shape for o in out]}")
        print(f"[IForest] Latency avg={avg_ms}ms  p99={p99_ms}ms  {status}")