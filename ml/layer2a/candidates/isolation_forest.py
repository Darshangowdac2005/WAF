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
    model.tune_threshold(X_normal_val, X_attack_val, target_fpr=0.05)
    result = evaluate_candidate(model, X_test, y_test, name="isolation_forest")
    model.export_onnx("exported_models/layer2a_best.onnx")

Anomaly score used by threat_scorer.py
---------------------------------------
    score = model.anomaly_scores(x)   # higher = more anomalous
    # mapped to 0-50 range in threat_scorer:
    #   l2a_contrib = min(50, score * 15)

THRESHOLD TUNING — history of bugs and final fix
-------------------------------------------------
Bug 1 (mixed val set, low→high sweep):
    _find_threshold received a COMBINED normal+attack array.
    At low thresholds, few normals in the mix made FPR look ≤ 5%.
    Sweep direction low→high picked the very first qualifying threshold.
    Result: threshold ≈ 0, TN=0, FPR=1.0 on test set.

Bug 2 (separate sets but sweep from absolute max):
    Sweep started at max(all scores) which could be very large.
    At that extreme value almost nothing is flagged → FPR=0 immediately.
    First candidate wins → threshold stuck at absurdly high value.
    Result: threshold=24.07, 70% of requests passed as normal, LFI=4%.

Final fix — bound sweep to normal score percentile range [P50 … P99]:
    At normal_P99 ≈ 1% of normal traffic is flagged  → FPR ≈ 1%  (strict end)
    At normal_P50 ≈ 50% of normal traffic is flagged → FPR ≈ 50% (lenient end)
    Sweep HIGH→LOW within this window. Correct threshold always lies here.
    Prevents both near-zero (Bug 1) and absurdly-high (Bug 2) outcomes.
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
)


# ── Hyperparameters ────────────────────────────────────────────────────────────

PARAMS = {
    "n_estimators":  200,
    "contamination": 0.05,
    "max_features":  1.0,
    "max_samples":   "auto",
    "bootstrap":     False,
    "random_state":  42,
    "n_jobs":        -1,
}

INPUT_DIM = 29   # must match len(FEATURE_NAMES) in extractor.py


# ── Model class ────────────────────────────────────────────────────────────────

class IsolationForestModel:
    """
    Wrapper around sklearn IsolationForest that exposes the standard
    Layer 2A interface expected by layer2a/evaluate.py and train.py.

        .train(X_normal)
        .tune_threshold(X_normal_val, X_attack_val, target_fpr=0.05)
        .anomaly_scores(X)   -> np.ndarray  (higher = more anomalous)
        .predict(X)          -> np.ndarray  (1=anomaly, 0=normal)
        .predict_single(x)   -> (is_anomaly: bool, score: float)
        .export_onnx(path)
        .save(path) / .load(path)
    """

    def __init__(self):
        self.pipeline  = self._build()
        self.threshold = None
        self._fitted   = False
        # stored in train() — used as sweep bounds in tune_threshold()
        self._normal_score_p50 = None
        self._normal_score_p99 = None

    # ── Build pipeline ─────────────────────────────────────────────────────────

    @staticmethod
    def _build() -> Pipeline:
        """
        StandardScaler → IsolationForest pipeline.
        Scaling ensures no single feature dominates the split selection.
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
        X_normal : (N, 25) float32 — normal traffic feature vectors only.
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

            # Store normal score percentiles — used as sweep bounds in tune_threshold
            # so the sweep is always constrained to the meaningful range.
            train_scores           = self.anomaly_scores(X_normal)
            self._normal_score_p50 = float(np.percentile(train_scores, 50))
            self._normal_score_p99 = float(np.percentile(train_scores, 99))

            mlflow.log_metrics({
                "normal_score_p50": self._normal_score_p50,
                "normal_score_p99": self._normal_score_p99,
            })

        print(f"[IForest] Trained on {len(X_normal):,} normal samples.")
        print(f"[IForest] Normal score distribution on training set:")
        print(f"  P50={self._normal_score_p50:.5f}  P99={self._normal_score_p99:.5f}")

    # ── Threshold tuning ───────────────────────────────────────────────────────

    def tune_threshold(
        self,
        X_normal_val: np.ndarray,
        X_attack_val: np.ndarray,
        target_fpr:   float = 0.05,
        max_fnr:      float = 0.05,
        n_steps:      int   = 500,
    ) -> float:
        """
        Find the anomaly score threshold using SEPARATE normal and attack
        validation splits.

        Parameters
        ----------
        X_normal_val : (N, 29) float32 — normal-only validation samples.
                       FPR is measured here. Never mix in attack samples.
        X_attack_val : (M, 29) float32 — attack-only validation samples.
                       TPR is measured here.
        target_fpr   : float — max acceptable false positive rate (default 0.05)
        max_fnr      : float — NEW: cap false-negative rate at 5% (min TPR 95%)
        n_steps      : int   — sweep resolution (default 500)

        Returns
        -------
        float — chosen threshold (also stored as self.threshold)
        """
        if self._normal_score_p50 is None:
            raise RuntimeError("Call train() before tune_threshold().")

        normal_scores = self.anomaly_scores(X_normal_val)
        attack_scores = self.anomaly_scores(X_attack_val)

        # ── Bound sweep to normal score percentile range ──────────────────────
        lo = float(np.percentile(normal_scores, 50))   # ~50% FPR at this point
        hi = float(np.percentile(normal_scores, 99))   # ~1%  FPR at this point

        print(f"[IForest] Threshold sweep: [{lo:.5f} … {hi:.5f}]")
        
        best_thr, best_tpr = None, 0.0

        for thr in np.linspace(hi, lo, n_steps):
            # FPR on normal-only val
            fp  = int(np.sum(normal_scores >= thr))
            tn  = int(np.sum(normal_scores <  thr))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            # TPR on attack-only val
            tp  = int(np.sum(attack_scores >= thr))
            fn  = int(np.sum(attack_scores <  thr))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fnr = 1.0 - tpr

            if tn == 0: continue

            # NEW: prioritize target_fpr, but also enforce max_fnr if possible
            if fpr <= target_fpr and fnr <= max_fnr:
                best_tpr = tpr
                best_thr = float(thr)
                break # Stop at strictest threshold that meets both
            
            if fpr <= target_fpr and tpr > best_tpr:
                best_tpr = tpr
                best_thr = float(thr)

        # ── Handle no qualifying threshold ────────────────────────────────────
        if best_thr is None:
            # Fall back to normal_P99: approximately 1% FPR, whatever TPR that gives
            best_thr = hi
            print(f"[IForest] WARNING: no threshold in [{lo:.5f}…{hi:.5f}] "
                  f"meets FPR≤{target_fpr}. Falling back to normal P99={hi:.5f}.")

        self.threshold = best_thr

        # ── Report final metrics ──────────────────────────────────────────────
        fp_f = int(np.sum(normal_scores >= self.threshold))
        tn_f = int(np.sum(normal_scores <  self.threshold))
        tp_f = int(np.sum(attack_scores >= self.threshold))
        fn_f = int(np.sum(attack_scores <  self.threshold))

        print(f"[IForest] Tuned threshold = {self.threshold:.6f}")
        print(f"  Val normal  → FPR={fp_f/(fp_f+tn_f):.4f}  "
              f"(FP={fp_f}, TN={tn_f})  target≤{target_fpr}")
        print(f"  Val attacks → TPR={tp_f/(tp_f+fn_f):.4f}  "
              f"(TP={tp_f}, FN={fn_f})")

        return self.threshold

    # ── Inference ──────────────────────────────────────────────────────────────

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores. Higher = more anomalous.
        IsolationForest.decision_function() returns higher values for
        normal samples, so we negate it.

        Parameters
        ----------
        X : (N, 25) float32 — feature vectors (un-normalised;
            the pipeline's StandardScaler handles it internally).
        """
        return -self.pipeline.decision_function(X.astype(np.float32))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = anomaly, 0 = normal."""
        if self.threshold is not None:
            return (self.anomaly_scores(X) >= self.threshold).astype(int)
        return (self.pipeline.predict(X.astype(np.float32)) == -1).astype(int)

    def predict_single(self, x: np.ndarray) -> tuple:
        """
        Predict one request. Returns (is_anomaly: bool, score: float).
        x shape: (1, 25)
        """
        score   = float(self.anomaly_scores(x)[0])
        thr     = self.threshold if self.threshold is not None else 0.0
        is_anom = score >= thr
        return is_anom, score

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "pipeline":           self.pipeline,
                "threshold":          self.threshold,
                "normal_score_p50":   self._normal_score_p50,
                "normal_score_p99":   self._normal_score_p99,
            }, f)
        print(f"[IForest] Saved → {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.pipeline           = data["pipeline"]
        self.threshold          = data["threshold"]
        self._normal_score_p50  = data.get("normal_score_p50")
        self._normal_score_p99  = data.get("normal_score_p99")
        self._fitted            = True
        print(f"[IForest] Loaded ← {path}")

    # ── ONNX export ────────────────────────────────────────────────────────────

    def export_onnx(self, output_path: str) -> None:
        """
        Export the fitted sklearn pipeline to ONNX.
        Companion threshold file is saved alongside the .onnx.
        Requires: pip install skl2onnx onnxruntime
        """
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnxruntime as ort
        import time

        if not self._fitted:
            raise RuntimeError("Model must be trained before export.")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        initial_types = [("features", FloatTensorType([None, INPUT_DIM]))]
        # target_opset as dict handles the skl2onnx ai.onnx.ml version requirement
        target_opsets = {
            "":           17,
            "ai.onnx.ml": 3,
        }
        proto = convert_sklearn(
            self.pipeline,
            name="isolation_forest_l2a",
            initial_types=initial_types,
            target_opset=target_opsets,
        )
        with open(output_path, "wb") as f:
            f.write(proto.SerializeToString())

        # Save threshold alongside ONNX so FastAPI can load it at startup
        thr_path = output_path.replace(".onnx", "_threshold.txt")
        with open(thr_path, "w") as f:
            f.write(str(self.threshold if self.threshold is not None else 0.0))

        # Validate and benchmark
        sess    = ort.InferenceSession(output_path)
        in_name = sess.get_inputs()[0].name
        dummy   = np.random.randn(1, INPUT_DIM).astype(np.float32)

        # warmup
        for _ in range(10):
            sess.run(None, {in_name: dummy})

        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            out = sess.run(None, {in_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = round(float(np.mean(times)), 3)
        p99_ms = round(float(np.percentile(times, 99)), 3)
        status = "PASS" if p99_ms <= 2.0 else f"WARN (p99={p99_ms}ms > 2ms target)"

        print(f"[IForest] ONNX exported     → {output_path}")
        print(f"[IForest] Threshold saved   → {thr_path}  (value={self.threshold:.6f})")
        print(f"[IForest] Output shape:       {[o.shape for o in out]}")
        print(f"[IForest] avg={avg_ms}ms  p99={p99_ms}ms  {status}")