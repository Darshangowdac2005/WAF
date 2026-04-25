"""
ml/layer2b/candidates/xgboost_model.py

Layer 2B Candidate 1 — XGBoost multi-class classifier
Input: 20-dim numeric feature vector from extractor.py (NOT token sequences)
Trains on CPU in < 2 minutes on CSIC 2012. No GPU needed.
Provides feature_importance() for explainability.
"""

import numpy as np
import pickle
import mlflow
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier

CLASS_NAMES = ["normal", "sqli", "xss", "lfi", "other_attack", "cmdi"]
NUM_CLASSES  = len(CLASS_NAMES)

PARAMS = {
    "n_estimators":        500,
    "max_depth":           6,
    "learning_rate":       0.05,
    "subsample":           0.8,
    "colsample_bytree":    0.8,
    "min_child_weight":    1,
    "gamma":               0.1,
    "reg_alpha":           0.1,
    "reg_lambda":          1.0,
    "eval_metric":         "mlogloss",
    "objective":           "multi:softprob",
    "num_class":           NUM_CLASSES,
    "tree_method":         "hist",
    "random_state":        42,
    "n_jobs":              -1,
    "early_stopping_rounds": 20,
}


class XGBoostModel:

    def __init__(self):
        self.scaler = StandardScaler()
        self.xgb    = None
        self._fitted = False

    # ── Updated Training Method ───────────────────────────────────────────────

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              run_name: str = "xgboost_l2b") -> None:
        """
        X arrays are numeric feature vectors (N, 29).
        Applies StandardScaler internally.
        """
        with mlflow.start_run(run_name=run_name, nested=True):
            # Log parameters to MLflow
            mlflow.log_params(PARAMS)

            # Preprocess data
            X_tr = self.scaler.fit_transform(X_train).astype(np.float32)
            X_v  = self.scaler.transform(X_val).astype(np.float32)

            # Extract parameters for the classifier
            # Modern XGBoost: early_stopping_rounds goes in the constructor
            xgb_params = {k: v for k, v in PARAMS.items()}
            
            self.xgb = XGBClassifier(**xgb_params)

            # Modern fit() call: early_stopping_rounds is removed from here
            # as it is already defined in self.xgb via the PARAMS dict
            self.xgb.fit(
                X_tr, y_train,
                eval_set=[(X_v, y_val)],
                verbose=100
            )
            
            self._fitted = True

            # Evaluate validation performance
            val_preds = self.xgb.predict(X_v)
            val_f1    = f1_score(y_val, val_preds, average="macro", zero_division=0)
            val_acc   = accuracy_score(y_val, val_preds)
            mlflow.log_metrics({"val_f1_macro": val_f1, "val_accuracy": val_acc})

        print(f"[XGB] Best iter={self.xgb.best_iteration}  "
              f"val_f1={val_f1:.4f}  val_acc={val_acc:.4f}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns class indices (N,)."""
        return self.xgb.predict(self.scaler.transform(X.astype(np.float32)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probabilities (N, 5)."""
        return self.xgb.predict_proba(self.scaler.transform(X.astype(np.float32)))

    def predict_single(self, x: np.ndarray) -> dict:
        """
        Predict one request. x shape (1, 20).
        Returns dict with class label, confidence, all probabilities.
        """
        proba     = self.predict_proba(x)[0]
        pred_idx  = int(np.argmax(proba))
        return {
            "label":      CLASS_NAMES[pred_idx],
            "confidence": round(float(proba[pred_idx]), 4),
            "proba":      {c: round(float(p), 4)
                           for c, p in zip(CLASS_NAMES, proba)},
        }

    # ── Explainability ────────────────────────────────────────────────────────

    def feature_importance(self, feature_names: list) -> dict:
        """Return feature importances sorted descending. Use in your report."""
        fi = dict(zip(feature_names, self.xgb.feature_importances_))
        return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"xgb": self.xgb, "scaler": self.scaler}, f)
        print(f"[XGB] Saved → {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.xgb    = data["xgb"]
        self.scaler = data["scaler"]
        self._fitted = True
        print(f"[XGB] Loaded ← {path}")

    # ── ONNX export ───────────────────────────────────────────────────────────

    def export_onnx(self, output_path: str, input_dim: int = 29) -> None:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([("scaler", self.scaler), ("xgb", self.xgb)])
        proto = convert_sklearn(
            pipeline,
            name="xgboost_l2b",
            initial_types=[("features", FloatTensorType([None, input_dim]))],
            target_opset=17,
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(proto.SerializeToString())

        import onnxruntime as ort, time
        sess  = ort.InferenceSession(output_path)
        dummy = np.random.randn(1, input_dim).astype(np.float32)
        times = [time.perf_counter() for _ in range(50)]
        for _ in range(50): sess.run(None, {"features": dummy})
        avg_ms = np.mean([
            (time.perf_counter() - t) * 1000 for t in times
        ])
        print(f"[XGB] ONNX exported → {output_path}  avg={avg_ms:.2f}ms")