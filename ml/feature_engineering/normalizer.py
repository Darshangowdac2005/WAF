"""
ml/feature_engineering/normalizer.py

Wraps sklearn StandardScaler with save/load so the SAME scaler
fitted on training data is used at inference time.

Why a separate file?
--------------------
Isolation Forest and the Shallow Autoencoder both need scaled features.
XGBoost is tree-based (scale-invariant) but we scale it anyway so the
pipeline is consistent and the same scaler can be reused everywhere.

The scaler is fitted ONCE on the training split and saved to disk.
At FastAPI startup it is loaded alongside the ONNX model.

Usage
-----
Training:
    norm = Normalizer()
    X_train_scaled = norm.fit(X_train)
    X_val_scaled   = norm.transform(X_val)
    norm.save("ml/exported_models/scaler_l2a.pkl")

Inference:
    norm = Normalizer.load("ml/exported_models/scaler_l2a.pkl")
    X_scaled = norm.transform(feature_vector)
"""

import numpy as np
import joblib  # type: ignore
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class Normalizer:
    """Thin wrapper around StandardScaler with save/load helpers."""

    def __init__(self):
        self._scaler = StandardScaler()
        self._fitted = False

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fit scaler on X and return scaled X.
        Call ONLY on training data — never on val/test.
        """
        X_scaled    = self._scaler.fit_transform(X.astype(np.float32))
        self._fitted = True
        return X_scaled.astype(np.float32)

    # ── transform ─────────────────────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted scaler to X. Must call fit() first."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return self._scaler.transform(X.astype(np.float32)).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one call (convenience)."""
        return self.fit(X)

    # ── stats (for reporting) ─────────────────────────────────────────────────

    @property
    def mean_(self) -> np.ndarray:
        return self._scaler.mean_ # type: ignore

    @property
    def scale_(self) -> np.ndarray:
        return self._scaler.scale_ # type: ignore

    def feature_stats(self, feature_names: list) -> dict:
        """Return mean and std per feature — useful for sanity-checking."""
        return {
            name: {"mean": float(m), "std": float(s)}
            for name, m, s in zip(feature_names, self.mean_, self.scale_)
        }

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save fitted scaler to disk as a .pkl file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, path)
        print(f"[Normalizer] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        """Load a previously saved scaler from disk."""
        norm = cls()
        norm._scaler = joblib.load(path)
        norm._fitted = True
        print(f"[Normalizer] Loaded from {path}")
        return norm