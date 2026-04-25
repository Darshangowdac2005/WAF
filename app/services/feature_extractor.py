"""app/services/feature_extractor.py
Runtime preprocessing that MUST exactly match training preprocessing.
Uses the same:
- extractor
- tokenizer
- Normalizer
from ml/ used during training.
"""
import os
import sys
from pathlib import Path

import numpy as np

from app.core.config import settings

# ---------------------------------------------------------------------
# Make ml/ importable
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
ML_PATH = os.environ.get("ML_PATH", str(ROOT / "ml"))

if ML_PATH not in sys.path:
    sys.path.insert(0, ML_PATH)

# ---------------------------------------------------------------------
# Import EXACT training-side code
# ---------------------------------------------------------------------
from ml.feature_engineering.extractor import extract_features, to_vector  # noqa: E402
from ml.feature_engineering.tokenizer import CharTokenizer                # noqa: E402
from ml.feature_engineering.normalizer import Normalizer           # noqa: E402

# ---------------------------------------------------------------------
# Shared tokenizer / normalizer
# ---------------------------------------------------------------------
_tokenizer = CharTokenizer(max_len=256)  # reduced from 512 to cut L2B latency ~40-50%
_normalizer = None


def _load_normalizer():
    global _normalizer
    if _normalizer is None:
        scaler_path = settings.SCALER_PATH
        if not scaler_path.exists():
            raise FileNotFoundError(f"Normalizer file not found: {scaler_path}")
        _normalizer = Normalizer.load(str(scaler_path))
    return _normalizer


def extract(request: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    request : dict
        Keys expected:
        - url
        - method
        - headers
        - body
        - ip

    Returns
    -------
    fvec_scaled : np.ndarray
        Shape (1, 25), dtype float32
        EXACT same normalized feature vector used in training

    token_ids : np.ndarray
        Shape (1, 512), dtype int64
        EXACT same tokenizer output used in training
    """
    # 1) Exact training-side feature extraction
    feats = extract_features(request)

    # 2) Exact training-side feature order
    fvec = to_vector(feats).astype(np.float32)

    if fvec.ndim == 1:
        fvec = fvec.reshape(1, -1)

    # 3) Exact training-side normalizer
    norm = _load_normalizer()
    fvec_scaled = norm.transform(fvec).astype(np.float32)

    # 4) Exact training-side tokenizer
    token_ids = _tokenizer.encode_request(request).reshape(1, -1).astype(np.int64)

    return fvec_scaled, token_ids