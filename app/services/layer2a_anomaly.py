"""app/services/layer2a_anomaly.py — ONNX anomaly detection (Layer 2A)"""
import numpy as np
import onnxruntime as ort
from app.core.config import settings
from app.core.logging import logger

_sess: ort.InferenceSession = None
_threshold: float = None
_in_name: str = "features"


def load() -> None:
    global _sess, _threshold, _in_name

    onnx_path = settings.L2A_ONNX_PATH
    thr_path = settings.L2A_THRESHOLD_PATH

    if not onnx_path.exists():
        raise FileNotFoundError(f"L2A ONNX not found: {onnx_path}")
    if not thr_path.exists():
        raise FileNotFoundError(f"L2A threshold not found: {thr_path}")

    # Tune session options for low-latency single-request CPU serving
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    _sess = ort.InferenceSession(str(onnx_path), sess_options=opts)
    _in_name = _sess.get_inputs()[0].name

    with open(thr_path, "r", encoding="utf-8") as f:
        _threshold = float(f.read().strip())

    logger.info("L2A loaded | input=%s | threshold=%.5f", _in_name, _threshold)


def infer(feature_vector: np.ndarray) -> tuple[bool, float]:
    """
    Parameters
    ----------
    feature_vector : (1, n_features) float32
        Already scaled feature vector

    Returns
    -------
    (is_anomaly: bool, score: float)
    """
    recon = _sess.run(None, {_in_name: feature_vector})[0]
    score = float(np.mean((feature_vector - recon) ** 2))
    return score >= _threshold, score