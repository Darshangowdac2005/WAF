"""app/services/layer2b_deep.py — ONNX deep classifier (Layer 2B)"""
import numpy as np
import onnxruntime as ort
import scipy.special
from app.core.config import settings
from app.core.logging import logger

_sess = None
_in_name = None

# MUST match training label order exactly.
# Current trained model: 0=normal,1=sqli,2=xss,3=lfi,4=other_attack (5 classes)
# Next retrain target:   0=normal,1=sqli,2=xss,3=lfi,4=other_attack,5=cmdi (6 classes)
# NOTE: cmdi was previously misplaced at index 4 — fixed here.
CLASS_NAMES = [
    "normal",
    "sqli",
    "xss",
    "lfi",
    "other_attack",   # index 4 — matches current trained model
    "cmdi",           # index 5 — active after 6-class retrain
]

# if your ONNX model expects token_ids, keep True
_uses_tokens = True


def load() -> None:
    global _sess, _in_name

    onnx_path = settings.L2B_ONNX_PATH
    if not onnx_path.exists():
        raise FileNotFoundError(f"L2B ONNX not found: {onnx_path}")

    # Tune session options for low-latency single-request CPU serving
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 2   # avoids spin-up overhead on small models
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    _sess = ort.InferenceSession(str(onnx_path), sess_options=opts)
    _in_name = _sess.get_inputs()[0].name

    logger.info("L2B loaded | input=%s | uses_tokens=%s | classes=%d",
                _in_name, _uses_tokens, len(CLASS_NAMES))


def infer(fvec_scaled: np.ndarray, token_ids: np.ndarray):
    """
    Returns
    -------
    label, confidence, probabilities
    """
    if _uses_tokens:
        logits = _sess.run(None, {_in_name: token_ids.astype(np.int64)})[0][0]
    else:
        logits = _sess.run(None, {_in_name: fvec_scaled.astype(np.float32)})[0][0]

    probs = scipy.special.softmax(logits)
    pred_cls = int(np.argmax(probs))
    pred_conf = float(probs[pred_cls])
    pred_label = CLASS_NAMES[pred_cls]

    return pred_label, pred_conf, probs.tolist()