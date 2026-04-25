"""
create_dummy_models.py
Generates stub ONNX models using the 'onnx' package.
Models behave as identity/no-op so the WAF server can boot for dev/demo.

L2A : Identity autoencoder  (1,29) float32 -> (1,29) float32  [25 orig + 4 new security features]
L2B : Slice+Cast+MatMul      (1,256) int64  -> (1,6)  float32  [6 classes incl. cmdi]
Scaler: StandardScaler (identity, 29 features)
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import onnx
from onnx import helper, TensorProto, numpy_helper

OUT = Path("ml/exported_models")
OUT.mkdir(parents=True, exist_ok=True)


# ─── L2A: Identity  features(1,29) -> output(1,29) ─────────────────────────────────
def make_l2a():
    # 29 features = 25 original + 4 new (has_ssrf, has_xxe, has_crlf, has_open_redirect)
    N_FEAT = 29
    node = helper.make_node("Identity", inputs=["features"], outputs=["output"])

    graph = helper.make_graph(
        [node],
        "l2a_graph",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, N_FEAT])],
        [helper.make_tensor_value_info("output",   TensorProto.FLOAT, [None, N_FEAT])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    path = OUT / "layer2a_best.onnx"
    onnx.save(model, str(path))
    print(f"[L2A] saved: {path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(path))
    x = np.random.randn(1, N_FEAT).astype(np.float32)
    res = sess.run(None, {"features": x})[0]
    assert res.shape == (1, N_FEAT) and np.allclose(res, x)
    print("[L2A] runtime OK")


# ─── L2B: Slice first 6 cols, Cast, MatMul with I_6 ─────────────────────────────────
def make_l2b():
    # Input: (1, 256) int64 tokens  Output: (1, 6) float32 logits
    # 6 classes: normal, sqli, xss, lfi, other_attack, cmdi
    SEQ_LEN = 256  # reduced from 512 to match runtime tokenizer max_len
    N_CLASSES = 6

    starts_init = numpy_helper.from_array(np.array([0], np.int64),           name="sl_starts")
    ends_init   = numpy_helper.from_array(np.array([N_CLASSES], np.int64),   name="sl_ends")
    axes_init   = numpy_helper.from_array(np.array([1], np.int64),           name="sl_axes")
    W6_init     = numpy_helper.from_array(np.eye(N_CLASSES, dtype=np.float32), name="W6")

    n_slice = helper.make_node("Slice",
                               ["token_ids", "sl_starts", "sl_ends", "sl_axes"],
                               ["sliced"])
    n_cast  = helper.make_node("Cast", ["sliced"], ["sliced_f"], to=TensorProto.FLOAT)
    n_mm    = helper.make_node("MatMul", ["sliced_f", "W6"], ["logits"])

    graph = helper.make_graph(
        [n_slice, n_cast, n_mm],
        "l2b_graph",
        [helper.make_tensor_value_info("token_ids", TensorProto.INT64,  [None, SEQ_LEN])],
        [helper.make_tensor_value_info("logits",    TensorProto.FLOAT, [None, N_CLASSES])],
        initializer=[starts_init, ends_init, axes_init, W6_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    path = OUT / "layer2b_best.onnx"
    onnx.save(model, str(path))
    print(f"[L2B] saved: {path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(path))
    x = np.zeros((1, SEQ_LEN), dtype=np.int64)
    res = sess.run(None, {"token_ids": x})[0]
    assert res.shape == (1, N_CLASSES), f"Bad shape: {res.shape}"
    print(f"[L2B] runtime OK  logits={res}  (seq_len={SEQ_LEN}, classes={N_CLASSES})")


# ─── Scaler + Threshold ────────────────────────────────────────────────────────────────────
def make_scaler():
    # 29 features = 25 original + 4 new security signal features
    sc = StandardScaler()
    sc.fit(np.eye(29, dtype=np.float32))
    path = OUT / "scaler_l2a.pkl"
    joblib.dump(sc, str(path))
    print(f"[Scaler] saved: {path}  (29 features)")


def make_threshold():
    # MSE of Identity op = 0, threshold=0.5 means all requests are "allow"
    path = OUT / "layer2a_best_threshold.txt"
    path.write_text("0.5")
    print(f"[Threshold] saved: {path}")


if __name__ == "__main__":
    print("=" * 55)
    print(f"onnx version: {onnx.__version__}")
    print("Building dummy ONNX models...")
    print("=" * 55)
    make_l2a()
    make_l2b()
    make_scaler()
    make_threshold()
    print("\nAll model artefacts ready in ml/exported_models/")
