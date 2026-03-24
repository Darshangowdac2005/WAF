"""
ml/layer2a/export_onnx.py

Standalone ONNX export script for Layer 2A.
Use this if you want to re-export a saved model without re-training.

Usage:
    python -m layer2a.export_onnx \
        --model isolation_forest \
        --model-path ml/exported_models/iforest_pipe.pkl \
        --out ml/exported_models/layer2a_best.onnx

    python -m layer2a.export_onnx \
        --model autoencoder \
        --model-path ml/exported_models/ae_model.pt \
        --out ml/exported_models/layer2a_best.onnx
"""

import argparse
import torch
import joblib
import numpy as np
import onnxruntime as ort
import time

from feature_engineering.extractor import INPUT_DIM


def export_sklearn(model_path: str, output_path: str):
    """Export a saved sklearn pipeline (IsolationForest) to ONNX."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    pipe  = joblib.load(model_path)
    proto = convert_sklearn(
        pipe,
        name="isolation_forest_l2a",
        initial_types=[("features", FloatTensorType([None, INPUT_DIM]))],
        target_opset=17,
    )
    with open(output_path, "wb") as f:
        f.write(proto.SerializeToString())
    print(f"[export] sklearn → ONNX: {output_path}")


def export_pytorch(model_path: str, output_path: str):
    """Export a saved PyTorch ShallowAutoencoder to ONNX."""
    from layer2a.candidates.autoencoder_shallow import ShallowAutoencoder

    model = ShallowAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, INPUT_DIM)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["features"],
        output_names=["reconstruction"],
        dynamic_axes={"features": {0: "batch"}, "reconstruction": {0: "batch"}},
        opset_version=17,
    )
    print(f"[export] PyTorch → ONNX: {output_path}")


def benchmark(onnx_path: str, n_runs: int = 200):
    """Benchmark ONNX inference latency and check against 2ms target."""
    sess  = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, INPUT_DIM).astype(np.float32)

    # warmup
    for _ in range(20):
        sess.run(None, {"features": dummy})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"features": dummy})
        times.append((time.perf_counter() - t0) * 1000)

    times  = sorted(times)
    mean   = round(float(np.mean(times)), 3)
    p95    = round(float(np.percentile(times, 95)), 3)
    p99    = round(float(np.percentile(times, 99)), 3)
    status = "PASS" if p99 <= 2.0 else "FAIL — above 2ms target"

    print(f"\n[benchmark] Layer 2A ONNX latency ({n_runs} runs):")
    print(f"  Mean : {mean} ms")
    print(f"  P95  : {p95} ms")
    print(f"  P99  : {p99} ms")
    print(f"  Result: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True,
                        choices=["isolation_forest", "autoencoder"],
                        help="Which model type to export")
    parser.add_argument("--model-path", required=True, help="Path to saved model")
    parser.add_argument("--out",        required=True, help="Output .onnx path")
    parser.add_argument("--benchmark",  action="store_true",
                        help="Run latency benchmark after export")
    args = parser.parse_args()

    if args.model == "isolation_forest":
        export_sklearn(args.model_path, args.out)
    else:
        export_pytorch(args.model_path, args.out)

    if args.benchmark:
        benchmark(args.out)