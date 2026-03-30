"""
ml/evaluation/benchmark.py

ONNX latency benchmarking for both layers.
Run from notebook 05 after exporting all models to ONNX.

Usage:
    from evaluation.benchmark import benchmark_onnx, benchmark_all, print_report
    results = benchmark_all(exported_models_dir="../exported_models")
"""

import numpy as np
import time
from pathlib import Path


# ── Single model benchmark ────────────────────────────────────────────────────

def benchmark_onnx(
    onnx_path: str,
    input_name: str,
    dummy_input: np.ndarray,
    n_warmup: int = 20,
    n_runs:   int = 200,
) -> dict:
    """
    Measure inference latency of one ONNX model.

    Parameters
    ----------
    onnx_path    : path to .onnx file
    input_name   : ONNX input tensor name ("features" or "token_ids")
    dummy_input  : np.ndarray matching the model's expected input shape
    n_warmup     : warmup runs (excluded from timing)
    n_runs       : timed runs

    Returns
    -------
    dict: mean_ms, p50_ms, p95_ms, p99_ms, min_ms, max_ms
    """
    import onnxruntime as ort

    sess  = ort.InferenceSession(onnx_path)
    feed  = {input_name: dummy_input}

    for _ in range(n_warmup):
        sess.run(None, feed)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)

    times = sorted(times)
    return {
        "model":    Path(onnx_path).stem,
        "mean_ms":  round(float(np.mean(times)), 3),
        "p50_ms":   round(float(np.percentile(times, 50)), 3),
        "p95_ms":   round(float(np.percentile(times, 95)), 3),
        "p99_ms":   round(float(np.percentile(times, 99)), 3),
        "min_ms":   round(float(times[0]), 3),
        "max_ms":   round(float(times[-1]), 3),
    }


# ── Benchmark all exported models ─────────────────────────────────────────────

def benchmark_all(exported_models_dir: str = "../exported_models") -> list:
    """
    Benchmark all .onnx files found in exported_models_dir.
    Automatically infers input format from filename:
      *l2a* or *autoencoder* or *isolation* → numeric features (1, 20) float32
      *l2b* or *cnn* or *gru* or *xgb*      → token ids (1, 512) int32

    Returns list of benchmark result dicts.
    """
    export_dir = Path(exported_models_dir)
    onnx_files = list(export_dir.glob("*.onnx"))

    if not onnx_files:
        print(f"[benchmark] No .onnx files found in {export_dir}")
        return []

    results = []
    for path in onnx_files:
        name = path.stem.lower()

        # determine input format
        if any(k in name for k in ["l2a", "autoencoder", "isolation", "iforest", "scaler"]):
            input_name  = "features"
            dummy_input = np.random.randn(1, 25).astype(np.float32)
            target_ms   = 2.0
            layer       = "L2A"
        else:
            input_name  = "token_ids"
            dummy_input = np.zeros((1, 512), dtype=np.int32)
            target_ms   = 20.0
            layer       = "L2B"

        print(f"[benchmark] Benchmarking {path.name} ({layer})...")
        try:
            res = benchmark_onnx(str(path), input_name, dummy_input)
            res["layer"]     = layer
            res["target_ms"] = target_ms
            res["pass"]      = res["p99_ms"] <= target_ms
            results.append(res)
        except Exception as e:
            print(f"[benchmark] ERROR on {path.name}: {e}")

    return results


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_report(results: list) -> None:
    """Print a formatted latency report table."""
    if not results:
        print("[benchmark] No results to display.")
        return

    header = (
        f"{'Model':<30} {'Layer':<5} {'Mean':>8} {'P95':>8} {'P99':>8} "
        f"{'Target':>8} {'Pass':>6}"
    )
    print("\n" + "=" * len(header))
    print("ONNX Inference Latency Report")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        status = "YES" if r.get("pass") else "NO "
        print(
            f"{r['model']:<30} {r.get('layer','?'):<5} "
            f"{r['mean_ms']:>7.2f}ms {r['p95_ms']:>7.2f}ms "
            f"{r['p99_ms']:>7.2f}ms {r.get('target_ms','-'):>7}ms "
            f"  {status}"
        )

    print("=" * len(header))
    passed = sum(1 for r in results if r.get("pass"))
    print(f"Passed: {passed}/{len(results)}")


# ── Per-model convenience functions ───────────────────────────────────────────

def benchmark_isolation_forest(path: str) -> dict:
    dummy = np.random.randn(1, 25).astype(np.float32)
    res   = benchmark_onnx(path, "features", dummy)
    res["pass"] = res["p99_ms"] <= 2.0
    return res


def benchmark_autoencoder(path: str) -> dict:
    dummy = np.random.randn(1, 25).astype(np.float32)
    res   = benchmark_onnx(path, "features", dummy)
    res["pass"] = res["p99_ms"] <= 2.0
    return res


def benchmark_xgboost(path: str) -> dict:
    dummy = np.random.randn(1, 25).astype(np.float32)
    res   = benchmark_onnx(path, "features", dummy)
    res["pass"] = res["p99_ms"] <= 20.0
    return res


def benchmark_cnn_1d(path: str) -> dict:
    dummy = np.zeros((1, 512), dtype=np.int32)
    res   = benchmark_onnx(path, "token_ids", dummy)
    res["pass"] = res["p99_ms"] <= 20.0
    return res


def benchmark_gru(path: str) -> dict:
    dummy = np.zeros((1, 512), dtype=np.int32)
    res   = benchmark_onnx(path, "token_ids", dummy)
    res["pass"] = res["p99_ms"] <= 20.0
    return res