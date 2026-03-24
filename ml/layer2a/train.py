"""
ml/layer2a/train.py

Trains BOTH Layer 2A candidates and saves the best one.

Usage (in Colab — run after notebook 02):
    !python -m layer2a.train \
        --train ml/data/splits/l2a_train.csv \
        --val   ml/data/splits/l2a_val.csv   \
        --out   ml/exported_models/

Alternatively, copy the logic into notebook 03 and run cell-by-cell.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.extractor import FEATURE_NAMES
from layer2a.candidates import isolation_forest as iforest
from layer2a.candidates import autoencoder_shallow as ae
from layer2a.evaluate import compare_candidates, select_best


def load_splits(train_path: str, val_path: str):
    """
    Load pre-split CSVs for L2A training.
    Both files should contain ONLY normal traffic rows.
    Label column (if present) is dropped.
    """
    df_tr  = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    for col in ["label", "class", "y", "target"]:
        df_tr  = df_tr.drop(columns=[col], errors="ignore")
        df_val = df_val.drop(columns=[col], errors="ignore")

    X_train = df_tr[FEATURE_NAMES].values.astype(np.float32)
    X_val   = df_val[FEATURE_NAMES].values.astype(np.float32)

    print(f"[train] X_train: {X_train.shape}  X_val: {X_val.shape}")
    return X_train, X_val


def load_test_split(test_path: str):
    """
    Load mixed test CSV (normal + attacks).
    Must contain a 'label' column: 0=normal, 1=attack.
    """
    df = pd.read_csv(test_path)
    y  = df["label"].values.astype(int)
    X  = df[FEATURE_NAMES].values.astype(np.float32)
    print(f"[train] X_test: {X.shape}  (normal={int((y==0).sum())} "
          f"attack={int((y==1).sum())})")
    return X, y


def run(
    train_path: str,
    val_path:   str,
    test_path:  str,
    out_dir:    str,
    device:     str = None,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_train, X_val = load_splits(train_path, val_path)
    X_test,  y_test = load_test_split(test_path)

    # ── Candidate 1: Isolation Forest ─────────────────────────────────────────
    print("\n" + "="*50)
    print("Candidate 1: Isolation Forest")
    print("="*50)
    iforest_pipe = iforest.train(X_train, run_name="iforest")
    iforest_thr  = iforest.find_threshold(iforest_pipe, X_val, None)
    # for val we need normal-only y — pass zeros
    y_val_normal = np.zeros(len(X_val), dtype=int)
    # re-evaluate on test
    iforest_results = iforest.evaluate(iforest_pipe, X_test, y_test,
                                       threshold=iforest_thr)

    # ── Candidate 2: Shallow Autoencoder ──────────────────────────────────────
    print("\n" + "="*50)
    print("Candidate 2: Shallow Autoencoder")
    print("="*50)
    ae_model, ae_threshold, ae_norm = ae.train(
        X_train, X_val, device=device, run_name="shallow_autoencoder"
    )
    ae_results = ae.evaluate(ae_model, ae_norm, ae_threshold, X_test, y_test)

    # ── Compare and select ────────────────────────────────────────────────────
    all_results = [iforest_results, ae_results]
    df_compare  = compare_candidates(all_results)
    winner      = select_best(all_results)

    print("\n" + "="*50)
    print("Comparison table:")
    print("="*50)
    print(df_compare.to_string(index=False))
    print(f"\nWinner: {winner['model']}")

    # ── Export winner to ONNX ─────────────────────────────────────────────────
    onnx_path = str(out / "layer2a_best.onnx")

    if winner["model"] == "isolation_forest":
        iforest.export_onnx(iforest_pipe, onnx_path)
        meta = {
            "model":     "isolation_forest",
            "threshold": iforest_thr,
            "scaler":    None,   # scaler is baked into the sklearn pipeline
        }
    else:
        ae.export_onnx(ae_model, onnx_path)
        scaler_path = str(out / "layer2a_scaler.pkl")
        ae_norm.save(scaler_path)
        meta = {
            "model":      "shallow_autoencoder",
            "threshold":  ae_threshold,
            "scaler":     scaler_path,
        }

    # save metadata (threshold, model name) for FastAPI to load
    meta_path = str(out / "layer2a_meta.json")
    with open(meta_path, "w") as f:
        json.dump({**meta, **winner}, f, indent=2)

    # save comparison table
    df_compare.to_csv(str(out / "layer2a_comparison.csv"), index=False)

    print(f"\n[train] All outputs saved to {out_dir}")
    print(f"  ONNX model : {onnx_path}")
    print(f"  Metadata   : {meta_path}")
    print(f"  Comparison : {out / 'layer2a_comparison.csv'}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Layer 2A candidates")
    parser.add_argument("--train",  required=True, help="Path to l2a_train.csv (normal only)")
    parser.add_argument("--val",    required=True, help="Path to l2a_val.csv (normal only)")
    parser.add_argument("--test",   required=True, help="Path to test_mixed.csv (normal + attacks)")
    parser.add_argument("--out",    default="ml/exported_models/", help="Output directory")
    parser.add_argument("--device", default=None, help="cuda | cpu")
    args = parser.parse_args()

    run(args.train, args.val, args.test, args.out, args.device)