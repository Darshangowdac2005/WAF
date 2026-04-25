"""
ml/layer2b/train.py

Trains all three Layer 2B candidates, evaluates them,
picks the winner, and exports to ONNX.

Data requirements in ml/data/splits/:
    l2b_train_X_numeric.npy   (N, 29) float32  — for XGBoost
    l2b_train_X_tokens.npy    (N, 256) int32    — for CNN, GRU
    l2b_train_y.npy           (N,)    int
    l2b_val_X_numeric.npy, l2b_val_X_tokens.npy, l2b_val_y.npy
    l2b_test_X_numeric.npy,  l2b_test_X_tokens.npy,  l2b_test_y.npy

Labels: 0=normal, 1=sqli, 2=xss, 3=lfi, 4=other_attack, 5=cmdi
"""

import numpy as np
import json
from pathlib import Path

from candidates.xgboost_model import XGBoostModel
from candidates.cnn_1d import CNN1DModel
from candidates.gru import GRUModel
from evaluate import evaluate_candidate, pick_best

SPLITS_DIR = Path("../data/splits")
EXPORT_DIR = Path("../exported_models")
EXPORT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("Layer 2B — Training all three candidates")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────────
    X_tr_num  = np.load(SPLITS_DIR / "l2b_train_X_numeric.npy").astype(np.float32)
    X_v_num   = np.load(SPLITS_DIR / "l2b_val_X_numeric.npy").astype(np.float32)
    X_te_num  = np.load(SPLITS_DIR / "l2b_test_X_numeric.npy").astype(np.float32)

    X_tr_tok  = np.load(SPLITS_DIR / "l2b_train_X_tokens.npy").astype(np.int32)
    X_v_tok   = np.load(SPLITS_DIR / "l2b_val_X_tokens.npy").astype(np.int32)
    X_te_tok  = np.load(SPLITS_DIR / "l2b_test_X_tokens.npy").astype(np.int32)

    y_train   = np.load(SPLITS_DIR / "l2b_train_y.npy").astype(int)
    y_val     = np.load(SPLITS_DIR / "l2b_val_y.npy").astype(int)
    y_test    = np.load(SPLITS_DIR / "l2b_test_y.npy").astype(int)

    print(f"Train: {X_tr_num.shape}  |  classes: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Val:   {X_v_num.shape}")
    print(f"Test:  {X_te_num.shape}")

    all_results = {}
    all_models  = {}

    # ── Candidate 1: XGBoost (numeric features) ───────────────────────────────
    print("\n--- Candidate 1: XGBoost ---")
    xgb = XGBoostModel()
    xgb.train(X_tr_num, y_train, X_v_num, y_val)
    res_xgb = evaluate_candidate(xgb, X_te_num, y_test, name="xgboost")
    all_results["xgboost"] = res_xgb
    all_models["xgboost"]  = xgb

    # ── Candidate 2: 1D CNN (token sequences) ─────────────────────────────────
    print("\n--- Candidate 2: 1D CNN ---")
    cnn = CNN1DModel()
    cnn.train(X_tr_tok, y_train, X_v_tok, y_val)
    res_cnn = evaluate_candidate(cnn, X_te_tok, y_test, name="cnn_1d")
    all_results["cnn_1d"] = res_cnn
    all_models["cnn_1d"]  = cnn

    # ── Candidate 3: GRU (token sequences) ────────────────────────────────────
    print("\n--- Candidate 3: GRU ---")
    gru = GRUModel()
    gru.train(X_tr_tok, y_train, X_v_tok, y_val)
    res_gru = evaluate_candidate(gru, X_te_tok, y_test, name="gru")
    all_results["gru"] = res_gru
    all_models["gru"]  = gru

    # ── Pick winner ───────────────────────────────────────────────────────────
    winner_name, winner_model = pick_best(
        list(all_results.values()),
        all_models,
    )

    # ── Export winner to ONNX ─────────────────────────────────────────────────
    print(f"\n[train] Exporting winner: {winner_name}")
    winner_model.export_onnx(str(EXPORT_DIR / "layer2b_best.onnx"))

    # ── Save full results ─────────────────────────────────────────────────────
    results_path = EXPORT_DIR / "l2b_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "results": list(all_results.values()),
            "winner":  winner_name,
        }, f, indent=2)
    print(f"[train] Results saved → {results_path}")

    print("\n" + "=" * 60)
    print(f"Layer 2B done. Winner: {winner_name.upper()}")
    print("=" * 60)


if __name__ == "__main__":
    main()