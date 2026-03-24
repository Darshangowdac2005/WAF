"""
ml/layer2a/candidates/autoencoder_shallow.py

Layer 2A Candidate 2 — Shallow Autoencoder (PyTorch)
------------------------------------------------------
One-class anomaly detector using reconstruction error.
Trained ONLY on normal traffic. No attack labels needed.

Architecture
------------
Encoder: 25 → 64 → 32 → 16  (bottleneck)
Decoder: 16 → 32 → 64 → 25

Anomaly score = MSE(input, reconstruction)
Higher score = reconstruction failed = anomalous request

Threshold = mean_train_error + 2 × std_train_error
(computed on training set after fitting — stored alongside ONNX)

This mirrors the autoencoder approach in Base Paper 2
(Babaey & Faragardi 2025). Your standalone result can be directly
compared to their stacked ensemble in the report.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch

from feature_engineering.normalizer import Normalizer
from feature_engineering.extractor import INPUT_DIM


# ── Architecture ──────────────────────────────────────────────────────────────

HIDDEN = [64, 32, 16]   # encoder dims; decoder is the mirror


class ShallowAutoencoder(nn.Module):
    """
    Symmetric autoencoder with BatchNorm and Dropout.

    Input/output dim = INPUT_DIM (25 features).
    Bottleneck dim   = HIDDEN[-1] (16).
    """

    def __init__(
        self,
        input_dim:   int   = INPUT_DIM,
        hidden_dims: list  = HIDDEN,
        dropout:     float = 0.1,
    ):
        super().__init__()

        # Encoder
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        self.encoder = nn.Sequential(*enc)

        # Decoder (mirror — skip the bottleneck layer itself)
        dec = []
        for h in reversed(hidden_dims[:-1]):
            dec += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE error, shape (N,). Higher = more anomalous."""
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=1)


# ── Training config ────────────────────────────────────────────────────────────

TRAIN_CFG = {
    "epochs":        60,
    "batch_size":    256,
    "lr":            1e-3,
    "weight_decay":  1e-5,
    "patience":      8,
    "threshold_std": 2.0,   # threshold = mean + N * std of train recon errors
    "dropout":       0.1,
}


# ── Train ──────────────────────────────────────────────────────────────────────

def train(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    device:  str = None,
    run_name: str = "shallow_autoencoder",
) -> tuple:
    """
    Train Shallow Autoencoder on NORMAL-ONLY scaled feature vectors.

    Parameters
    ----------
    X_train  : (N_train, 25) float32 — normal traffic only, already scaled
    X_val    : (N_val,   25) float32 — normal traffic only, already scaled
    device   : "cuda" | "cpu" | None (auto-detect)
    run_name : MLflow run name

    Returns
    -------
    (model, threshold, normalizer)
        model      : trained ShallowAutoencoder (on CPU, ready to export)
        threshold  : float — anomaly score cutoff
        normalizer : fitted Normalizer (save this alongside the ONNX model)

    Note: X_train and X_val should be RAW (unscaled) — this function
    fits the Normalizer internally and returns it.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[AE] Device: {device}")

    # scale features
    norm      = Normalizer()
    X_tr_sc   = norm.fit(X_train)
    X_val_sc  = norm.transform(X_val)

    tr_ds  = TensorDataset(torch.from_numpy(X_tr_sc))
    val_ds = TensorDataset(torch.from_numpy(X_val_sc))
    tr_dl  = DataLoader(tr_ds,  batch_size=TRAIN_CFG["batch_size"],
                        shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=512)

    model     = ShallowAutoencoder(dropout=TRAIN_CFG["dropout"]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_CFG["lr"],
        weight_decay=TRAIN_CFG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )

    best_val  = float("inf")
    patience  = 0
    best_wts  = None

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**TRAIN_CFG, "device": device,
                           "input_dim": INPUT_DIM})

        for epoch in range(1, TRAIN_CFG["epochs"] + 1):
            # train
            model.train()
            tr_loss = 0.0
            for (xb,) in tr_dl:
                xb = xb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), xb)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
            tr_loss /= len(tr_dl)

            # validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (xb,) in val_dl:
                    val_loss += criterion(model(xb.to(device)), xb.to(device)).item()
            val_loss /= len(val_dl)
            scheduler.step(val_loss)

            mlflow.log_metrics({"train_loss": tr_loss, "val_loss": val_loss},
                               step=epoch)

            if epoch % 10 == 0:
                print(f"[AE] Epoch {epoch:3d} | train={tr_loss:.5f} "
                      f"| val={val_loss:.5f}")

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                best_wts = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= TRAIN_CFG["patience"]:
                    print(f"[AE] Early stop at epoch {epoch}")
                    break

        model.load_state_dict(best_wts)

        # compute threshold on training reconstruction errors
        model.eval()
        errors = []
        with torch.no_grad():
            for (xb,) in DataLoader(tr_ds, batch_size=512):
                errors.append(
                    model.reconstruction_error(xb.to(device)).cpu().numpy()
                )
        errors    = np.concatenate(errors)
        mean_e    = float(errors.mean())
        std_e     = float(errors.std())
        threshold = mean_e + TRAIN_CFG["threshold_std"] * std_e

        mlflow.log_metrics({
            "recon_mean":       mean_e,
            "recon_std":        std_e,
            "anomaly_threshold": threshold,
            "best_val_loss":    best_val,
        })
        print(f"[AE] Threshold: {threshold:.6f}  "
              f"(mean={mean_e:.5f} + {TRAIN_CFG['threshold_std']}×std={std_e:.5f})")

    model.cpu()
    return model, threshold, norm


# ── Evaluate ───────────────────────────────────────────────────────────────────

def evaluate(
    model:     ShallowAutoencoder,
    norm:      Normalizer,
    threshold: float,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
) -> dict:
    """
    Evaluate on mixed test set (0=normal, 1=attack).
    X_test should be RAW (unscaled).
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

    X_sc = norm.transform(X_test)
    X_t  = torch.from_numpy(X_sc)

    model.eval()
    with torch.no_grad():
        errors = model.reconstruction_error(X_t).numpy()

    preds = (errors > threshold).astype(int)
    auc   = roc_auc_score(y_test, errors)
    ap    = average_precision_score(y_test, errors)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    results = {
        "model":         "shallow_autoencoder",
        "auc":           round(float(auc), 4),
        "avg_precision": round(float(ap),  4),
        "fpr":           round(fpr, 4),
        "tpr":           round(tpr, 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "threshold":     round(threshold, 6),
    }

    print("\n[AE] Evaluation:")
    for k, v in results.items():
        print(f"  {k:<22}: {v}")

    return results


# ── ONNX export ────────────────────────────────────────────────────────────────

def export_onnx(model: ShallowAutoencoder, output_path: str):
    """Export PyTorch model to ONNX. Normalizer must be saved separately."""
    import onnxruntime as ort

    model.eval().cpu()
    dummy = torch.randn(1, INPUT_DIM)

    torch.onnx.export(
        model, dummy, output_path,
        input_names=["features"],
        output_names=["reconstruction"],
        dynamic_axes={
            "features":       {0: "batch"},
            "reconstruction": {0: "batch"},
        },
        opset_version=17,
    )

    sess = ort.InferenceSession(output_path)
    out  = sess.run(None, {"features": dummy.numpy()})
    print(f"[AE] ONNX exported → {output_path}")
    print(f"[AE] ONNX validation OK. Output shape: {out[0].shape}")