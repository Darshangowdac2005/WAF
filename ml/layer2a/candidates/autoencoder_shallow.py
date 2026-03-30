"""
ml/layer2a/candidates/autoencoder_shallow.py

Layer 2A Candidate 2 — Shallow Autoencoder (PyTorch)
One-class anomaly detection via reconstruction error.
Trained only on normal traffic.

Architecture: Input(20) → 64 → 32 → 16 → 32 → 64 → Output(20)
Anomaly score = per-sample MSE reconstruction error.
Threshold = mean + N*std of training errors (learned, not hardcoded).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import mlflow


# ── Constants ─────────────────────────────────────────────────────────────────

INPUT_DIM    = 25
HIDDEN_DIMS  = [64, 32, 16]
THRESHOLD_STD = 2.5   # anomaly threshold = mean + N*std of train errors

TRAIN_PARAMS = {
    "epochs":       60,
    "batch_size":   256,
    "lr":           1e-3,
    "weight_decay": 1e-5,
    "patience":     8,
    "dropout":      0.1,
}


# ── Network ───────────────────────────────────────────────────────────────────

class ShallowAE(nn.Module):
    """
    Symmetric autoencoder.
    Encoder: 20 → 64 → 32 → 16
    Decoder: 16 → 32 → 64 → 20
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, dropout=0.1):
        super().__init__()

        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*enc)

        dec = []
        for h in reversed(hidden_dims[:-1]):
            dec += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        """Per-sample MSE. Shape: (N,)"""
        with torch.no_grad():
            recon = self.forward(x)
            return torch.mean((x - recon) ** 2, dim=1)


# ── Model wrapper ─────────────────────────────────────────────────────────────

class ShallowAutoencoderModel:
    """
    Wrapper exposing the standard Layer 2A interface:
        .train(X_normal, X_val)
        .anomaly_scores(X)  -> np.ndarray
        .predict(X)         -> np.ndarray (1=anomaly, 0=normal)
        .export_onnx(path)
    """

    def __init__(self):
        self.net       = None
        self.threshold = None
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X_normal: np.ndarray, X_val: np.ndarray,
              run_name: str = "shallow_autoencoder") -> None:
        """
        Train on normal-only data. X arrays should already be normalised.
        Early stopping on val reconstruction loss.
        Threshold set automatically after training.
        """
        device = self.device
        print(f"[AE] Training on {device}")

        X_tr = torch.from_numpy(X_normal.astype(np.float32))
        X_v  = torch.from_numpy(X_val.astype(np.float32))
        tr_dl = DataLoader(TensorDataset(X_tr), batch_size=TRAIN_PARAMS["batch_size"],
                           shuffle=True, drop_last=True)
        v_dl  = DataLoader(TensorDataset(X_v),  batch_size=512)

        net = ShallowAE(dropout=TRAIN_PARAMS["dropout"]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=TRAIN_PARAMS["lr"],
                               weight_decay=TRAIN_PARAMS["weight_decay"])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
        crit = nn.MSELoss()

        best_loss, patience_ctr, best_state = float("inf"), 0, None

        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_params(TRAIN_PARAMS)

            for epoch in range(1, TRAIN_PARAMS["epochs"] + 1):
                net.train()
                tr_loss = 0.0
                for (xb,) in tr_dl:
                    xb = xb.to(device)
                    opt.zero_grad()
                    loss = crit(net(xb), xb)
                    loss.backward()
                    opt.step()
                    tr_loss += loss.item()
                tr_loss /= len(tr_dl)

                net.eval()
                v_loss = 0.0
                with torch.no_grad():
                    for (xb,) in v_dl:
                        v_loss += crit(net(xb.to(device)), xb.to(device)).item()
                v_loss /= len(v_dl)
                sch.step(v_loss)

                mlflow.log_metrics({"train_loss": tr_loss, "val_loss": v_loss}, step=epoch)

                if epoch % 10 == 0:
                    print(f"  epoch {epoch:3d} | train={tr_loss:.5f} | val={v_loss:.5f}")

                if v_loss < best_loss:
                    best_loss    = v_loss
                    patience_ctr = 0
                    best_state   = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    patience_ctr += 1
                    if patience_ctr >= TRAIN_PARAMS["patience"]:
                        print(f"[AE] Early stopping at epoch {epoch}")
                        break

            net.load_state_dict(best_state)
            self.net = net

            # compute threshold from training reconstruction errors
            net.eval()
            errs = []
            with torch.no_grad():
                for (xb,) in DataLoader(TensorDataset(X_tr), batch_size=512):
                    errs.append(net.reconstruction_error(xb.to(device)).cpu().numpy())
            errs = np.concatenate(errs)
            self.threshold = float(errs.mean() + THRESHOLD_STD * errs.std())

            mlflow.log_metrics({
                "recon_mean": errs.mean(),
                "recon_std":  errs.std(),
                "threshold":  self.threshold,
            })

        print(f"[AE] Threshold = {self.threshold:.6f}  "
              f"(mean={errs.mean():.5f} + {THRESHOLD_STD}*std={errs.std():.5f})")

    # ── Inference ─────────────────────────────────────────────────────────────

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Reconstruction error per sample (higher = more anomalous)."""
        self.net.eval()
        t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            return self.net.reconstruction_error(t).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1=anomaly, 0=normal."""
        return (self.anomaly_scores(X) >= self.threshold).astype(int)

    def predict_single(self, x: np.ndarray) -> tuple:
        """Returns (is_anomaly: bool, score: float). x shape (1, 20)."""
        score   = float(self.anomaly_scores(x)[0])
        is_anom = score >= (self.threshold or 0.0)
        return is_anom, score

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_weights(self, path: str) -> None:
        torch.save({
            "state_dict": self.net.state_dict(),
            "threshold":  self.threshold,
        }, path)
        print(f"[AE] Weights saved → {path}")

    def load_weights(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device)
        self.net = ShallowAE().to(self.device)
        self.net.load_state_dict(ck["state_dict"])
        self.threshold = ck["threshold"]
        self.net.eval()
        print(f"[AE] Weights loaded ← {path}")

    # ── ONNX export ───────────────────────────────────────────────────────────

    def export_onnx(self, output_path: str) -> None:
        self.net.eval().cpu()
        dummy = torch.randn(1, INPUT_DIM)

        torch.onnx.export(
            self.net, dummy, output_path,
            input_names=["features"],
            output_names=["reconstruction"],
            dynamic_axes={
                "features":       {0: "batch"},
                "reconstruction": {0: "batch"},
            },
            opset_version=17,
        )

        # save threshold
        thr_path = output_path.replace(".onnx", "_threshold.txt")
        with open(thr_path, "w") as f:
            f.write(str(self.threshold or 0.0))

        # validate
        import onnxruntime as ort, time
        sess  = ort.InferenceSession(output_path)
        dummy_np = dummy.numpy()
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            sess.run(None, {"features": dummy_np})
            times.append((time.perf_counter() - t0) * 1000)
        avg_ms = np.mean(times)

        print(f"[AE] ONNX exported → {output_path}")
        print(f"[AE] Avg inference: {avg_ms:.3f}ms  |  "
              f"{'PASS' if avg_ms < 2.0 else 'WARN >2ms'}")

        self.net.to(self.device)  # restore device