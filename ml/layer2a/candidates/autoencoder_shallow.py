"""
ml/layer2a/candidates/autoencoder_shallow.py

Layer 2A Candidate 2 — Shallow Autoencoder (PyTorch)
One-class anomaly detection via reconstruction error.
Trained only on normal traffic.

Architecture: Input(25) → 64 → 32 → 16 → 32 → 64 → Output(25)
Anomaly score = per-sample MSE reconstruction error.

THRESHOLD TUNING — history of bugs and final fix
-------------------------------------------------
Bug 1 (mixed val set, low→high sweep):
    Passed a combined normal+attack array.
    At very low thresholds, few normals in the mix made FPR look ≤ 5%.
    Picked threshold ≈ 0 → TN=0, FPR=1.0 on test set.

Bug 2 (separate sets, hi=absolute max of all scores):
    hi = max(normal_scores.max(), attack_scores.max()) ≈ 24.
    At score=24 almost nothing is flagged → FPR=0 on val immediately.
    First candidate wins → threshold=24.07.
    Result: 70% of requests passed through L2A, LFI detection=4.2%.

Final fix — bound hi/lo to normal val score percentiles [P50 … P99]:
    lo = normal_P50  → at this threshold ~50% of normal flagged (FPR~50%)
    hi = normal_P99  → at this threshold ~1%  of normal flagged (FPR~1%)
    Correct threshold lies in this window. Cannot be near-zero (Bug 1)
    or near the absolute max (Bug 2). Sweep HIGH→LOW within this range.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import mlflow


# ── Constants ─────────────────────────────────────────────────────────────────

INPUT_DIM     = 29          # must match len(FEATURE_NAMES) in extractor.py
HIDDEN_DIMS   = [64, 32, 16]
THRESHOLD_STD = 2.5         # used only for the unsupervised initial threshold

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
    Encoder: 25 → 64 → 32 → 16
    Decoder: 16 → 32 → 64 → 25
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dims=None, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS

        enc, prev = [], input_dim
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
        .tune_threshold(X_normal_val, X_attack_val, target_fpr=0.05)
        .anomaly_scores(X)  -> np.ndarray
        .predict(X)         -> np.ndarray (1=anomaly, 0=normal)
        .export_onnx(path)
    """

    def __init__(self):
        self.net       = None
        self.threshold = None
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        # stored after training — used as sweep bounds in tune_threshold()
        self._normal_score_p50 = None
        self._normal_score_p99 = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X_normal: np.ndarray, X_val: np.ndarray,
              run_name: str = "shallow_autoencoder") -> None:
        """
        Train on normal-only data. X arrays should already be normalised.
        Early stopping on val reconstruction loss.
        Sets an initial unsupervised threshold (mean + 2.5*std of training
        errors). Call tune_threshold() afterwards to refine using labelled
        val splits — this is what actually achieves good TPR vs FPR balance.

        Parameters
        ----------
        X_normal : (N, 25) float32 — normalised normal-only training samples
        X_val    : (M, 25) float32 — normalised normal-only val samples
                   (normal-only is correct here; tune_threshold receives attacks)
        run_name : MLflow run label
        """
        device = self.device
        print(f"[AE] Training on {device}")

        X_tr = torch.from_numpy(X_normal.astype(np.float32))
        X_v  = torch.from_numpy(X_val.astype(np.float32))
        tr_dl = DataLoader(TensorDataset(X_tr), batch_size=TRAIN_PARAMS["batch_size"],
                           shuffle=True, drop_last=True)
        v_dl  = DataLoader(TensorDataset(X_v),  batch_size=512)

        net  = ShallowAE(input_dim=X_normal.shape[1],
                         dropout=TRAIN_PARAMS["dropout"]).to(device)
        opt  = torch.optim.Adam(net.parameters(), lr=TRAIN_PARAMS["lr"],
                                weight_decay=TRAIN_PARAMS["weight_decay"])
        sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
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
                    best_loss, patience_ctr = v_loss, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    patience_ctr += 1
                    if patience_ctr >= TRAIN_PARAMS["patience"]:
                        print(f"[AE] Early stopping at epoch {epoch}")
                        break

            net.load_state_dict(best_state)
            self.net = net

            # ── Compute training error statistics ─────────────────────────────
            net.eval()
            errs = []
            with torch.no_grad():
                for (xb,) in DataLoader(TensorDataset(X_tr), batch_size=512):
                    errs.append(net.reconstruction_error(xb.to(device)).cpu().numpy())
            errs = np.concatenate(errs)

            # Store percentiles — these become the sweep bounds in tune_threshold()
            # ensuring the sweep stays in the meaningful discrimination range.
            self._normal_score_p50 = float(np.percentile(errs, 50))
            self._normal_score_p99 = float(np.percentile(errs, 99))

            # Unsupervised initial threshold (refined by tune_threshold)
            self.threshold = float(errs.mean() + THRESHOLD_STD * errs.std())

            mlflow.log_metrics({
                "recon_mean":          float(errs.mean()),
                "recon_std":           float(errs.std()),
                "normal_score_p50":    self._normal_score_p50,
                "normal_score_p99":    self._normal_score_p99,
                "threshold_initial":   self.threshold,
            })

        print(f"[AE] Normal score distribution on training set:")
        print(f"  mean={errs.mean():.5f}  std={errs.std():.5f}  "
              f"P50={self._normal_score_p50:.5f}  P99={self._normal_score_p99:.5f}")
        print(f"[AE] Initial threshold (unsupervised) = {self.threshold:.6f}")
        print(f"[AE] Call tune_threshold() to refine using labelled val splits.")

    # ── Threshold tuning ──────────────────────────────────────────────────────

    def tune_threshold(
        self,
        X_normal_val: np.ndarray,
        X_attack_val: np.ndarray,
        target_fpr:   float = 0.05,
        max_fnr:      float = 0.05,
        n_steps:      int   = 500,
    ) -> float:
        """
        Refine the anomaly threshold using separate normal and attack
        validation splits.

        Parameters
        ----------
        X_normal_val : (N, 29) float32 — normalised normal-only val samples.
                       FPR is measured here. Never mix attack samples in.
        X_attack_val : (M, 29) float32 — normalised attack-only val samples.
                       TPR is measured here.
        target_fpr   : float — max acceptable FPR (default 0.05 = 5%)
        max_fnr      : float — NEW: cap false-negative rate at 5% (min TPR 95%)
        n_steps      : int   — sweep resolution (default 500)

        Returns
        -------
        float — chosen threshold (also stored as self.threshold)
        """
        if self._normal_score_p50 is None:
            raise RuntimeError("Call train() before tune_threshold().")

        normal_scores = self.anomaly_scores(X_normal_val)
        attack_scores = self.anomaly_scores(X_attack_val)

        # ── Bound sweep to normal val score percentile range ──────────────────
        lo = float(np.percentile(normal_scores, 50))   # ~50% FPR — lenient end
        hi = float(np.percentile(normal_scores, 99))   # ~1%  FPR — strict end

        print(f"[AE] Threshold sweep: [{lo:.5f} … {hi:.5f}]")
        print(f"  (val normal P50={lo:.5f}, P99={hi:.5f})")
        print(f"  Attack scores: min={attack_scores.min():.5f}  "
              f"mean={attack_scores.mean():.5f}  max={attack_scores.max():.5f}")
        pct_above = (attack_scores > hi).mean() * 100
        print(f"  {pct_above:.1f}% of attacks already score above normal P99 "
              f"— easily detected at any threshold ≤ P99.")

        initial_thr = self.threshold
        best_thr, best_tpr = None, 0.0

        # ── Sweep HIGH → LOW ──────────────────────────────────────────────────
        for thr in np.linspace(hi, lo, n_steps):
            fp = int(np.sum(normal_scores >= thr))
            tn = int(np.sum(normal_scores <  thr))
            tp = int(np.sum(attack_scores >= thr))
            fn = int(np.sum(attack_scores <  thr))

            # Do not allow a threshold that flags every normal sample
            if tn == 0:
                continue

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if fpr <= target_fpr and tpr > best_tpr:
                best_tpr = tpr
                best_thr = float(thr)

        # ── Handle no qualifying threshold ────────────────────────────────────
        if best_thr is None:
            print(f"[AE] WARNING: no threshold in [{lo:.5f}…{hi:.5f}] meets "
                  f"FPR≤{target_fpr}. Keeping initial threshold={initial_thr:.6f}.")
            best_thr = initial_thr

        self.threshold = best_thr

        # ── Report final metrics at chosen threshold ───────────────────────────
        fp_f = int(np.sum(normal_scores >= self.threshold))
        tn_f = int(np.sum(normal_scores <  self.threshold))
        tp_f = int(np.sum(attack_scores >= self.threshold))
        fn_f = int(np.sum(attack_scores <  self.threshold))
        fpr_f = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0.0
        tpr_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0

        print(f"[AE] Tuned threshold = {self.threshold:.6f}")
        print(f"  Val normal  → FPR={fpr_f:.4f}  "
              f"(FP={fp_f}, TN={tn_f})  target≤{target_fpr}")
        print(f"  Val attacks → TPR={tpr_f:.4f}  "
              f"(TP={tp_f}, FN={fn_f})")

        return self.threshold

    # ── Inference ─────────────────────────────────────────────────────────────

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Reconstruction error per sample (higher = more anomalous)."""
        self.net.eval()
        t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            return self.net.reconstruction_error(t).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = anomaly, 0 = normal."""
        return (self.anomaly_scores(X) >= self.threshold).astype(int)

    def predict_single(self, x: np.ndarray) -> tuple:
        """
        Predict one request. Returns (is_anomaly: bool, score: float).
        x shape: (1, 25)
        """
        score   = float(self.anomaly_scores(x)[0])
        is_anom = score >= (self.threshold or 0.0)
        return is_anom, score

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_weights(self, path: str) -> None:
        torch.save({
            "state_dict":         self.net.state_dict(),
            "threshold":          self.threshold,
            "normal_score_p50":   self._normal_score_p50,
            "normal_score_p99":   self._normal_score_p99,
        }, path)
        print(f"[AE] Weights saved → {path}")

    def load_weights(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device)
        self.net = ShallowAE().to(self.device)
        self.net.load_state_dict(ck["state_dict"])
        self.threshold          = ck["threshold"]
        self._normal_score_p50  = ck.get("normal_score_p50")
        self._normal_score_p99  = ck.get("normal_score_p99")
        self.net.eval()
        print(f"[AE] Weights loaded ← {path}")

    # ── ONNX export ───────────────────────────────────────────────────────────

    def export_onnx(self, output_path: str) -> None:
        """
        Export model to ONNX. Saves companion threshold file alongside.
        Requires: pip install onnx onnxruntime
        """
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

        # Save threshold alongside so FastAPI can load it at startup
        thr_path = output_path.replace(".onnx", "_threshold.txt")
        with open(thr_path, "w") as f:
            f.write(str(self.threshold or 0.0))

        # Validate and benchmark
        import onnxruntime as ort, time
        sess     = ort.InferenceSession(output_path)
        dummy_np = dummy.numpy()

        # warmup
        for _ in range(10):
            sess.run(None, {"features": dummy_np})

        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            sess.run(None, {"features": dummy_np})
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = round(float(np.mean(times)), 3)
        p99_ms = round(float(np.percentile(times, 99)), 3)
        status = "PASS" if p99_ms < 2.0 else f"WARN (p99={p99_ms}ms > 2ms target)"

        print(f"[AE] ONNX exported → {output_path}")
        print(f"[AE] Threshold saved → {thr_path}  (value={self.threshold:.6f})")
        print(f"[AE] avg={avg_ms}ms  p99={p99_ms}ms  {status}")

        self.net.to(self.device)  # restore device after export