"""
ml/layer2b/candidates/cnn_1d.py

Layer 2B Candidate 2 — 1D CNN multi-class attack classifier
Input: character token sequences (N, MAX_LEN) int32 from CharTokenizer

Architecture:
    Embedding(VOCAB) → Conv1D [kernel 3] + Conv1D [kernel 5] + Conv1D [kernel 7]
    → GlobalMaxPool per branch → Concat → Dropout → Linear → Softmax

Multi-kernel design captures patterns at different n-gram widths:
  kernel=3 catches short patterns (OR, AND, --)
  kernel=5 catches medium patterns (UNION, alert, ../../../)
  kernel=7 catches longer sequences (<script>, SELECT *)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import mlflow

from feature_engineering.tokenizer import VOCAB_SIZE

CLASS_NAMES  = ["normal", "sqli", "xss", "lfi", "other_attack", "cmdi"]
NUM_CLASSES  = len(CLASS_NAMES)
MAX_LEN      = 256  # reduced from 512; matches runtime tokenizer max_len

TRAIN_PARAMS = {
    "epochs":       35,
    "batch_size":   256,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "patience":     6,
    "embed_dim":    64,
    "num_filters":  128,
    "kernel_sizes": [3, 5, 7],
    "dropout":      0.3,
}


# ── Network ───────────────────────────────────────────────────────────────────

class CNN1D(nn.Module):

    def __init__(self, vocab_size=VOCAB_SIZE + 1, embed_dim=64,
                 num_filters=128, kernel_sizes=None, num_classes=NUM_CLASSES,
                 dropout=0.3):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])

        total = num_filters * len(kernel_sizes)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).permute(0, 2, 1)   # (B, E, L)
        pooled = [
            F.adaptive_max_pool1d(F.relu(conv(emb)), 1).squeeze(2)
            for conv in self.convs
        ]
        return self.classifier(torch.cat(pooled, dim=1))


# ── Model wrapper ─────────────────────────────────────────────────────────────

class CNN1DModel:

    def __init__(self):
        self.net    = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              run_name: str = "cnn_1d_l2b") -> None:
        device = self.device
        print(f"[CNN1D] Training on {device}")

        tr_dl = DataLoader(
            TensorDataset(torch.from_numpy(X_train).long(),
                          torch.from_numpy(y_train).long()),
            batch_size=TRAIN_PARAMS["batch_size"], shuffle=True, drop_last=True,
        )
        v_dl = DataLoader(
            TensorDataset(torch.from_numpy(X_val).long(),
                          torch.from_numpy(y_val).long()),
            batch_size=TRAIN_PARAMS["batch_size"],
        )

        net = CNN1D(
            embed_dim=TRAIN_PARAMS["embed_dim"],
            num_filters=TRAIN_PARAMS["num_filters"],
            kernel_sizes=TRAIN_PARAMS["kernel_sizes"],
            dropout=TRAIN_PARAMS["dropout"],
        ).to(device)

        weights = _class_weights(y_train, NUM_CLASSES, device)
        crit    = nn.CrossEntropyLoss(weight=weights)
        opt     = torch.optim.Adam(net.parameters(),
                                   lr=TRAIN_PARAMS["lr"],
                                   weight_decay=TRAIN_PARAMS["weight_decay"])
        sch     = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=TRAIN_PARAMS["epochs"])

        best_f1, patience_ctr, best_state = 0.0, 0, None

        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_params(TRAIN_PARAMS)

            for epoch in range(1, TRAIN_PARAMS["epochs"] + 1):
                net.train()
                tr_loss = 0.0
                for xb, yb in tr_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = crit(net(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    tr_loss += loss.item()
                tr_loss /= len(tr_dl)
                sch.step()

                val_f1, val_acc = _validate(net, v_dl, device)
                mlflow.log_metrics({"train_loss": tr_loss,
                                    "val_f1": val_f1, "val_acc": val_acc},
                                   step=epoch)

                if epoch % 5 == 0:
                    print(f"  epoch {epoch:3d} | loss={tr_loss:.4f} "
                          f"| val_f1={val_f1:.4f} | val_acc={val_acc:.4f}")

                if val_f1 > best_f1:
                    best_f1, patience_ctr = val_f1, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    patience_ctr += 1
                    if patience_ctr >= TRAIN_PARAMS["patience"]:
                        print(f"[CNN1D] Early stopping at epoch {epoch}")
                        break

            net.load_state_dict(best_state)
            mlflow.log_metric("best_val_f1", best_f1)

        self.net = net
        print(f"[CNN1D] Best val F1: {best_f1:.4f}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        dl = DataLoader(TensorDataset(torch.from_numpy(X).long()),
                        batch_size=512)
        preds = []
        with torch.no_grad():
            for (xb,) in dl:
                preds.append(self.net(xb.to(self.device)).argmax(1).cpu().numpy())
        return np.concatenate(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        dl = DataLoader(TensorDataset(torch.from_numpy(X).long()),
                        batch_size=512)
        probs = []
        with torch.no_grad():
            for (xb,) in dl:
                probs.append(
                    F.softmax(self.net(xb.to(self.device)), dim=1).cpu().numpy()
                )
        return np.concatenate(probs)

    def predict_single(self, token_ids: np.ndarray) -> dict:
        """token_ids shape (1, MAX_LEN)"""
        proba    = self.predict_proba(token_ids)[0]
        pred_idx = int(np.argmax(proba))
        return {
            "label":      CLASS_NAMES[pred_idx],
            "confidence": round(float(proba[pred_idx]), 4),
            "proba":      {c: round(float(p), 4)
                           for c, p in zip(CLASS_NAMES, proba)},
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_weights(self, path: str) -> None:
        torch.save({"state_dict": self.net.state_dict(),
                    "params": TRAIN_PARAMS}, path)
        print(f"[CNN1D] Saved → {path}")

    def load_weights(self, path: str) -> None:
        ck      = torch.load(path, map_location=self.device)
        self.net = CNN1D().to(self.device)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        print(f"[CNN1D] Loaded ← {path}")

    # ── ONNX export ───────────────────────────────────────────────────────────

    def export_onnx(self, output_path: str) -> dict:
        """Export to ONNX and return latency benchmark dict for pick_best()."""
        self.net.eval().cpu()
        dummy = torch.zeros(1, MAX_LEN, dtype=torch.long)
        torch.onnx.export(
            self.net, dummy, output_path,
            input_names=["token_ids"], output_names=["logits"],
            dynamic_axes={
                "token_ids": {0: "batch", 1: "seq_len"},  # both dims dynamic
                "logits":    {0: "batch"},
            },
            opset_version=17,
        )
        self.net.to(self.device)

        import onnxruntime as ort, time
        sess = ort.InferenceSession(output_path)
        dummy_np = dummy.numpy()
        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            sess.run(None, {"token_ids": dummy_np})
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = round(np.mean(times), 3)
        p99_ms = round(np.percentile(times, 99), 3)
        status = "PASS" if p99_ms < 20 else "WARN >20ms (latency gate may disqualify)"
        print(f"[CNN1D] ONNX exported -> {output_path}")
        print(f"[CNN1D] avg={avg_ms}ms  p99={p99_ms}ms  {status}")
        return {"avg_ms": avg_ms, "p99_ms": p99_ms}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _class_weights(y: np.ndarray, n_classes: int, device: str) -> torch.Tensor:
    counts  = np.bincount(y, minlength=n_classes).astype(float)
    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)


def _validate(net, dl, device) -> tuple:
    from sklearn.metrics import f1_score, accuracy_score
    net.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in dl:
            preds.extend(net(xb.to(device)).argmax(1).cpu().numpy())
            labels.extend(yb.numpy())
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return float(f1), float(acc)