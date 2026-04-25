"""
ml/layer2b/candidates/gru.py

Layer 2B Candidate 3 — Bidirectional GRU with Bahdanau Attention
Input: character token sequences (N, MAX_LEN) int32 from CharTokenizer

Architecture:
    Embedding → BiGRU (2 layers) → Bahdanau Attention → Linear → Softmax

The attention layer produces per-character weights — used for
the dashboard's "why was this blocked" explainability feature.

This mirrors the GRU approach in Base Paper 2 (Babaey & Faragardi 2025).
Your standalone result can be directly compared to their ensemble.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import mlflow

from feature_engineering.tokenizer import VOCAB_SIZE, CharTokenizer

CLASS_NAMES = ["normal", "sqli", "xss", "lfi", "other_attack", "cmdi"]
NUM_CLASSES = len(CLASS_NAMES)
MAX_LEN     = 256  # reduced from 512; matches runtime tokenizer max_len

TRAIN_PARAMS = {
    "epochs":       45,
    "batch_size":   128,
    "lr":           3e-4,
    "weight_decay": 1e-5,
    "patience":     7,
    "embed_dim":    64,
    "hidden_dim":   128,
    "num_layers":   2,
    "dropout":      0.3,
    "grad_clip":    1.0,
}


# ── Network ───────────────────────────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, gru_out: torch.Tensor):
        """
        gru_out: (B, L, H)
        Returns context (B, H) and weights (B, L)
        """
        scores  = self.w(gru_out).squeeze(-1)          # (B, L)
        weights = F.softmax(scores, dim=1)              # (B, L)
        context = torch.bmm(weights.unsqueeze(1), gru_out).squeeze(1)  # (B, H)
        return context, weights


class GRUClassifier(nn.Module):

    def __init__(self, vocab_size=VOCAB_SIZE + 1, embed_dim=64,
                 hidden_dim=128, num_layers=2, num_classes=NUM_CLASSES,
                 dropout=0.3):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop  = nn.Dropout(dropout)
        self.gru         = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention   = BahdanauAttention(hidden_dim * 2)
        self.classifier  = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        emb     = self.embed_drop(self.embedding(x))    # (B, L, E)
        gru_out, _ = self.gru(emb)                      # (B, L, H*2)
        ctx, attn  = self.attention(gru_out)             # (B, H*2), (B, L)
        logits     = self.classifier(ctx)
        if return_attn:
            return logits, attn
        return logits


# ── Model wrapper ─────────────────────────────────────────────────────────────

class GRUModel:

    def __init__(self):
        self.net    = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              run_name: str = "gru_l2b") -> None:
        device = self.device
        print(f"[GRU] Training on {device}")

        tr_dl = DataLoader(
            TensorDataset(torch.from_numpy(X_train).long(),
                          torch.from_numpy(y_train).long()),
            batch_size=TRAIN_PARAMS["batch_size"], shuffle=True,
            num_workers=2, pin_memory=(device == "cuda"),
        )
        v_dl = DataLoader(
            TensorDataset(torch.from_numpy(X_val).long(),
                          torch.from_numpy(y_val).long()),
            batch_size=TRAIN_PARAMS["batch_size"], num_workers=2,
        )

        net = GRUClassifier(
            embed_dim=TRAIN_PARAMS["embed_dim"],
            hidden_dim=TRAIN_PARAMS["hidden_dim"],
            num_layers=TRAIN_PARAMS["num_layers"],
            dropout=TRAIN_PARAMS["dropout"],
        ).to(device)

        weights = _class_weights(y_train, NUM_CLASSES, device)
        crit    = nn.CrossEntropyLoss(weight=weights)
        opt     = torch.optim.AdamW(net.parameters(),
                                    lr=TRAIN_PARAMS["lr"],
                                    weight_decay=TRAIN_PARAMS["weight_decay"])
        sch     = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=TRAIN_PARAMS["lr"] * 10,
            epochs=TRAIN_PARAMS["epochs"], steps_per_epoch=len(tr_dl),
        )

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
                    nn.utils.clip_grad_norm_(net.parameters(), TRAIN_PARAMS["grad_clip"])
                    opt.step()
                    sch.step()
                    tr_loss += loss.item()
                tr_loss /= len(tr_dl)

                val_f1, val_acc = _validate(net, v_dl, device)
                mlflow.log_metrics({
                    "train_loss": tr_loss,
                    "val_f1": val_f1, "val_acc": val_acc,
                    "lr": opt.param_groups[0]["lr"],
                }, step=epoch)

                if epoch % 5 == 0:
                    print(f"  epoch {epoch:3d} | loss={tr_loss:.4f} "
                          f"| val_f1={val_f1:.4f} | val_acc={val_acc:.4f}")

                if val_f1 > best_f1:
                    best_f1, patience_ctr = val_f1, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    patience_ctr += 1
                    if patience_ctr >= TRAIN_PARAMS["patience"]:
                        print(f"[GRU] Early stopping at epoch {epoch}")
                        break

            net.load_state_dict(best_state)
            mlflow.log_metric("best_val_f1", best_f1)

        self.net = net
        print(f"[GRU] Best val F1: {best_f1:.4f}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        preds = []
        with torch.no_grad():
            for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X).long()),
                                    batch_size=256):
                preds.append(self.net(xb.to(self.device)).argmax(1).cpu().numpy())
        return np.concatenate(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        probs = []
        with torch.no_grad():
            for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X).long()),
                                    batch_size=256):
                probs.append(
                    F.softmax(self.net(xb.to(self.device)), dim=1).cpu().numpy()
                )
        return np.concatenate(probs)

    def predict_single(self, token_ids: np.ndarray) -> dict:
        """token_ids shape (1, MAX_LEN). Returns label, confidence, proba."""
        proba    = self.predict_proba(token_ids)[0]
        pred_idx = int(np.argmax(proba))
        return {
            "label":      CLASS_NAMES[pred_idx],
            "confidence": round(float(proba[pred_idx]), 4),
            "proba":      {c: round(float(p), 4)
                           for c, p in zip(CLASS_NAMES, proba)},
        }

    # ── Explainability ────────────────────────────────────────────────────────

    def attention_map(self, token_ids: np.ndarray,
                      tokenizer: CharTokenizer) -> list:
        """
        Return (character, weight) pairs for a single request.
        Use this in the dashboard to highlight suspicious characters.

        Parameters
        ----------
        token_ids : (1, MAX_LEN) int32
        tokenizer : CharTokenizer instance

        Returns
        -------
        list of (char: str, weight: float) sorted by position
        """
        self.net.eval()
        x = torch.from_numpy(token_ids).long().to(self.device)
        with torch.no_grad():
            _, weights = self.net(x, return_attn=True)
        weights = weights[0].cpu().numpy()
        text    = tokenizer.decode(token_ids[0])
        return [(ch, round(float(w), 5))
                for ch, w in zip(text, weights[:len(text)])]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_weights(self, path: str) -> None:
        torch.save({"state_dict": self.net.state_dict(),
                    "params": TRAIN_PARAMS}, path)
        print(f"[GRU] Saved → {path}")

    def load_weights(self, path: str) -> None:
        ck      = torch.load(path, map_location=self.device)
        self.net = GRUClassifier().to(self.device)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        print(f"[GRU] Loaded ← {path}")

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
        print(f"[GRU] ONNX exported -> {output_path}")
        print(f"[GRU] avg={avg_ms}ms  p99={p99_ms}ms  {status}")
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