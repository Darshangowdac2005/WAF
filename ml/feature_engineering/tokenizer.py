"""
ml/feature_engineering/tokenizer.py

Character-level tokeniser for Layer 2B sequence models (1D CNN, GRU).
XGBoost uses extractor.py numeric vectors instead — NOT this file.

Design
------
- Vocabulary: printable ASCII characters that appear in HTTP traffic
- Index 0  : PAD  (padding token, masked in attention / ignored in pool)
- Index 1–N: real characters
- Index N+1: UNK  (characters outside vocab)

Stateless — safe to instantiate once at app startup and reuse.

Used in
-------
  Training  : ml/layer2b/candidates/cnn_1d.py, gru.py
  Inference : app/services/layer2b_deep.py
"""

import numpy as np
from typing import Union

# ── Vocabulary ────────────────────────────────────────────────────────────────
# Covers every character commonly seen in HTTP request URLs and bodies.
# Order must never change after training.

_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,;:!?@#$%^&*()-_=+[]{}|/<>\\\"'`~\n\t\r"
)

PAD_IDX   = 0
_CHAR2IDX = {c: i + 1 for i, c in enumerate(_CHARS)}   # 1-indexed
VOCAB_SIZE = len(_CHARS) + 1                            # +1 for PAD
UNK_IDX   = VOCAB_SIZE                                  # out-of-vocab → end


class CharTokenizer:
    """
    Stateless character-level tokeniser.

    Parameters
    ----------
    max_len : int
        Sequence length. Strings longer than max_len are truncated from
        the RIGHT (we keep the start — method + URL start — which contains
        the most attack signal). Shorter strings are right-padded with PAD_IDX.
    """

    def __init__(self, max_len: int = 512):
        self.max_len    = max_len
        self.vocab_size = VOCAB_SIZE
        self._c2i       = _CHAR2IDX

    # ── single sample ─────────────────────────────────────────────────────────

    def encode(self, text: str, pad: bool = True) -> np.ndarray:
        """
        Encode a single string → int32 array of shape (max_len,).

        Characters not in vocabulary map to UNK_IDX.
        """
        indices = [self._c2i.get(c, UNK_IDX) for c in text[: self.max_len]]
        if pad:
            indices += [PAD_IDX] * max(0, self.max_len - len(indices))
        return np.array(indices, dtype=np.int32)

    def encode_request(self, request: dict) -> np.ndarray:
        """
        Build canonical request string and encode it.

        Format: "METHOD URL BODY"
        The space separators act as natural delimiters between sections.
        """
        text = (
            request.get("method", "GET") + " "
            + request.get("url", "") + " "
            + request.get("body", "")
        )
        return self.encode(text, pad=True)

    # ── batch ─────────────────────────────────────────────────────────────────

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode list of strings → (N, max_len) int32 array."""
        return np.stack([self.encode(t) for t in texts])

    def encode_requests(self, requests: list[dict]) -> np.ndarray:
        """Encode list of request dicts → (N, max_len) int32 array."""
        texts = [
            r.get("method", "GET") + " "
            + r.get("url", "") + " "
            + r.get("body", "")
            for r in requests
        ]
        return self.encode_batch(texts)

    # ── decode (for debugging / attention visualisation) ──────────────────────

    def decode(self, indices: Union[list, np.ndarray]) -> str:
        """Decode int indices back to string, stopping at first PAD."""
        idx2char = {v: k for k, v in self._c2i.items()}
        chars = []
        for i in indices:
            i = int(i)
            if i == PAD_IDX:
                break
            chars.append(idx2char.get(i, "?"))
        return "".join(chars)

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size}, max_len={self.max_len})"