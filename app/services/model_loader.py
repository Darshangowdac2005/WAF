"""
app/services/model_loader.py

Central model loading orchestrator.
Called by main.py lifespan on startup and by /api/models/reload.

Provides load_all() which loads L2A and L2B in order,
validates both sessions are live with a dummy inference pass,
and raises ModelNotLoadedError on any failure so the
app refuses to start rather than silently running unprotected.
"""
import numpy as np
from app.core.logging import logger
from app.core.exceptions import ModelNotLoadedError
import app.services.layer2a_anomaly as l2a
import app.services.layer2b_deep as l2b


def load_all() -> dict:
    """
    Load both ONNX models and validate them with dummy inference.

    Returns
    -------
    dict with keys l2a_threshold, l2b_input, l2b_uses_tokens
    """
    errors = []

    # ── Layer 2A ──────────────────────────────────────────────────────────────
    try:
        l2a.load()
        # Validate: run a dummy (1, 29) float32 vector through the session
        dummy_fvec = np.zeros((1, 29), dtype=np.float32)
        _, score = l2a.infer(dummy_fvec)
        logger.info("L2A validation passed (dummy score=%.5f, threshold=%.5f)",
                    score, l2a._threshold)
    except Exception as e:
        errors.append(f"L2A load/validate failed: {e}")
        logger.error("L2A load failed: %s", e)

    # ── Layer 2B ──────────────────────────────────────────────────────────────
    try:
        l2b.load()
        # Validate: run dummy inputs matching the winner model type
        dummy_fvec   = np.zeros((1, 29), dtype=np.float32)
        dummy_tokens = np.zeros((1, 256), dtype=np.int64)
        label, conf, _ = l2b.infer(dummy_fvec, dummy_tokens)
        logger.info("L2B validation passed (dummy → label=%s conf=%.4f)", label, conf)
    except Exception as e:
        errors.append(f"L2B load/validate failed: {e}")
        logger.error("L2B load failed: %s", e)

    if errors:
        raise ModelNotLoadedError(
            "One or more models failed to load:\n" + "\n".join(errors)
        )

    return {
        "l2a_threshold":   l2a._threshold,
        "l2b_input":       l2b._in_name,
        "l2b_uses_tokens": l2b._uses_tokens,
    }