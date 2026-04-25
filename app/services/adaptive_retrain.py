"""app/services/retrain.py — adaptive retraining pipeline"""
import asyncio
from datetime import datetime
from app.core.config import settings
from app.core.logging import logger
from app.db.collections import feedback_queue, retrain_log

POISON_MAX_RATE_PER_IP = 20  # max verified samples from one IP

async def run_retrain_cycle() -> dict:
    """
    Anti-poisoning safeguards:
    1. Only verified (human-reviewed) samples are used.
    2. Samples flagged as poisoning_flag=True are excluded.
    3. No single IP contributes more than POISON_MAX_RATE_PER_IP samples.
    4. Each sample is re-scored by Layer 1 — if L1 catches it, discard.
    5. Minimum sample count required before triggering.

    Returns dict with run metadata.
    """
    # Fetch verified non-poisoned feedback
    cursor = feedback_queue().find(
        {"verified_label": {"$ne": None}, "poisoning_flag": False},
        {"_id": 0}
    )
    samples = await cursor.to_list(length=10000)

    if len(samples) < settings.RETRAIN_MIN_SAMPLES:
        logger.info("Retrain skipped: only %d verified samples (min=%d)",
                    len(samples), settings.RETRAIN_MIN_SAMPLES)
        return {"status": "skipped", "reason": "insufficient_samples",
                "n_samples": len(samples)}

    # Anti-poisoning: per-IP cap
    from collections import defaultdict
    import app.services.layer1_filter as l1
    ip_counts: dict = defaultdict(int)
    clean = []
    for s in samples:
        ip = s.get("ip", "unknown")
        if ip_counts[ip] >= POISON_MAX_RATE_PER_IP:
            continue
        # Re-scan with L1: if L1 catches it, it's a known attack — don't retrain on it
        blocked, _ = l1.check(s.get("url", ""), s.get("body", ""))
        if blocked:
            continue
        ip_counts[ip] += 1
        clean.append(s)

    logger.info("Retrain: %d/%d samples passed anti-poisoning checks",
                len(clean), len(samples))

    run_doc = {
        "timestamp":    datetime.utcnow(),
        "status":       "queued",
        "n_raw":        len(samples),
        "n_clean":      len(clean),
        "note":         "Full retraining runs offline in Colab/Kaggle. "
                        "This logs the trigger event and clean sample count.",
    }
    await retrain_log().insert_one(run_doc)
    return {**run_doc, "_id": str(run_doc.get("_id", ""))}