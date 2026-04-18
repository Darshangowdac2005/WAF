"""app/services/health.py — server health monitor + retraining trigger"""
import asyncio, httpx
from datetime import datetime
from app.core.config import settings
from app.core.logging import logger
from app.db.queries import insert_health_snapshot, get_pending_feedback

_running = False

async def start_monitor() -> None:
    global _running
    _running = True
    asyncio.create_task(_loop())
    logger.info("Health monitor started (interval=%ds)",
                settings.HEALTH_CHECK_INTERVAL_SEC)

async def stop_monitor() -> None:
    global _running
    _running = False

async def _loop() -> None:
    while _running:
        await asyncio.sleep(settings.HEALTH_CHECK_INTERVAL_SEC)
        try:
            snapshot = await _check_app_health()
            await insert_health_snapshot(snapshot)
            if snapshot.get("error_rate", 0) >= settings.ERROR_RATE_THRESHOLD:
                logger.warning("Error rate %.2f%% exceeds threshold — triggering audit",
                               snapshot["error_rate"] * 100)
                await _trigger_audit()
        except Exception as e:
            logger.error("Health monitor error: %s", e)

# app/services/health_monitor.py
async def _check_app_health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:  # was 5.0
            r = await client.get(f"{settings.PROTECTED_APP_URL}/health")
            app_data = r.json() if r.status_code == 200 else {}
    except Exception:
        app_data = {}

    return {
        "timestamp":   datetime.utcnow(),
        "app_ok":      bool(app_data),
        "error_rate":  app_data.get("error_rate", 0.0),
        "latency_p99": app_data.get("latency_p99", 0.0),
        "cpu_pct":     app_data.get("cpu_pct", 0.0),
    }