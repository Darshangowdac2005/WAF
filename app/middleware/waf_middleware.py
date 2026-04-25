"""app/middleware/waf_middleware.py
Starlette middleware that intercepts every proxied request,
runs the WAF pipeline, and either forwards or blocks.

IMPORTANT: This middleware only intercepts requests on /proxy/* path.
Direct API calls to /api/* bypass the WAF pipeline.
"""
import time
import uuid
from datetime import datetime

import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

import app.services.layer1_filter as l1
import app.services.layer2a_anomaly as l2a
import app.services.layer2b_deep as l2b
import app.services.threat_scorer as scorer
from app.services.feature_extractor import extract
from app.middleware.rate_limiter import rate_limiter
from app.db.queries import insert_request_log, insert_threat_event
from app.core.config import settings
from app.core.logging import logger


class WAFMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Only intercept proxy traffic
        if not request.url.path.startswith("/proxy"):
            return await call_next(request)

        t0 = time.perf_counter()
        request_id = str(uuid.uuid4())

        # ── Rate Limiter (cheapest check — runs FIRST) ────────────────────────
        rl_response = await rate_limiter.check(request)
        if rl_response is not None:
            rl_response.headers["X-Request-ID"] = request_id
            return rl_response


        # ── Read request body safely ──────────────────────────────────────────
        raw_body = b""
        body_text = ""
        try:
            raw_body = await request.body()
            body_text = raw_body.decode("utf-8", errors="replace")
        except Exception:
            pass

        # ── IMPORTANT: use clean path + query, not absolute URL ──────────────
        clean_path = request.url.path.replace("/proxy", "", 1)
        clean_query = f"?{request.url.query}" if request.url.query else ""
        clean_url = clean_path + clean_query

        req_dict = {
            "url": clean_url,
            "method": request.method,
            "headers": dict(request.headers),
            "body": body_text,
            "ip": request.client.host if request.client else None,
        }

        url = req_dict["url"]
        ip = req_dict["ip"]

        # ── Layer 1: Rule-based filter ───────────────────────────────────────
        l1_blocked, l1_reason = l1.check(url, body_text)
        if l1_blocked:
            ms = round((time.perf_counter() - t0) * 1000, 2)
            await _log_and_store(
                request_id=request_id,
                ip=ip,
                method=request.method,
                url=url,
                body_len=len(body_text),
                decision="block",
                score=100,
                label=l1_reason,
                layer="L1",
                latency_ms=ms,
                l2a_score=0.0,
                confidence=1.0,
            )
            return JSONResponse(
                status_code=403,
                content={
                    "blocked": True,
                    "reason": l1_reason,
                    "request_id": request_id,
                },
                headers={
                    "X-WAF-Decision": "block",
                    "X-Request-ID": request_id,
                    "X-WAF-Latency-ms": str(ms),
                },
            )

        # ── Feature extraction (must match training pipeline) ────────────────
        try:
            fvec, token_ids = extract(req_dict)

            logger.info(
                "FEATURE DEBUG | fvec shape=%s | fvec=%s",
                fvec.shape, fvec.tolist()
            )
            logger.info(
                "TOKEN DEBUG | token_ids shape=%s | first_30=%s",
                token_ids.shape,
                token_ids[0][:30].tolist()
                if len(token_ids.shape) > 1 else token_ids[:30].tolist()
            )
        except Exception as e:
            logger.error("Feature extraction failed: %s", e, exc_info=True)
            return await _forward(request, raw_body)

        # ── Layer 2A: Anomaly detector (gatekeeper) ──────────────────────────
        try:
            is_anomaly, l2a_score = l2a.infer(fvec)
        except Exception as e:
            logger.error("L2A inference failed: %s", e, exc_info=True)
            is_anomaly, l2a_score = False, 0.0

        # If normal → allow immediately (same as training pipeline)
        if not is_anomaly:
            ms = round((time.perf_counter() - t0) * 1000, 2)

            logger.info(
                "WAF DEBUG | url=%s | l2a_score=%.5f | is_anomaly=%s | label=normal | confidence=1.0000",
                url, l2a_score, is_anomaly
            )
            logger.info(
                "WAF DECISION | url=%s | score=%s | decision=%s",
                url, 0, "allow"
            )

            await _log_and_store(
                request_id=request_id,
                ip=ip,
                method=request.method,
                url=url,
                body_len=len(body_text),
                decision="allow",
                score=0,
                label="normal",
                layer="L2A",
                latency_ms=ms,
                l2a_score=l2a_score,
                confidence=1.0,
            )
            fwd = await _forward(request, raw_body)
            fwd.headers["X-WAF-Decision"] = "allow"
            fwd.headers["X-Request-ID"] = request_id
            fwd.headers["X-WAF-Latency-ms"] = str(ms)
            return fwd

        # ── Layer 2B: Classifier (only for anomalies) ────────────────────────
        try:
            label, confidence, _ = l2b.infer(fvec, token_ids)
        except Exception as e:
            logger.error("L2B inference failed: %s", e, exc_info=True)
            label, confidence = "other_attack", 0.5

        # ── Threat scoring (must match training logic) ───────────────────────
        score, decision = scorer.compute(l2a_score, label, confidence)
        ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "WAF DEBUG | url=%s | l2a_score=%.5f | is_anomaly=%s | label=%s | confidence=%.4f",
            url, l2a_score, is_anomaly, label, confidence
        )
        logger.info(
            "WAF DECISION | url=%s | score=%s | decision=%s",
            url, score, decision
        )

        await _log_and_store(
            request_id=request_id,
            ip=ip,
            method=request.method,
            url=url,
            body_len=len(body_text),
            decision=decision,
            score=score,
            label=label,
            layer="L2B",
            latency_ms=ms,
            l2a_score=l2a_score,
            confidence=confidence,
        )

        if decision == "block":
            return JSONResponse(
                status_code=403,
                content={
                    "blocked": True,
                    "label": label,
                    "score": score,
                    "request_id": request_id,
                },
                headers={
                    "X-WAF-Decision": "block",
                    "X-Request-ID": request_id,
                    "X-WAF-Latency-ms": str(ms),
                },
            )

        fwd = await _forward(request, raw_body)
        fwd.headers["X-WAF-Decision"] = decision  # "allow" or "log"
        fwd.headers["X-Request-ID"] = request_id
        fwd.headers["X-WAF-Latency-ms"] = str(ms)
        return fwd


async def _forward(request: Request, raw_body: bytes) -> Response:
    """Proxy request to the protected app."""
    target = settings.PROTECTED_APP_URL + request.url.path.replace("/proxy", "", 1)

    # preserve query string
    if request.url.query:
        target += f"?{request.url.query}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.request(
                method=request.method,
                url=target,
                headers={
                    k: v for k, v in request.headers.items()
                    if k.lower() not in ("host", "content-length")
                },
                content=raw_body,
            )

        return Response(
            content=r.content,
            status_code=r.status_code,
            headers=dict(r.headers),
            media_type=r.headers.get("content-type"),
        )
    except Exception as e:
        logger.error("Proxy forward failed: %s", e, exc_info=True)
        return Response(content=b"Bad Gateway", status_code=502)


async def _log_and_store(
    request_id, ip, method, url, body_len,
    decision, score, label, layer, latency_ms,
    l2a_score=None, confidence=None,
):
    now = datetime.utcnow()
    doc = {
        "request_id": request_id,
        "ip": ip,
        "method": method,
        "url": url[:500],
        "body_len": body_len,
        "decision": decision,
        "score": score,
        "label": label,
        "layer": layer,
        "latency_ms": latency_ms,
        "timestamp": now,
    }

    try:
        await insert_request_log(doc)
    except Exception as e:
        logger.error("Log insert failed: %s", e, exc_info=True)

    if decision in ("block", "log"):
        threat_doc = {
            **doc,
            "l2a_score": l2a_score,
            "confidence": confidence,
        }
        try:
            await insert_threat_event(threat_doc)
        except Exception as e:
            logger.error("Threat insert failed: %s", e, exc_info=True)

    # ── NEW: borderline requests go to feedback queue for human review ──
    if decision == "log":
        from app.db.collections import feedback_queue
        feedback_doc = {
            **doc,
            "l2a_score": l2a_score,
            "confidence": confidence,
            "verified_label": None,
            "poisoning_flag": False,
            "auto_classified": False,
        }
        try:
            await feedback_queue().insert_one(feedback_doc)
        except Exception as e:
            logger.error("Feedback insert failed: %s", e, exc_info=True)