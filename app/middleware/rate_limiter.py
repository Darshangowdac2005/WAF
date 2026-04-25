"""app/middleware/rate_limiter.py
Sliding-window rate limiter with per-IP ban capability.

Replaces the original slowapi stub with a fully async, in-memory
implementation that supports:
  - Per-IP sliding-window request counting (1-minute window)
  - Configurable request limit (RATE_LIMIT_PER_MIN)
  - Auto-ban after IP_BAN_THRESHOLD violations (returns 429)
  - IP ban duration: IP_BAN_DURATION_SEC seconds
  - Proper Retry-After header in 429 responses
"""
import asyncio
import time
from collections import deque, defaultdict
from typing import Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import settings
from app.core.logging import logger


class SlidingWindowRateLimiter:
    """
    Thread-safe, async sliding-window rate limiter with IP ban support.

    Data structures (all in-memory, reset on restart):
      _windows : IP → deque of timestamps (last 60 s)
      _violations : IP → violation count
      _ban_until  : IP → float (unix timestamp when ban expires)
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._windows: dict[str, deque] = defaultdict(deque)
        self._violations: dict[str, int] = defaultdict(int)
        self._ban_until: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check(self, request: Request) -> Optional[JSONResponse]:
        """
        Returns None if the request is allowed, or a JSONResponse(429)
        if the IP is rate-limited or banned.
        Called by WAFMiddleware before any WAF logic.
        """
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        limit = settings.RATE_LIMIT_PER_MIN
        window_sec = 60
        ban_threshold = settings.IP_BAN_THRESHOLD
        ban_duration = settings.IP_BAN_DURATION_SEC

        async with self._lock:
            # ── 1. Check if IP is currently banned ────────────────────
            ban_expires = self._ban_until.get(ip, 0.0)
            if now < ban_expires:
                retry_after = int(ban_expires - now) + 1
                logger.warning("Rate limiter: BANNED IP %s | retry_after=%ds", ip, retry_after)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "too_many_requests",
                        "detail": "Your IP has been temporarily banned due to excessive requests.",
                        "retry_after_sec": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            # ── 2. Slide the window (evict timestamps older than 60 s) ─
            win = self._windows[ip]
            while win and now - win[0] > window_sec:
                win.popleft()

            # ── 3. Check rate limit ────────────────────────────────────
            if len(win) >= limit:
                self._violations[ip] += 1
                vcount = self._violations[ip]
                logger.warning(
                    "Rate limiter: LIMIT HIT ip=%s | reqs_in_window=%d | violations=%d",
                    ip, len(win), vcount,
                )
                # Auto-ban after too many violations
                if vcount >= ban_threshold:
                    self._ban_until[ip] = now + ban_duration
                    logger.warning(
                        "Rate limiter: AUTO-BAN ip=%s | banned for %ds",
                        ip, ban_duration,
                    )
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "too_many_requests",
                            "detail": "Your IP has been temporarily banned.",
                            "retry_after_sec": ban_duration,
                        },
                        headers={"Retry-After": str(ban_duration)},
                    )
                # Normal rate-limit response (not yet banned)
                oldest = win[0] if win else now
                retry_after = int(window_sec - (now - oldest)) + 1
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "too_many_requests",
                        "detail": f"Rate limit exceeded: {limit} requests per minute.",
                        "retry_after_sec": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            # ── 4. Request is allowed — record timestamp ───────────────
            win.append(now)
            return None

    def is_banned(self, ip: str) -> bool:
        """Utility: check if an IP is currently banned (non-async)."""
        return time.monotonic() < self._ban_until.get(ip, 0.0)

    async def unban(self, ip: str) -> None:
        """Manually lift a ban on an IP (e.g. via admin API)."""
        async with self._lock:
            self._ban_until.pop(ip, None)
            self._violations[ip] = 0
            logger.info("Rate limiter: UNBANNED ip=%s", ip)


# Singleton — imported by WAFMiddleware
rate_limiter = SlidingWindowRateLimiter()