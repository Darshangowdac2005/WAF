"""
ml/feature_engineering/extractor.py

Shared numeric feature extractor used by:
  - Layer 2A training  (Isolation Forest, Shallow Autoencoder)
  - Layer 2B training  (XGBoost)
  - FastAPI inference  (app/services/layer2a_anomaly.py, layer2b_deep.py)

Rules:
  - Pure stdlib + numpy ONLY — no sklearn/torch dependency
  - Every feature is a float
  - Feature order must never change after training (ONNX is order-sensitive)
  - Add new features at the END of FEATURE_NAMES only
"""

import math
import re
from urllib.parse import urlparse, parse_qs
import numpy as np

# ── Attack pattern regexes ────────────────────────────────────────────────────

_SQLI = re.compile(
    r"\b(SELECT|UNION|INSERT|UPDATE|DELETE|DROP|ALTER|EXEC|CAST|CONVERT"
    r"|CHAR|DECLARE|WAITFOR|BENCHMARK|SLEEP)\b"
    r"|\bOR\s+\d+\s*=\s*\d+"
    r"|\bAND\s+\d+\s*=\s*\d+"
    r"|--|;|/\*|\*/|0x[0-9a-fA-F]+",
    re.IGNORECASE,
)

_XSS = re.compile(
    r"<script[\s>]|</script>"
    r"|javascript\s*:"
    r"|on\w+\s*="
    r"|<iframe[\s>]"
    r"|<img[^>]+onerror"
    r"|alert\s*\("
    r"|document\.cookie"
    r"|eval\s*\("
    r"|String\.fromCharCode",
    re.IGNORECASE,
)

_LFI = re.compile(
    r"\.\./|\.\.%2[fF]|\.\.%5[cC]"
    r"|/etc/(passwd|shadow|hosts)"
    r"|/proc/self"
    r"|boot\.ini|win\.ini"
    r"|php://|expect://|data://",
    re.IGNORECASE,
)

_OSCI = re.compile(
    r"[;&|`]\s*(ls|cat|wget|curl|bash|sh|python|perl|nc|ncat|rm|chmod|chown)"
    r"|\$\(|\$\{"
    r"|%0[aA]|%0[dD]",
    re.IGNORECASE,
)

# ── HTTP encoding maps ────────────────────────────────────────────────────────

_HTTP_METHODS = {
    "GET": 0, "POST": 1, "PUT": 2, "DELETE": 3,
    "PATCH": 4, "HEAD": 5, "OPTIONS": 6, "TRACE": 7,
}

_CONTENT_TYPES = {
    "application/json": 0,
    "application/x-www-form-urlencoded": 1,
    "multipart/form-data": 2,
    "text/plain": 3,
    "text/xml": 4,
    "application/xml": 5,
    "text/html": 6,
}

# ── Feature names — ORDER IS FIXED, never reorder ────────────────────────────

FEATURE_NAMES = [
    # URL structure
    "url_length",              # 0
    "path_depth",              # 1
    "param_count",             # 2
    "param_value_total_len",   # 3
    "fragment_length",         # 4

    # Character ratios (computed on full URL)
    "special_char_ratio",      # 5  <>'"();=&%+
    "digit_ratio",             # 6
    "uppercase_ratio",         # 7
    "encoded_char_ratio",      # 8  %XX sequences

    # Body
    "payload_length",          # 9
    "payload_entropy",         # 10  Shannon entropy

    # Attack flags (binary)
    "has_sqli",                # 11
    "has_xss",                 # 12
    "has_lfi",                 # 13
    "has_osci",                # 14

    # Attack match counts
    "sqli_match_count",        # 15
    "xss_match_count",         # 16
    "lfi_match_count",         # 17
    "osci_match_count",        # 18

    # Request metadata
    "http_method",             # 19  encoded int
    "content_type",            # 20  encoded int
    "header_count",            # 21
    "has_user_agent",          # 22  binary
    "has_referer",             # 23  binary
    "cookie_length",           # 24
]

INPUT_DIM = len(FEATURE_NAMES)   # 25


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((cnt / n) * math.log2(cnt / n) for cnt in freq.values())


def _ratio(count: int, total: int) -> float:
    return count / total if total > 0 else 0.0


def _encoded_char_count(s: str) -> int:
    return len(re.findall(r"%[0-9a-fA-F]{2}", s))


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_features(request: dict) -> dict:
    """
    Extract a fixed-length numeric feature dict from a parsed request dict.

    Parameters
    ----------
    request : dict with keys:
        url      str   — full URL including query string
        method   str   — HTTP method (GET, POST, ...)
        headers  dict  — header name → value
        body     str   — request body (decoded)

    Returns
    -------
    dict[str, float] with keys matching FEATURE_NAMES (order preserved).
    All values are Python floats.
    """
    url     = request.get("url", "")
    method  = request.get("method", "GET").upper().strip()
    headers = request.get("headers", {})
    body    = request.get("body", "")

    parsed   = urlparse(url)
    params   = parse_qs(parsed.query)
    path     = parsed.path
    fragment = parsed.fragment

    scan = url + " " + body

    url_len   = len(url)
    path_depth = len([p for p in path.split("/") if p])
    param_count = len(params)
    param_val_total = sum(len(v) for vals in params.values() for v in vals)

    special = len(re.findall(r"[<>'\"\(\);=&%+\[\]{}|\\^`~]", url))
    digits  = sum(1 for c in url if c.isdigit())
    uppers  = sum(1 for c in url if c.isupper())
    encoded = _encoded_char_count(url)

    sqli_m = _SQLI.findall(scan)
    xss_m  = _XSS.findall(scan)
    lfi_m  = _LFI.findall(scan)
    osci_m = _OSCI.findall(scan)

    headers_lower  = {k.lower(): v for k, v in headers.items()}
    cookie_len     = len(headers_lower.get("cookie", ""))
    ct_raw         = headers_lower.get("content-type", "").split(";")[0].strip().lower()
    content_type_i = _CONTENT_TYPES.get(ct_raw, len(_CONTENT_TYPES))

    return {
        "url_length":            float(url_len),
        "path_depth":            float(path_depth),
        "param_count":           float(param_count),
        "param_value_total_len": float(param_val_total),
        "fragment_length":       float(len(fragment)),
        "special_char_ratio":    _ratio(special, url_len),
        "digit_ratio":           _ratio(digits, url_len),
        "uppercase_ratio":       _ratio(uppers, url_len),
        "encoded_char_ratio":    _ratio(encoded, url_len),
        "payload_length":        float(len(body)),
        "payload_entropy":       _shannon_entropy(body),
        "has_sqli":              float(bool(sqli_m)),
        "has_xss":               float(bool(xss_m)),
        "has_lfi":               float(bool(lfi_m)),
        "has_osci":              float(bool(osci_m)),
        "sqli_match_count":      float(len(sqli_m)),
        "xss_match_count":       float(len(xss_m)),
        "lfi_match_count":       float(len(lfi_m)),
        "osci_match_count":      float(len(osci_m)),
        "http_method":           float(_HTTP_METHODS.get(method, 8)),
        "content_type":          float(content_type_i),
        "header_count":          float(len(headers)),
        "has_user_agent":        float("user-agent" in headers_lower),
        "has_referer":           float("referer" in headers_lower),
        "cookie_length":         float(cookie_len),
    }


def to_vector(features: dict) -> np.ndarray:
    """Convert feature dict → numpy float32 row vector of shape (1, INPUT_DIM)."""
    return np.array(
        [features[k] for k in FEATURE_NAMES], dtype=np.float32
    ).reshape(1, -1)


def extract_vector(request: dict) -> np.ndarray:
    """Convenience: extract_features + to_vector in one call."""
    return to_vector(extract_features(request))