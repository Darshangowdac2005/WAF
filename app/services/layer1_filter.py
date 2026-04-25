"""app/services/layer1.py -- rule-based filter (Layer 1)"""
import re
from urllib.parse import unquote_plus
from app.core.logging import logger

SQLI_RE = re.compile(
    r"union\s+.*\bselect\b"
    r"|or\s+\d+\s*=\s*\d+"          # OR 1=1
    r"|or\s+['\"]\w+['\"]\s*=\s*['\"]" # OR '1'='1  (quoted variant)
    r"|drop\s+table|--|sleep\s*\(|benchmark\s*\(",
    re.I)
XSS_RE = re.compile(
    r"<\s*script|onerror\s*=|javascript\s*:|alert\s*\(|svg.*onload",
    re.I)
LFI_RE = re.compile(
    r"\.\./|\.\.[/\\]|/etc/passwd|boot\.ini|/proc/self"
    r"|\.\.%2[fF]|%c0%af|%252f"           # encoded traversal variants
    r"|\.\.[\\/]{1}",                       # Windows-style ..\ traversal
    re.I)
CMDI_RE = re.compile(
    r";\s*(ls|cat|id|whoami|pwd|wget|curl|bash|sh|python|perl|nc|ncat|rm|chmod)\b"
    r"|%3[bB]\s*(ls|cat|id|whoami|wget|curl|bash|sh)\b"  # URL-encoded semicolon
    r"|&&|\|\|",
    re.I)

# NEW: Server-Side Request Forgery (SSRF)
SSRF_RE = re.compile(
    r"https?://(?:169\.254|127\.\d+\.\d+\.\d+|0\.0\.0\.0|localhost)"
    r"|file://|dict://|gopher://|tftp://|ldap://",
    re.I)

# NEW: XML External Entity injection (XXE)
XXE_RE = re.compile(
    r"<!ENTITY\b|SYSTEM\s+[\"']file://|<!DOCTYPE\b.*\[",
    re.I)

# NEW: CRLF / Header Injection
CRLF_RE = re.compile(
    r"%0[dD]%0[aA]|%0[dD]|%0[aA]|\r\n|\r|\n",
)

# NEW: Open Redirect
OPEN_REDIRECT_RE = re.compile(
    r"(?:^|[\?&=])(?:url|next|redirect|return|dest|location)\s*=\s*(?://|https?://|javascript:)",
    re.I)

_RULES = [
    (SQLI_RE,         "sqli_rule"),
    (XSS_RE,          "xss_rule"),
    (LFI_RE,          "lfi_rule"),
    (CMDI_RE,         "cmdi_rule"),
    (SSRF_RE,         "ssrf_rule"),
    (XXE_RE,          "xxe_rule"),
    (CRLF_RE,         "crlf_rule"),
    (OPEN_REDIRECT_RE,"open_redirect_rule"),
]

def check(url: str, body: str) -> tuple[bool, str]:
    """Returns (blocked: bool, reason: str)
    Applies URL-decoding before pattern matching to catch %xx-encoded evasion.
    """
    # Decode once to catch %3B -> ; %3C -> < etc.
    # Double-decode catches double-encoded payloads like %253B -> %3B -> ;
    decoded_url  = unquote_plus(unquote_plus(url))
    decoded_body = unquote_plus(unquote_plus(body))
    text = decoded_url + " " + decoded_body
    for pattern, label in _RULES:
        if pattern.search(text):
            logger.debug("L1 block: %s on %s", label, url[:80])
            return True, label
    return False, ""