"""app/services/threat_scorer.py"""
from typing import Tuple

# Per-attack severity multipliers applied to the L2B confidence contribution.
# Higher values = more aggressive scoring for dangerous attack types.
_SEVERITY: dict[str, float] = {
    "sqli":         1.0,   # full weight — critical data exfiltration risk
    "cmdi":         1.0,   # full weight — RCE risk
    "lfi":          0.95,  # near-critical — file disclosure risk
    "xxe":          0.95,  # near-critical — SSRF/file read via XML
    "ssrf":         0.90,  # high — internal network access risk
    "xss":          0.85,  # medium-high — session hijack / phishing
    "other_attack": 0.80,  # medium — generic anomaly, less certain
    "normal":       0.0,   # no contribution
}


def compute(l2a_score: float, label: str, confidence: float) -> Tuple[int, str]:
    """
    Compute a 0–100 threat score and a decision string.

    Scoring formula
    ---------------
    l2a_contrib  = min(50, l2a_score × 15)          — anomaly gate contribution
    l2b_contrib  = confidence × 50 × severity_mult   — classifier contribution
    threat_score = clamp(l2a_contrib + l2b_contrib, 0, 100)

    Decision thresholds (match training pipeline)
    ─────────────────────────────────────────────
    ≥ 70  → block
    ≥ 30  → log   (borderline — goes to feedback queue for human review)
    < 30  → allow

    Returns
    -------
    (threat_score: int, decision: str)
    """
    l2a_contrib = min(50.0, l2a_score * 15)

    severity = _SEVERITY.get(label, 0.8)   # default: treat unknowns as medium
    l2b_contrib = confidence * 50.0 * severity if label != "normal" else 0.0

    threat_score = max(0, min(100, int(l2a_contrib + l2b_contrib)))

    if threat_score >= 70:
        decision = "block"
    elif threat_score >= 30:
        decision = "log"
    else:
        decision = "allow"

    return threat_score, decision