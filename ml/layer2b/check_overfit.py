"""
ml/layer2b/check_overfit.py
============================
Detects overfitting in L2B models by comparing train vs val vs test F1.
Also tests for URL-template memorisation (a known CSIC-2010 risk).

Usage:
    cd ml/layer2b
    python check_overfit.py --train-f1 0.999 --val-f1 0.994 --test-f1 0.993

Or import and call check() directly from train.py after fitting.
"""
import argparse
import sys


OVERFIT_GAP_THRESHOLD = 0.03   # train_f1 - val_f1 > 3% = overfit warning
CRITICAL_GAP          = 0.05   # > 5% = critical overfit


def check(train_f1: float, val_f1: float, test_f1: float,
          model_name: str = "model") -> bool:
    """
    Run overfit diagnostics. Returns True if the model passes (no overfit).

    Checks
    ------
    1. Train/val F1 gap
    2. Val/test F1 gap (checks for test-set leakage in model selection)
    3. Absolute test F1 floor (>= 0.95 required)
    """
    print(f"\n{'='*55}")
    print(f"Overfit Check: {model_name}")
    print(f"{'='*55}")
    print(f"  Train F1 : {train_f1:.4f}")
    print(f"  Val F1   : {val_f1:.4f}")
    print(f"  Test F1  : {test_f1:.4f}")

    train_val_gap = train_f1 - val_f1
    val_test_gap  = val_f1  - test_f1
    passed = True

    # Check 1: Train-val gap
    if train_val_gap > CRITICAL_GAP:
        print(f"\n  [CRITICAL] Train-val gap = {train_val_gap:.4f} > {CRITICAL_GAP}"
              f" → model is OVERFITTING.")
        print("  Remedies: increase dropout, reduce epochs, add weight decay, "
              "augment training data.")
        passed = False
    elif train_val_gap > OVERFIT_GAP_THRESHOLD:
        print(f"\n  [WARNING]  Train-val gap = {train_val_gap:.4f} > {OVERFIT_GAP_THRESHOLD}"
              f" → early signs of overfitting.")
    else:
        print(f"\n  [OK] Train-val gap = {train_val_gap:.4f} (within threshold)")

    # Check 2: Val-test gap (leakage indicator)
    if val_test_gap > OVERFIT_GAP_THRESHOLD:
        print(f"  [WARNING]  Val-test gap = {val_test_gap:.4f} → possible test-set "
              f"leakage in model selection (test used as secondary val).")
        print("  Fix: use a true held-out test set never seen during selection.")
        passed = False
    else:
        print(f"  [OK] Val-test gap = {val_test_gap:.4f} (no leakage signal)")

    # Check 3: Absolute floor
    if test_f1 < 0.95:
        print(f"  [WARNING]  Test F1 = {test_f1:.4f} < 0.95 floor → insufficient accuracy.")
        passed = False
    else:
        print(f"  [OK] Test F1 = {test_f1:.4f} (>= 0.95 floor)")

    # CSIC-2010 specific: if test F1 is suspiciously high (>0.998), warn about memorisation
    if test_f1 > 0.998:
        print(f"\n  [WARNING]  Test F1 = {test_f1:.4f} > 0.998 on CSIC-2010.")
        print("  This dataset has very rigid URL templates — the model may be")
        print("  memorising /tienda1/publico/*.jsp paths rather than attack patterns.")
        print("  Run holdout_eval.py with out-of-distribution requests to verify.")
        passed = False

    result = "PASS" if passed else "FAIL"
    print(f"\n  Overfit check: {result}")
    print(f"{'='*55}\n")
    return passed


def check_url_template_bias(model_predict_fn, tokenizer) -> None:
    """
    Tests whether the model relies on CSIC URL paths.
    Sends the same attack payload on a DIFFERENT URL template.
    If the model misclassifies, it's using URL structure as a cue.
    """
    print("\n[URL Template Bias Test]")

    # Same SQLi payload, different URL templates
    test_cases = [
        # (description, request_dict, expected_label)
        ("SQLi on CSIC URL",    {"method": "GET", "url": "/tienda1/publico/buscar.jsp?texto=1'+UNION+SELECT+*+FROM+users--", "body": "", "headers": {}}, "sqli"),
        ("SQLi on /api/ URL",   {"method": "GET", "url": "/api/v1/search?q=1'+UNION+SELECT+*+FROM+users--",                  "body": "", "headers": {}}, "sqli"),
        ("SQLi on /search URL", {"method": "GET", "url": "/search?query=1'+UNION+SELECT+*+FROM+users--",                     "body": "", "headers": {}}, "sqli"),
        ("SQLi in POST body",   {"method": "POST","url": "/login",  "body": "username=admin'--&password=x",                  "headers": {}}, "sqli"),
        ("XSS on random URL",   {"method": "GET", "url": "/comment?text=<script>alert(1)</script>",                          "body": "", "headers": {}}, "xss"),
    ]

    consistent = 0
    for desc, req, expected in test_cases:
        tokens = tokenizer.encode_request(req).reshape(1, -1)
        label = model_predict_fn(tokens)
        ok = (label == expected)
        consistent += ok
        status = "PASS" if ok else "FAIL (template bias?)"
        print(f"  {status}: {desc} -> predicted={label!r} expected={expected!r}")

    rate = consistent / len(test_cases)
    print(f"\n  Consistency across URL templates: {consistent}/{len(test_cases)} ({rate:.0%})")
    if rate < 0.8:
        print("  [WARNING] Model is URL-template sensitive — high overfitting risk.")
    else:
        print("  [OK] Model is reasonably URL-template agnostic.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2B overfit diagnostic")
    parser.add_argument("--train-f1", type=float, required=True)
    parser.add_argument("--val-f1",   type=float, required=True)
    parser.add_argument("--test-f1",  type=float, required=True)
    parser.add_argument("--model",    type=str,   default="model")
    args = parser.parse_args()
    passed = check(args.train_f1, args.val_f1, args.test_f1, args.model)
    sys.exit(0 if passed else 1)
