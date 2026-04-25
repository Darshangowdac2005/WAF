"""smoke_test.py -- Live WAF integration smoke test
WAF proxies protected app via /proxy/* prefix.
Send all traffic to http://127.0.0.1:8000/proxy/<backend-path>
"""
import urllib.request, urllib.error, sys

BASE  = "http://127.0.0.1:8000"
PROXY = BASE + "/proxy"   # WAF intercepts /proxy/* and forwards to backend
PASS = 0; FAIL = 0

def check(label, url, expected_code):
    global PASS, FAIL
    try:
        r = urllib.request.urlopen(url, timeout=5)
        code = r.status
        hdrs = dict(r.getheaders())
    except urllib.error.HTTPError as e:
        code = e.code
        hdrs = dict(e.headers)
    decision = hdrs.get("X-WAF-Decision",   "MISSING")
    latency  = hdrs.get("X-WAF-Latency-ms", "MISSING")
    req_id   = hdrs.get("X-Request-ID",     "MISSING")
    rid_short = req_id[:8] if req_id != "MISSING" else "MISSING"
    ok = (code == expected_code)
    sym = "PASS" if ok else "FAIL"
    if ok: PASS += 1
    else:  FAIL += 1
    print("  %s  %-38s HTTP=%d  decision=%-12s  latency=%sms  rid=%s..." % (
        sym, label, code, decision, latency, rid_short))

print("\n" + "="*70)
print("  WAF Live Smoke Test  (WAF: %s)" % BASE)
print("="*70 + "\n")

print("-- WAF own routes (no proxy) --")
check("GET /  (WAF root)",          BASE+"/",                          200)
check("GET /api/health",            BASE+"/api/health",                200)

print("\n-- Normal Traffic via /proxy/* (expect 200) --")
check("GET /proxy/hello",           PROXY+"/hello",                                   200)
check("GET /proxy/search?q=laptop", PROXY+"/search?q=laptop",                         200)
check("GET /proxy/ productos.jsp",  PROXY+"/tienda1/publico/productos.jsp?id=1",      200)

print("\n-- SQLi Attacks via /proxy/* (expect 403) --")
check("UNION SELECT",   PROXY+"/search?q=1'+UNION+SELECT+*+FROM+users--",   403)
check("OR 1=1",         PROXY+"/search?q=admin'+OR+'1'%3D'1",               403)
check("SLEEP()",        PROXY+"/search?q=1%3BSELECT+SLEEP(5)--",            403)

print("\n-- XSS Attacks via /proxy/* (expect 403) --")
check("<script>",       PROXY+"/search?q=%3Cscript%3Ealert(1)%3C%2Fscript%3E", 403)
check("onerror=",       PROXY+"/search?q=%3Cimg+onerror%3Dalert(1)%3E",        403)

print("\n-- LFI via /proxy/* (expect 403) --")
check("../etc/passwd",  PROXY+"/search?q=../../../../etc/passwd",             403)
check("encoded %252f",  PROXY+"/search?q=..%252fetc/passwd",                  403)

print("\n-- SSRF (expect 403, NEW rule) --")
check("169.254.*",      PROXY+"/search?q=http://169.254.169.254/latest/",     403)
check("gopher://",      PROXY+"/search?q=gopher://127.0.0.1:6379/_PING",      403)

print("\n-- XXE (expect 403, NEW rule) --")
check("<!ENTITY>",      PROXY+"/search?q=%3C!ENTITY+xxe+SYSTEM+file:///etc/passwd%3E", 403)

print("\n-- CRLF Injection (expect 403, NEW rule) --")
check("%0d%0a inj",     PROXY+"/search?q=test%0d%0aSet-Cookie:+evil%3D1",    403)

print("\n-- Open Redirect (expect 403, NEW rule) --")
check("redirect=//evil",PROXY+"/search?q=x&redirect=//evil.com/phish",       403)

print("\n-- CMDi (expect 403) --")
check(";whoami",        PROXY+"/search?q=test%3Bwhoami",                     403)

print("\n" + "="*70)
total = PASS + FAIL
print("  Results: %d/%d passed" % (PASS, total))
if FAIL == 0:
    print("  ALL TESTS PASSED")
else:
    print("  %d FAILED" % FAIL)
print("="*70 + "\n")
sys.exit(0 if FAIL == 0 else 1)
