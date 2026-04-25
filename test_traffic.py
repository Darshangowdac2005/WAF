import requests
import time
import random

WAF_URL = "http://127.0.0.1:8000/proxy"

TRAFFIC_SAMPLES = [
    # Normal traffic
    {"url": "/", "method": "GET", "desc": "Normal Home Page"},
    {"url": "/about", "method": "GET", "desc": "Normal About Page"},
    {"url": "/contact?name=John&email=john@example.com", "method": "GET", "desc": "Normal Contact Page"},
    {"url": "/search?q=security", "method": "GET", "desc": "Normal Search"},
    
    # Layer 1: SQL Injection (Regex triggered)
    {"url": "/login?user=' OR 1=1 --", "method": "GET", "desc": "SQLi (Simple ' OR 1=1)"},
    {"url": "/products?id=1; DROP TABLE users", "method": "GET", "desc": "SQLi (Drop Table)"},
    {"url": "/api/v1/user?id=1' UNION SELECT username, password FROM users--", "method": "GET", "desc": "SQLi (Union Select)"},
    
    # Layer 1: XSS (Regex triggered)
    {"url": "/comment?msg=<script>alert('XSS')</script>", "method": "GET", "desc": "XSS (Script tag)"},
    {"url": "/profile?name=<img src=x onerror=alert(1)>", "method": "GET", "desc": "XSS (Img onerror)"},
    
    # Layer 1: LFI/CMDI
    {"url": "/view?file=../../../../etc/passwd", "method": "GET", "desc": "LFI (etc/passwd)"},
    {"url": "/exec?cmd=cat /etc/shadow", "method": "GET", "desc": "CMDI (cat shadow)"},
    
    # Borderline/Anomaly (ML should handle these)
    {"url": "/api/data?q=..././..././..././etc/passwd", "method": "GET", "desc": "Obfuscated LFI"},
    {"url": "/search?q=%27%20select%20*%20from%20users%20--", "method": "GET", "desc": "Encoded SQLi"},
]

def send_traffic():
    print(f"Sending {len(TRAFFIC_SAMPLES)} traffic samples to {WAF_URL}...")
    print("-" * 60)
    
    for sample in TRAFFIC_SAMPLES:
        url = WAF_URL + sample["url"]
        desc = sample["desc"]
        method = sample["method"]
        
        try:
            if method == "GET":
                resp = requests.get(url, timeout=5)
            else:
                resp = requests.post(url, json={}, timeout=5)
            
            status = resp.status_code
            waf_decision = resp.headers.get("X-WAF-Decision", "N/A")
            latency = resp.headers.get("X-WAF-Latency-ms", "N/A")
            
            print(f"[{status}] {desc:<25} | Decision: {waf_decision:<6} | Latency: {latency}ms")
            
        except Exception as e:
            print(f"[ERR] {desc:<25} | Error: {str(e)}")
            
        time.sleep(0.5)

if __name__ == "__main__":
    send_traffic()
    print("-" * 60)
    print("Traffic simulation complete. Check the dashboard at http://127.0.0.1:8000/dashboard")
