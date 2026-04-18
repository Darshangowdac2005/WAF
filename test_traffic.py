# test_traffic.py — run from project root
import requests

BASE = "http://127.0.0.1:8000/proxy"

# These should score in the borderline zone (log, not block)
# Slightly suspicious but not obvious attacks
borderline = [
    "/tienda1/publico/buscar.jsp?texto=select+nombre+from+usuarios",
    "/tienda1/publico/login.jsp?usuario=admin'--&password=test",
    "/tienda1/publico/ver.jsp?file=../config.php",
    "/tienda1/publico/buscar.jsp?texto=<b>test</b>",
    "/tienda1/publico/comentarios.jsp?texto=javascript:void(0)",
    "/tienda1/publico/download.jsp?doc=....//....//etc/hosts",
    "/tienda1/publico/buscar.jsp?texto=1+OR+1=1",
    "/tienda1/publico/usuarios.jsp?nombre=admin'+OR+'1'='1",
    "/tienda1/publico/ver.jsp?template=../templates/base",
    "/tienda1/publico/check.jsp?value=;whoami",
]

# These should be clearly blocked (score >= 70)
attacks = [
    "/tienda1/publico/buscar.jsp?texto=1'+UNION+SELECT+*+FROM+users--",
    "/tienda1/publico/login.jsp?usuario=<script>alert(1)</script>",
    "/tienda1/publico/ver.jsp?file=../../../../etc/passwd",
]

# These should be allowed (score < 30)
normal = [
    "/tienda1/publico/anadir.jsp?id=1&nombre=laptop",
    "/tienda1/publico/productos.jsp?categoria=electronics",
    "/tienda1/publico/login.jsp?usuario=john&password=hello123",
    "/tienda1/publico/buscar.jsp?texto=zapatos+rojos",
    "/hello",
    "/search?q=products",
]

print("=== Sending normal traffic ===")
for path in normal:
    r = requests.get(BASE + path)
    print(f"  {r.status_code} {path[:60]}")

print("\n=== Sending borderline traffic ===")
for path in borderline:
    r = requests.get(BASE + path)
    print(f"  {r.status_code} {path[:60]}")

print("\n=== Sending attack traffic ===")
for path in attacks:
    r = requests.get(BASE + path)
    print(f"  {r.status_code} {path[:60]}")

print("\nDone. Check http://127.0.0.1:8000/dashboard/feedback")