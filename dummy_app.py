from fastapi import FastAPI, Query

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Protected backend app is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/hello")
def hello():
    return {"message": "Hello from protected app"}


@app.get("/search")
def search(q: str = ""):
    return {"query": q}


# ------------------------------------------------------------------
# CSIC-style e-commerce endpoints
# ------------------------------------------------------------------

@app.get("/tienda1/publico/anadir.jsp")
def anadir(id: int = 0, nombre: str = ""):
    return {
        "action": "add_product",
        "id": id,
        "nombre": nombre
    }


@app.get("/tienda1/publico/registro.jsp")
def registro(
    nombre: str = "",
    apellido: str = "",
    email: str = "",
    telefono: str = ""
):
    return {
        "action": "register",
        "nombre": nombre,
        "apellido": apellido,
        "email": email,
        "telefono": telefono
    }


@app.get("/tienda1/publico/login.jsp")
def login(usuario: str = "", password: str = ""):
    return {
        "action": "login",
        "usuario": usuario,
        "password": password
    }


@app.get("/tienda1/publico/buscar.jsp")
def buscar(texto: str = ""):
    return {
        "action": "search_products",
        "texto": texto
    }


@app.get("/tienda1/publico/productos.jsp")
def productos(categoria: str = "", id: int = 0):
    return {
        "action": "view_products",
        "categoria": categoria,
        "id": id
    }


@app.get("/tienda1/publico/detalles.jsp")
def detalles(id: int = 0):
    return {
        "action": "product_details",
        "id": id
    }


@app.get("/tienda1/publico/carrito.jsp")
def carrito(id: int = 0, cantidad: int = 1):
    return {
        "action": "cart",
        "id": id,
        "cantidad": cantidad
    }


@app.get("/tienda1/publico/comentarios.jsp")
def comentarios(id: int = 0, texto: str = ""):
    return {
        "action": "comment",
        "id": id,
        "texto": texto
    }


@app.get("/tienda1/publico/contacto.jsp")
def contacto(asunto: str = "", mensaje: str = ""):
    return {
        "action": "contact",
        "asunto": asunto,
        "mensaje": mensaje
    }


@app.get("/tienda1/publico/usuarios.jsp")
def usuarios(nombre: str = ""):
    return {
        "action": "user_lookup",
        "nombre": nombre
    }


# ------------------------------------------------------------------
# Extra endpoints useful for attack simulation
# ------------------------------------------------------------------

@app.get("/tienda1/publico/ver.jsp")
def ver(file: str = "", template: str = ""):
    return {
        "action": "view_file",
        "file": file,
        "template": template
    }


@app.get("/tienda1/publico/download.jsp")
def download(doc: str = ""):
    return {
        "action": "download",
        "doc": doc
    }


@app.get("/tienda1/publico/admin.jsp")
def admin(page: str = ""):
    return {
        "action": "admin_page",
        "page": page
    }


@app.get("/tienda1/publico/exec.jsp")
def exec_cmd(cmd: str = ""):
    return {
        "action": "execute",
        "cmd": cmd
    }


@app.get("/tienda1/publico/run.jsp")
def run(input: str = ""):
    return {
        "action": "run_input",
        "input": input
    }


@app.get("/tienda1/publico/test.jsp")
def test(x: str = ""):
    return {
        "action": "test",
        "x": x
    }


@app.get("/tienda1/publico/check.jsp")
def check(value: str = ""):
    return {
        "action": "check",
        "value": value
    }