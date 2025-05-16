from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer, util
from starlette.middleware.sessions import SessionMiddleware
import requests

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="clave_secreta")

app.mount("/static", StaticFiles(directory="static"), name="biblioteca")
templates = Jinja2Templates(directory="templates")

API_KEY = "F5E52AD27E9DC7006A2068AA05B6EE04"
modelo_nlp = SentenceTransformer("all-MiniLM-L6-v2")
juegos_cache = []

# Página de login
@app.get("/", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("iniciosesion.html", {"request": request})

@app.post("/login")
async def login(request: Request, steam_id: str = Form(...)):
    # Verifica si el usuario existe
    url = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={API_KEY}&steamid={steam_id}&include_appinfo=true"
    res = requests.get(url)
    datos = res.json()

    if not datos.get("response") or datos["response"].get("game_count", 0) == 0:
        return templates.TemplateResponse(
            "iniciosesion.html",
            {"request": request, "error": "❌ Steam ID no válido o sin juegos."},
            status_code=400
        )

    # Obtener nombre y avatar del usuario
    perfil_url = f"https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/?key={API_KEY}&steamids={steam_id}"
    perfil_res = requests.get(perfil_url).json()
    jugador = perfil_res["response"]["players"][0]
    
    avatar = jugador.get("avatarfull", "")
    nombre = jugador.get("personaname", "Usuario")

    request.session["steam_id"] = steam_id
    request.session["avatar"] = avatar
    request.session["nombre"] = nombre

    return RedirectResponse("/biblioteca", status_code=302)



@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=302)

# Página principal
@app.get("/biblioteca", response_class=HTMLResponse)
async def mostrar_biblioteca(request: Request):
    if not request.session.get("steam_id"):
        return RedirectResponse("/", status_code=302)

    return templates.TemplateResponse("biblioteca.html", {
        "request": request,
        "avatar": request.session.get("avatar"),
        "nombre": request.session.get("nombre")
    })


# Cargar juegos
@app.get("/juegos")
async def obtener_juegos(request: Request):
    steam_id = request.session.get("steam_id")
    if not steam_id:
        return JSONResponse(content={"error": "No autorizado"}, status_code=403)

    url = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={API_KEY}&steamid={steam_id}&include_appinfo=true"
    res = requests.get(url)
    juegos = res.json().get("response", {}).get("games", [])

    juegos_info = []
    for juego in juegos[:100]:
        appid = juego["appid"]
        nombre = juego["name"]
        imagen = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{appid}/header.jpg"

        url_detalle = f"https://store.steampowered.com/api/appdetails?appids={appid}&cc=es&l=spanish"
        try:
            response = requests.get(url_detalle, timeout=5)
            data_json = response.json()
            if not isinstance(data_json, dict) or str(appid) not in data_json:
                continue
            detalle = data_json[str(appid)]
        except Exception:
            continue

        descripcion = "Sin descripción."
        categorias = []

        if detalle.get("success"):
            data = detalle["data"]
            descripcion = data.get("short_description", descripcion)
            categorias = [c["description"] for c in data.get("categories", [])]

        juegos_info.append({
            "appid": appid,
            "nombre": nombre,
            "imagen": imagen,
            "descripcion": descripcion,
            "categorias": categorias
        })

    global juegos_cache
    juegos_cache = juegos_info
    return JSONResponse(content=juegos_info)

# Buscador exacto
@app.get("/buscar")
async def buscar_juego(q: str):
    if not juegos_cache:
        return JSONResponse(content=[])

    q = q.strip().lower()
    resultados = [j for j in juegos_cache if q in j["nombre"].lower()]
    return JSONResponse(content=resultados)

@app.get("/amigos")
async def obtener_amigos(request: Request):
    steam_id = request.session.get("steam_id")
    if not steam_id:
        return JSONResponse(content=[], status_code=403)

    # Obtener lista de SteamIDs de los amigos
    amigos_url = f"https://api.steampowered.com/ISteamUser/GetFriendList/v1/?key={API_KEY}&steamid={steam_id}&relationship=friend"
    res = requests.get(amigos_url)
    amigos_data = res.json().get("friendslist", {}).get("friends", [])

    if not amigos_data:
        return JSONResponse(content=[])

    steam_ids_amigos = [amigo["steamid"] for amigo in amigos_data[:30]]  # límite para no saturar
    ids_str = ",".join(steam_ids_amigos)

    # Obtener información de los amigos
    info_url = f"https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/?key={API_KEY}&steamids={ids_str}"
    res_info = requests.get(info_url)
    players = res_info.json().get("response", {}).get("players", [])

    amigos_info = [{
        "nombre": jugador.get("personaname", "Desconocido"),
        "avatar": jugador.get("avatarfull", "")
    } for jugador in players]

    return JSONResponse(content=amigos_info)

