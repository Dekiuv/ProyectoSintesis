from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer, util
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
import os
import requests
import pandas as pd
import random
from gensim.models import Word2Vec
import json

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="clave_secreta")

# Inicio aplicación
app.mount("/static", StaticFiles(directory="static"), name="biblioteca")
templates = Jinja2Templates(directory="templates")

load_dotenv()
STEAM_KEYS = [
    ("DIEGO", os.getenv("STEAM_API_KEY_DIEGO")),
    ("ALVARO", os.getenv("STEAM_API_KEY_ALVARO")),
    ("ARITZ", os.getenv("STEAM_API_KEY_ARITZ")),
    ("VICTOR", os.getenv("STEAM_API_KEY_VICTOR")),
    ("RAUL", os.getenv("STEAM_API_KEY_RAUL")),
]

# === Cargar modelo de recomendación ===
modelo = Word2Vec.load("modelo_word2vec_mejorado.model")
df_meta = pd.read_csv("steam_juegos_metadata.csv").fillna("")
df_meta["appid"] = df_meta["appid"].astype(str)
metadata_dict = df_meta.set_index("appid").to_dict(orient="index")

modelo_nlp = SentenceTransformer("all-MiniLM-L6-v2")
juegos_cache = []

# Cargar CSV y modelo
chatbot_df = pd.read_csv("data/ChatbotSteam.csv", sep=';').dropna()
preguntas_codificadas = modelo_nlp.encode(chatbot_df['Question'].tolist(), convert_to_tensor=True)

def llamar_api_steam(url, params, timeout=5):
    for nombre, key in STEAM_KEYS:
        try:
            full_params = params.copy()
            full_params["key"] = key
            res = requests.get(url, params=full_params, timeout=timeout)
            res.raise_for_status()
            print(f"✅ Usando API Key de {nombre}")
            return res.json()
        except Exception as e:
            print(f"⚠️ Error con la clave de {nombre}: {e}")
    return None

# Página de login
@app.get("/", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("iniciosesion.html", {"request": request})

@app.post("/login")
async def login(request: Request, steam_id: str = Form(...)):
    datos = llamar_api_steam(
        "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/",
        {"steamid": steam_id, "include_appinfo": "true"}
    )

    if not datos or not datos.get("response") or datos["response"].get("game_count", 0) == 0:
        return templates.TemplateResponse(
            "iniciosesion.html",
            {"request": request, "error": "❌ Steam ID no válido o sin juegos."},
            status_code=400
        )

    perfil_res = llamar_api_steam(
        "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/",
        {"steamids": steam_id}
    )

    if not perfil_res or not perfil_res.get("response", {}).get("players"):
        return templates.TemplateResponse(
            "iniciosesion.html",
            {"request": request, "error": "❌ No se pudo obtener el perfil."},
            status_code=400
        )

    jugador = perfil_res["response"]["players"][0]
    avatar = jugador.get("avatarfull", "")
    nombre = jugador.get("personaname", "Usuario")

    request.session["steam_id"] = steam_id
    request.session["avatar"] = avatar
    request.session["nombre"] = nombre

    return RedirectResponse("/tienda", status_code=302)

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=302)

@app.get("/biblioteca", response_class=HTMLResponse)
async def mostrar_biblioteca(request: Request):
    if not request.session.get("steam_id"):
        return RedirectResponse("/", status_code=302)

    return templates.TemplateResponse("biblioteca.html", {
        "request": request,
        "avatar": request.session.get("avatar"),
        "nombre": request.session.get("nombre")
    })

@app.get("/juegos")
async def obtener_juegos(request: Request):
    steam_id = request.session.get("steam_id")
    if not steam_id:
        return JSONResponse(content={"error": "No autorizado"}, status_code=403)

    datos = llamar_api_steam(
        "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/",
        {"steamid": steam_id, "include_appinfo": "true"}
    )
    juegos = datos.get("response", {}).get("games", []) if datos else []

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

    amigos_data = llamar_api_steam(
        "https://api.steampowered.com/ISteamUser/GetFriendList/v1/",
        {"steamid": steam_id, "relationship": "friend"}
    )

    amigos_lista = amigos_data.get("friendslist", {}).get("friends", []) if amigos_data else []
    if not amigos_lista:
        return JSONResponse(content=[])

    ids_str = ",".join([a["steamid"] for a in amigos_lista[:30]])
    res_info = llamar_api_steam(
        "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/",
        {"steamids": ids_str}
    )

    players = res_info.get("response", {}).get("players", []) if res_info else []
    amigos_info = [{
        "nombre": jugador.get("personaname", "Desconocido"),
        "avatar": jugador.get("avatarfull", "")
    } for jugador in players]

    return JSONResponse(content=amigos_info)

@app.get("/soporte", response_class=HTMLResponse)
async def soporte(request: Request):
    temas = ["Steam", "Juegos", "Amigos", "Perfil", "Devolución"]
    return templates.TemplateResponse("soporte.html", 
                                      {"request": request,
                                                        "preguntas": temas,
                                                        "avatar":request.session.get("avatar"),
                                                        "nombre":request.session.get("nombre")})

@app.post("/preguntar")
async def preguntar(data: dict):
    pregunta = data.get("pregunta", "")
    pregunta_vec = modelo_nlp.encode(pregunta, convert_to_tensor=True)
    similitudes = util.pytorch_cos_sim(pregunta_vec, preguntas_codificadas)[0]
    top_indices = similitudes.topk(k=5).indices.tolist()
    sugerencias = [chatbot_df.iloc[i]["Question"] for i in top_indices]
    return JSONResponse({"sugerencias": sugerencias})

@app.post("/respuesta")
async def respuesta(data: dict):
    pregunta = data.get("pregunta", "")
    pregunta_vec = modelo_nlp.encode(pregunta, convert_to_tensor=True)
    similitudes = util.pytorch_cos_sim(pregunta_vec, preguntas_codificadas)[0]
    idx = similitudes.argmax().item()
    respuesta = chatbot_df.iloc[idx]["Answer"]
    return JSONResponse({"respuesta": respuesta})

@app.post("/sugerencias")
async def sugerencias(data: dict):
    pregunta = data.get("pregunta", "")
    pregunta_vec = modelo_nlp.encode(pregunta, convert_to_tensor=True)
    similitudes = util.pytorch_cos_sim(pregunta_vec, preguntas_codificadas)[0]
    top_indices = similitudes.topk(k=10).indices.tolist()
    random.shuffle(top_indices)
    top5 = top_indices[:5]
    sugerencias = [chatbot_df.iloc[i]["Question"] for i in top5]
    return JSONResponse({"sugerencias": sugerencias})

def obtener_juegos_usuario_con_tiempo(steam_id):
    datos = llamar_api_steam(
        "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/",
        {"steamid": steam_id, "include_appinfo": "true"}
    )
    juegos = datos.get("response", {}).get("games", []) if datos else []
    return {str(j["appid"]): j.get("playtime_forever", 0) for j in juegos if j.get("playtime_forever", 0) > 0}

def recomendar_desde_juegos(juegos_dict, topn=20, alpha=0.7):
    similares = {}
    for item_id, playtime in juegos_dict.items():
        if item_id in modelo.wv:
            peso = min(playtime / 60, 10)
            try:
                similares_raw = modelo.wv.most_similar(item_id, topn=50)
                print(similares_raw)
            except KeyError:
                continue

            for similar_id, score in similares_raw:
                if similar_id not in metadata_dict or similar_id in juegos_dict:
                    continue

                metacritic = metadata_dict[similar_id].get("metacritic_score", 0)
                try:
                    metacritic = float(metacritic) / 100 if metacritic else 0
                except:
                    metacritic = 0

                score_final = peso * (alpha * score + (1 - alpha) * metacritic)
                similares[similar_id] = similares.get(similar_id, 0) + score_final

    recomendaciones = sorted(similares.items(), key=lambda x: x[1], reverse=True)[:topn]
    resultado = []
    for item_id, score in recomendaciones:
        meta = metadata_dict[item_id]
        resultado.append({
            "item_id": item_id,
            "nombre": meta["name"],
            "score": round(score, 4),
            "metacritic": meta.get("metacritic_score"),
            "imagen": meta.get("header_image"),
            "descripcion": meta.get("short_description")
        })
    return resultado

# === Página de tienda ===
@app.get("/tienda", response_class=HTMLResponse)
async def mostrar_tienda(request: Request):
    steam_id = request.session.get("steam_id")
    if not steam_id:
        return RedirectResponse("/", status_code=302)

    juegos_dict = obtener_juegos_usuario_con_tiempo(steam_id)
    juegos_validos = {j: p for j, p in juegos_dict.items() if j in modelo.wv}
    recomendaciones = recomendar_desde_juegos(juegos_validos, 50, 0.8)

    return templates.TemplateResponse("tienda.html", {
        "request": request,
        "avatar": request.session.get("avatar"),
        "nombre": request.session.get("nombre"),
        "recomendaciones": recomendaciones
    })
