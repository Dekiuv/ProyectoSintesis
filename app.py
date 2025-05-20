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
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
from joblib import load
from pathlib import Path
from ast import literal_eval

# ========== CARGAR MODELO SENTIMIENTO ==========
print("⏳ Cargando modelo de sentimiento...")
tokenizer_sent = AutoTokenizer.from_pretrained("modelo_sentimiento")
model_sent = AutoModelForSequenceClassification.from_pretrained("modelo_sentimiento")
sentiment_analyzer = pipeline("sentiment-analysis", model=model_sent, tokenizer=tokenizer_sent)
print("✅ Modelo de sentimiento cargado.")


app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="clave_secreta")

# Inicio aplicación
app.mount("/static", StaticFiles(directory="static"), name="biblioteca")
templates = Jinja2Templates(directory="templates")

load_dotenv()
STEAM_KEYS = [
    # ("DIEGO", os.getenv("STEAM_API_KEY_DIEGO")),
    ("ALVARO", os.getenv("STEAM_API_KEY_ALVARO")),
    ("ARITZ", os.getenv("STEAM_API_KEY_ARITZ")),
    # ("VICTOR", os.getenv("STEAM_API_KEY_VICTOR")),
    # ("RAUL", os.getenv("STEAM_API_KEY_RAUL")),
]

# === Cargar modelo de recomendación ===
modelo = Word2Vec.load("word2vec_steam.model")
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


def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+|www.\S+", "", texto)
    texto = re.sub(r"[^\w\sáéíóúüñçàèìòùâêîôûäëïöü]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# Página de login
@app.get("/", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("iniciosesion.html", {"request": request})

@app.post("/login")
async def login(request: Request, steam_id: str = Form(...)):
    print("🧪 steam_id recibido:", steam_id)

    datos = llamar_api_steam(
        "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/",
        {"steamid": steam_id, "include_appinfo": "true"}
    )
    print("📦 Respuesta de GetOwnedGames:", datos)

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
        imagen = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{appid}/capsule_616x353.jpg"

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

import os
import requests
import pandas as pd

from fastapi.responses import JSONResponse

@app.get("/api/recomendacionesw2v")
async def api_recomendaciones(request: Request):
    steam_id = request.session.get("steam_id")
    if not steam_id:
        return JSONResponse(content={"error": "No hay usuario en sesión"}, status_code=401)

    _, recomendaciones, _ = recomendar_juegos_word2vec_con_nombres_unificado(modelo, steam_id)
    return JSONResponse(content={"recomendaciones": recomendaciones})


# Lista de claves con prioridad
def recomendar_juegos_word2vec_con_nombres_unificado(modelo, steam_id, topn=30):
    """
    Recomienda juegos a un usuario de Steam usando un modelo Word2Vec entrenado.
    Selecciona automáticamente la mejor API key disponible.

    Returns:
        jugados_nombres (list[str]): juegos jugados con nombre.
        recomendados (list[dict]): recomendaciones detalladas.
        mensaje (str): mensaje resumen del proceso.
    """

    # Cargar nombres
    try:
        appid_to_name = pd.read_csv("nombres_juegos.csv").set_index("appid")["name"].to_dict()
    except Exception as e:
        return [], [], f"❌ Error al cargar nombres de juegos: {e}"

    # Buscar una API Key válida
    appids = []
    clave_usada = None
    for nombre, clave in STEAM_KEYS:
        if not clave:
            continue

        url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
        params = {
            "key": clave,
            "steamid": steam_id,
            "include_appinfo": 1,
            "format": "json"
        }

        try:
            response = requests.get(url, params=params)
            print(response)
            response.raise_for_status()
            juegos_usuario = response.json().get("response", {}).get("games", [])
            appids = [str(j["appid"]) for j in juegos_usuario if j.get("playtime_forever", 0) > 0]
            clave_usada = nombre
            break
        except Exception:
            continue  # Probar la siguiente clave

    if not appids:
        print("❌ No se pudieron obtener los juegos del usuario con ninguna API Key.")
        return [], [], "❌ No se pudieron obtener los juegos del usuario con ninguna API Key."

    # Filtrar appids válidos
    appids_validos = [a for a in appids if a in modelo.wv.key_to_index]

    ignorados = len(appids) - len(appids_validos)
    jugados_nombres = [appid_to_name.get(int(a), f"Unknown ({a})") for a in appids_validos]

    if not appids_validos:
        return jugados_nombres, [], "⚠️ Ningún juego jugado está en el vocabulario del modelo."

    # Generar recomendaciones
    try:
        recomendaciones = modelo.wv.most_similar(appids_validos, topn=topn * 2)
        recomendados = []
        for appid, similitud in recomendaciones:
            if appid in appids_validos:
                continue
            nombre = appid_to_name.get(int(appid), f"Unknown ({appid})")
            info_extra = obtener_datos_juego(appid)

            recomendados.append({
                "item_id": int(appid),
                "nombre": nombre,
                "score": round(similitud, 4),
                "metacritic": None,
                "imagen": info_extra["imagen"],
                "descripcion": info_extra["descripcion"]
            })

            if len(recomendados) >= topn:
                break

    except Exception as e:
        return jugados_nombres, [], f"❌ Error al generar recomendaciones: {e}"

    mensaje = f"✅ Recomendaciones generadas correctamente con clave {clave_usada}"
    if ignorados > 0:
        mensaje += f" (⚠️ {ignorados} juegos ignorados por no estar en el modelo)"
    return jugados_nombres, recomendados, mensaje


def obtener_datos_juego(appid):
    """Llama a la API de Steam Store para obtener imagen y descripción"""
    try:
        url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=spanish"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        datos = res.json()
        info = datos.get(str(appid), {})
        
        if not info.get("success", False):
            print(f"⚠️ AppID {appid} no tiene datos válidos (success=False)")
            return {"imagen": None, "descripcion": None}

        data = info.get("data", {})
        return {
            "imagen": data.get("header_image"),
            "descripcion": data.get("short_description")
        } if data else {"imagen": None, "descripcion": None}
    except Exception as e:
        print(f"❌ Error al obtener datos del juego {appid}: {e}")
        return {"imagen": None, "descripcion": None}



def obtener_top_juegos():
    url = "https://store.steampowered.com/search/?filter=topsellers&cc=es"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "lxml")

    juegos = []
    for entry in soup.select(".search_result_row")[:12]:
        nombre = entry.find("span", class_="title").text.strip()
        imagen = entry.find("img")["src"]
        url_juego = entry["href"]
        appid_match = re.search(r'/app/(\d+)/', url_juego)
        appid = appid_match.group(1) if appid_match else None

        price_tag = entry.find('div', class_='search_price')
        precio = price_tag.text.strip() if price_tag else "No disponible"

        if appid:
            juegos.append({
                "nombre": nombre,
                "imagen": imagen,
                "appid": appid
            })

    return juegos


# === Página de tienda ===
@app.get("/tienda", response_class=HTMLResponse)
async def mostrar_tienda(request: Request):
    steam_id = request.session.get("steam_id")
    if not steam_id:
        return RedirectResponse("/", status_code=302)

    juegos_dict = obtener_juegos_usuario_con_tiempo(steam_id)
    juegos_validos = {j: p for j, p in juegos_dict.items() if j in modelo.wv}
    _, recomendaciones, _  = recomendar_juegos_word2vec_con_nombres_unificado(modelo, steam_id, 50)
    print("---------------------_",recomendaciones)
    top_juegos = obtener_top_juegos()
    
    return templates.TemplateResponse("tienda.html", {
        "request": request,
        "avatar": request.session.get("avatar"),
        "nombre": request.session.get("nombre"),
        "recomendaciones": recomendaciones,
        "top_juegos": top_juegos
    })

@app.get("/juego/{appid}", response_class=HTMLResponse)
async def detalle_juego(request: Request, appid: int):
    # Datos del juego
    resumen_sentimiento = "Sin reseñas suficientes"
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&cc=es&l=spanish"
    res = requests.get(url)
    data = res.json()
    
    juego_data = data.get(str(appid), {}).get("data", {})
    if not juego_data:
        return HTMLResponse("Juego no encontrado", status_code=404)

    juego = {
        "appid": appid,
        "nombre": juego_data.get("name"),
        "descripcion": juego_data.get("short_description", "Sin descripción."),
        "imagen": juego_data.get("header_image"),
        "categorias": [c["description"] for c in juego_data.get("categories", [])],
        "precio": juego_data.get("price_overview", {}).get("final_formatted", "Gratis")
    }

    # Reviews del juego
    res_reviews = requests.get(f"https://store.steampowered.com/appreviews/{appid}?json=1&language=spanish&num_per_page=10")
    reviews_raw = res_reviews.json().get("reviews", [])

    reviews = []
    for review in reviews_raw:
        steamid = review["author"]["steamid"]
        texto = review["review"]

        # Avatar y nombre del usuario
        perfil = llamar_api_steam(
            "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/",
            {"steamids": steamid}
        )

        jugador = perfil.get("response", {}).get("players", [{}])[0]

        avatar = jugador.get("avatarfull", "/static/default_avatar.png")
        nombre = jugador.get("personaname", "Usuario")

        # Analizar sentimiento
        limpio = limpiar_texto(texto)
        resultado = sentiment_analyzer(limpio, truncation=True)[0]
        estrellas = int(resultado['label'][0])  # asume formato "5 stars"
        imagen_estrella = f"/static/{estrellas}.png"

        reviews.append({
            "usuario": nombre,
            "avatar": avatar,
            "texto": texto,
            "estrellas": estrellas,
            "imagen_estrella": imagen_estrella
        })

        # Calcular media de estrellas
        if reviews:
            promedio = sum([r["estrellas"] for r in reviews]) / len(reviews)
            if promedio < 1.5:
                resumen_sentimiento = "Muy malo"
            elif promedio < 2.5:
                resumen_sentimiento = "Malo"
            elif promedio < 3.5:
                resumen_sentimiento = "Neutral"
            elif promedio < 4.5:
                resumen_sentimiento = "Bueno"
            else:
                resumen_sentimiento = "Muy bueno"
        else:
            resumen_sentimiento = "Sin reseñas suficientes"



    return templates.TemplateResponse("juego.html", {
    "request": request,
    "juego": juego,
    "reviews": reviews,
    "nombre": request.session.get("nombre"),
    "avatar": request.session.get("avatar"),
    "resumen_sentimiento": resumen_sentimiento
    })


@app.get("/api/recomendacionesmba")
async def recomendar_mba_para_usuario(request: Request):
    # Paso 1: Obtener juegos del usuario desde la API de Steam usando función reutilizable
    steam_id = request.session.get("steam_id")
    
    datos = llamar_api_steam(
        "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/",
        {"steamid": steam_id, "format": "json"}
    )

    if not datos or "games" not in datos.get("response", {}):
        print("❌ Error al obtener los juegos del usuario desde la API.")
        return {"recomendaciones": []}

    juegos_usuario = datos["response"]["games"]
    appids_usuario = {j["appid"] for j in juegos_usuario if j.get("playtime_forever", 0) > 0}

    if not appids_usuario:
        print("⚠️ El usuario no tiene juegos jugados.")
        return {"recomendaciones": []}

    # Paso 2: Cargar matriz binaria original y modelos
    matriz_binaria = pd.read_csv("matriz_binaria_filtrada.csv", index_col=0)
    pca = load("modelo_pca.joblib")
    kmeans = load("modelo_kmeans.joblib")
    columnas_appids = pd.read_pickle("appids_entrenados.pkl")

    # Paso 3: Crear vector binario para este usuario
    vector_usuario = pd.Series(0, index=columnas_appids)
    vector_usuario[list(appids_usuario & set(columnas_appids))] = 1

    # Paso 4: Reducir y asignar cluster
    vector_reducido = pca.transform([vector_usuario])
    cluster = int(kmeans.predict(vector_reducido)[0])
    print(f"🧠 Usuario asignado al cluster {cluster}")

    # Paso 5: Cargar reglas de asociación
    reglas_path = f"reglas_cluster_{cluster}.csv"
    if not Path(reglas_path).exists():
        print("⚠️ No hay reglas guardadas para este cluster.")
        return {"recomendaciones": []}

    reglas = pd.read_csv(reglas_path)

    # Evaluar 'frozenset' correctamente si están en string
    if isinstance(reglas['antecedents'].iloc[0], str) and reglas['antecedents'].str.startswith('frozenset').any():
        reglas['antecedents'] = reglas['antecedents'].apply(eval)
        reglas['consequents'] = reglas['consequents'].apply(eval)
    elif isinstance(reglas['antecedents'].iloc[0], str):
        reglas['antecedents'] = reglas['antecedents'].apply(lambda x: frozenset(literal_eval(x)))
        reglas['consequents'] = reglas['consequents'].apply(lambda x: frozenset(literal_eval(x)))

    # Paso 6: Cargar nombres de juegos
    nombres_juegos = pd.read_csv("nombres_juegos.csv").set_index("appid")["name"].to_dict()

    # Paso 7: Filtrar reglas aplicables
    recomendaciones = []
    for _, fila in reglas.iterrows():
        if fila['antecedents'].issubset(appids_usuario) and not fila['consequents'].issubset(appids_usuario):
            appid = list(fila['consequents'])[0]
            recomendaciones.append((appid, fila['confidence'], fila['lift']))

    if not recomendaciones:
        print("🤷 No se encontraron recomendaciones para este usuario.")
        return {"recomendaciones": []}

    # Paso 8: Mostrar top recomendaciones con nombres
    recomendaciones = sorted(recomendaciones, key=lambda x: (-x[1], -x[2]))[:10]
    resultados = [{
        "item_id": appid,
        "nombre": nombres_juegos.get(appid, f"Juego {appid}")
    } for appid, _, _ in recomendaciones]

    return {"recomendaciones": resultados}


@app.post("/agregar_al_carrito")
async def agregar_al_carrito(request: Request, appid: int = Form(...)):
    if "carrito" not in request.session:
        request.session["carrito"] = []

    appid_str = str(appid)

    # Consultar API para tener precio más actualizado
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&cc=es&l=spanish"
    res = requests.get(url)
    data = res.json()
    juego_data = data.get(appid_str, {}).get("data", {})

    if not juego_data:
        return RedirectResponse("/carrito", status_code=302)

    nombre = juego_data.get("name", "Desconocido")
    imagen = juego_data.get("header_image", "/static/default.png")
    precio_raw = juego_data.get("price_overview", {}).get("final_formatted", "Gratis")

    # Convertir precio a número para poder sumar
    try:
        if "gratis" in precio_raw.lower():
            precio = 0.0
        else:
            precio = float(precio_raw.replace("€", "").replace(",", ".").strip())
    except Exception:
        precio = 0.0  # Si falla, consideramos gratis

    request.session["carrito"].append({
        "appid": appid,
        "nombre": nombre,
        "imagen": imagen,
        "precio": precio,
        "precio_str": precio_raw  # guardamos string para mostrar
    })

    return RedirectResponse("/carrito", status_code=302)


def convertir_precio(precio_str):
    precio_str = str(precio_str)
    if "gratis" in precio_str.lower():
        return 0.0
    return float(precio_str.replace("€", "").replace(",", ".").strip())


@app.get("/carrito", response_class=HTMLResponse)
async def mostrar_carrito(request: Request):
    carrito = request.session.get("carrito", [])
    total = sum(convertir_precio(j["precio"]) for j in carrito)
    return templates.TemplateResponse("carrito.html", {
        "request": request,
        "carrito": carrito,
        "total_precio": f"{total:.2f}",
        "avatar": request.session.get("avatar"),
        "nombre": request.session.get("nombre"),
    })


@app.post("/eliminar_del_carrito")
async def eliminar_del_carrito(request: Request, appid: int = Form(...)):
    carrito = request.session.get("carrito", [])
    request.session["carrito"] = [j for j in carrito if j["appid"] != appid]
    return RedirectResponse("/carrito", status_code=302)

@app.get("/gracias", response_class=HTMLResponse)
async def gracias(request: Request):
    compra = request.session.get("compra_realizada", [])
    email = request.session.get("correo", "")
    return templates.TemplateResponse("gracias.html", {
        "request": request,
        "compra": compra,
        "email": email
    })


from fastapi import Form

@app.post("/comprar")
async def procesar_compra(request: Request, email: str = Form(...)):
    carrito = request.session.get("carrito", [])
    total = sum(float(j["precio"]) for j in carrito)

    # Guardar compra para página de agradecimiento
    request.session["compra_realizada"] = carrito
    request.session["carrito"] = []
    request.session["correo"] = email

    # ✉️ Enviar correo
    enviar_correo_confirmacion(email, carrito, total)

    return RedirectResponse("/gracias", status_code=302)


@app.get("/api/recomendacionesmba_carrito")
async def recomendar_mba_para_carrito(request: Request):
    # 1. Obtener juegos del carrito desde la sesión
    carrito = request.session.get("carrito", [])
    appids_carrito = {j["appid"] for j in carrito}

    # 2. Obtener juegos jugados por el usuario desde la API de Steam
    steam_id = request.session.get("steam_id")
    datos = llamar_api_steam(
        "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/",
        {"steamid": steam_id, "include_appinfo": "true"}
    )
    appids_steam = {
        juego["appid"] for juego in datos.get("response", {}).get("games", [])
        if juego.get("playtime_forever", 0) > 0
    } if datos else set()

    # 3. Juegos que hay que evitar (carrito + ya jugados)
    appids_evitar = appids_carrito.union(appids_steam)

    if not appids_carrito:
        return {"recomendaciones": []}

    # 4. Cargar modelos y datos
    matriz_binaria = pd.read_csv("matriz_binaria_filtrada.csv", index_col=0)
    pca = load("modelo_pca.joblib")
    kmeans = load("modelo_kmeans.joblib")
    columnas_appids = pd.read_pickle("appids_entrenados.pkl")

    # 5. Crear vector binario del carrito
    vector_usuario = pd.Series(0, index=columnas_appids)
    vector_usuario[list(appids_carrito & set(columnas_appids))] = 1

    # 6. Reducir y predecir cluster
    vector_reducido = pca.transform([vector_usuario])
    cluster = int(kmeans.predict(vector_reducido)[0])
    print(f"🧠 Carrito asignado al cluster {cluster}")

    # 7. Cargar reglas del cluster
    reglas_path = f"reglas_cluster_{cluster}.csv"
    if not Path(reglas_path).exists():
        return {"recomendaciones": []}
    reglas = pd.read_csv(reglas_path)

    # 8. Evaluar frozensets
    if isinstance(reglas['antecedents'].iloc[0], str):
        if reglas['antecedents'].str.startswith('frozenset').any():
            reglas['antecedents'] = reglas['antecedents'].apply(eval)
            reglas['consequents'] = reglas['consequents'].apply(eval)
        else:
            from ast import literal_eval
            reglas['antecedents'] = reglas['antecedents'].apply(lambda x: frozenset(literal_eval(x)))
            reglas['consequents'] = reglas['consequents'].apply(lambda x: frozenset(literal_eval(x)))

    # 9. Nombres de juegos
    nombres_juegos = pd.read_csv("nombres_juegos.csv").set_index("appid")["name"].to_dict()

    # 10. Reglas aplicables y filtradas
    recomendaciones = []
    for _, fila in reglas.iterrows():
        if fila['antecedents'].issubset(appids_carrito) and not fila['consequents'].issubset(appids_evitar):
            appid = list(fila['consequents'])[0]
            if appid not in appids_evitar:
                recomendaciones.append((appid, fila['confidence'], fila['lift']))

    if not recomendaciones:
        return {"recomendaciones": []}

    # 11. Construir respuesta con imagen + descripción
    recomendaciones = sorted(recomendaciones, key=lambda x: (-x[1], -x[2]))
    vistos = set()
    resultados = []

    for appid, _, _ in recomendaciones:
        if appid in vistos or appid in appids_evitar:
            continue
        vistos.add(appid)

        nombre = nombres_juegos.get(appid, f"Juego {appid}")
        info = obtener_datos_juego(appid)

        resultados.append({
            "item_id": appid,
            "nombre": nombre,
            "imagen": info["imagen"],
            "descripcion": info["descripcion"]
        })

        if len(resultados) >= 10:
            break

    return {"recomendaciones": resultados}


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def enviar_correo_confirmacion(destinatario: str, carrito: list, total: float):
    remitente = "steamia.soporte@gmail.com"
    contraseña = "bsty vcgw wxlc btyi"

    # Crear el contenido del mensaje
    mensaje = MIMEMultipart("alternative")
    mensaje["Subject"] = "Confirmación de compra - Steamia"
    mensaje["From"] = remitente
    mensaje["To"] = destinatario

    cuerpo_html = "<h2>Gracias por tu compra en Steamia 🧠</h2>"
    cuerpo_html += "<ul>"
    for juego in carrito:
        cuerpo_html += f"<li>{juego['nombre']} - {juego['precio_str']}</li>"
    cuerpo_html += "</ul>"
    cuerpo_html += f"<p><strong>Total:</strong> {total:.2f}€</p>"

    mensaje.attach(MIMEText(cuerpo_html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as servidor:
            servidor.login(remitente, contraseña)
            servidor.sendmail(remitente, destinatario, mensaje.as_string())
        print("✅ Correo enviado con éxito.")
    except Exception as e:
        print(f"❌ Error al enviar el correo: {e}")