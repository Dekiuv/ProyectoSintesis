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
app.add_middleware(SessionMiddleware, secret_key="clave_secreta", https_only=False, same_site="lax")

# Inicio aplicación
app.mount("/static", StaticFiles(directory="static"), name="biblioteca")
templates = Jinja2Templates(directory="templates")

load_dotenv()
STEAM_KEYS = [
    # ("DIEGO", os.getenv("STEAM_API_KEY_DIEGO")),
    ("ALVARO", os.getenv("STEAM_API_KEY_ALVARO")),
    ("ARITZ", os.getenv("STEAM_API_KEY_ARITZ")),
    ("VICTOR", os.getenv("STEAM_API_KEY_VICTOR")),
    ("RAUL", os.getenv("STEAM_API_KEY_RAUL")),
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
@app.get("/debug-session")
async def debug_session(request: Request):
    return JSONResponse(content=dict(request.session))

@app.post("/login")
async def login(request: Request, steam_id: str = Form(...)):
    print("🧪 steam_id recibido:", steam_id)

    datos = llamar_api_steam(
        "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/",
        {"steamid": steam_id, "include_appinfo": "true"}
    )
    # print("📦 Respuesta de GetOwnedGames:", datos)

    if not datos or not datos.get("response") or datos["response"].get("game_count", 0) == 0:
        return templates.TemplateResponse(
            "iniciosesion.html",
            {"request": request, "error": "❌ Steam ID no válido o sin juegos."},
            status_code=400
        )

    juegos = datos["response"].get("games", [])
    
    # ✅ Guardamos metadata en CSV
    actualizar_metadata_juegos(juegos)

    # ✅ Guardamos solo appids en la sesión
    request.session["appids_usuario"] = [j["appid"] for j in juegos]

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

    # ✅ Guardar info de sesión
    request.session["steam_id"] = steam_id
    request.session["avatar"] = avatar
    request.session["nombre"] = nombre

    # ✅ Redirigir
    return RedirectResponse(url="/tienda", status_code=302)


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
    global juegos_cache  # ✅ Declaración al inicio
    steam_id = request.session.get("steam_id")
    appids_usuario = request.session.get("appids_usuario")

    if not steam_id:
        return JSONResponse(content={"error": "No autorizado"}, status_code=403)

    juegos_info = []

    # ✅ Si tenemos appids en sesión, los usamos
    if appids_usuario:
        print("✅ Usando appids_usuario desde sesión")
        try:
            df_meta = pd.read_csv("data/juegos_metadata.csv").dropna()
            df_meta["appid"] = df_meta["appid"].astype(int)
            df_filtrados = df_meta[df_meta["appid"].isin(appids_usuario)]

            for _, row in df_filtrados.iterrows():
                juegos_info.append({
                    "appid": row["appid"],
                    "nombre": row["name"],
                    "imagen": row["imagen_url"],
                    "descripcion": row["descripcion"],
                    "categorias": row.get("categorias", "").split("|") if pd.notna(row.get("categorias", "")) else []
                })

            juegos_cache = juegos_info
            return JSONResponse(content=juegos_info)
        except Exception as e:
            print(f"⚠️ Error al cargar desde CSV: {e}")
            # Si falla, seguimos con llamada a la API

    # 🔄 Si no hay datos en sesión o el CSV ha fallado
    print("🔄 Llamando a Steam API para obtener juegos.")
    datos = llamar_api_steam(
        "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/",
        {"steamid": steam_id, "include_appinfo": "true"}
    )

    juegos = datos.get("response", {}).get("games", []) if datos else []
    appids = [j["appid"] for j in juegos]
    request.session["appids_usuario"] = appids  # 🔐 Guardamos en la sesión

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

    juegos_cache = juegos_info  # ✅ Se actualiza correctamente
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

def obtener_juegos_usuario_con_tiempo(steam_id, appids_usuario=None):
    if appids_usuario:
        print("✅ Usando appids de la sesión")
        # Llamamos a la API para obtener tiempos actualizados, pero usamos solo los appids del usuario
        datos = llamar_api_steam(
            "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/",
            {"steamid": steam_id, "include_appinfo": "false"}
        )
        juegos = datos.get("response", {}).get("games", []) if datos else []
        return {
            str(j["appid"]): j.get("playtime_forever", 0)
            for j in juegos
            if j.get("playtime_forever", 0) > 0 and j["appid"] in appids_usuario
        }

    else:
        print("🔄 No hay appids en sesión. Haciendo llamada completa a la API con appinfo=true")
        datos = llamar_api_steam(
            "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/",
            {"steamid": steam_id, "include_appinfo": "true"}
        )
        juegos = datos.get("response", {}).get("games", []) if datos else []
        return {
            str(j["appid"]): j.get("playtime_forever", 0)
            for j in juegos
            if j.get("playtime_forever", 0) > 0
        }

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


def recomendar_juegos_word2vec_con_nombres_unificado(modelo, steam_id, appids_usuario=None, topn=30):
    METADATA_PATH = Path("data/juegos_metadata.csv")
    metadata_df = pd.read_csv(METADATA_PATH).dropna()
    metadata_df["appid"] = metadata_df["appid"].astype(str)

    appid_to_name = metadata_df.set_index("appid")["name"].to_dict()
    appid_to_img = metadata_df.set_index("appid")["imagen_url"].to_dict()
    appids_csv = set(appid_to_name.keys())

    # Paso 1: Obtener appids del usuario
    if appids_usuario:
        print("✅ Usando appids_usuario desde sesión")
        appids = [str(a) for a in appids_usuario if isinstance(a, (int, str))]
    else:
        appids = []
        for nombre, clave in STEAM_KEYS:
            if not clave:
                continue
            try:
                response = requests.get(
                    "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/",
                    params={
                        "key": clave,
                        "steamid": steam_id,
                        "include_appinfo": 1,
                        "format": "json"
                    }
                )
                response.raise_for_status()
                juegos_usuario = response.json().get("response", {}).get("games", [])
                appids = [str(j["appid"]) for j in juegos_usuario if j.get("playtime_forever", 0) > 0]
                break
            except Exception:
                continue
        if not appids:
            return [], [], "❌ No se pudieron obtener los juegos del usuario."

    # Paso 2: Filtrar appids válidos
    appids_validos = [a for a in appids if a in modelo.wv.key_to_index]
    ignorados = len(appids) - len(appids_validos)
    jugados_nombres = [appid_to_name.get(a, f"Unknown ({a})") for a in appids_validos]

    if not appids_validos:
        return jugados_nombres, [], "⚠️ Ningún juego jugado está en el vocabulario del modelo."

    # Paso 3: Generar recomendaciones
    try:
        recomendaciones = modelo.wv.most_similar(appids_validos, topn=topn * 2)
        recomendados = []
        nuevos_juegos = []

        for appid, similitud in recomendaciones:
            if appid in appids_validos:
                continue

            if appid in appid_to_name:
                nombre = appid_to_name.get(appid)
                imagen = appid_to_img.get(appid)
                descripcion = None
            else:
                # Obtener desde la Steam Store
                try:
                    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=spanish"
                    res = requests.get(url, timeout=5)
                    res.raise_for_status()
                    datos = res.json()
                    info = datos.get(str(appid), {}).get("data", {})

                    nombre = info.get("name", f"Juego {appid}")
                    imagen = info.get("header_image")
                    descripcion = info.get("short_description", None)

                    # Guardar en el CSV de forma segura usando la función común
                    actualizar_metadata_juegos([{
                        "appid": appid,
                        "name": nombre,
                        "img_icon_url": None  # si tu función lo ignora, no pasa nada
                    }])

                    appid_to_name[appid] = nombre
                    appid_to_img[appid] = imagen

                except Exception as e:
                    print(f"⚠️ No se pudo obtener info del juego {appid}: {e}")
                    continue

            recomendados.append({
                "item_id": int(appid),
                "nombre": nombre,
                "score": round(similitud, 4),
                "metacritic": None,
                "imagen": imagen,
                "descripcion": descripcion
            })

            if len(recomendados) >= topn:
                break

        # Paso 4: Guardar nuevos juegos en el CSV
        if nuevos_juegos:
            nuevos_df = pd.DataFrame(nuevos_juegos)
            nuevos_df.to_csv(METADATA_PATH, mode="a", header=False, index=False)

    except Exception as e:
        return jugados_nombres, [], f"❌ Error al generar recomendaciones: {e}"

    mensaje = "✅ Recomendaciones generadas correctamente"
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
    for entry in soup.select(".search_result_row")[:5]:
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
    appids_usuario = request.session.get("appids_usuario")

    if not steam_id:
        return RedirectResponse("/", status_code=302)

    _, recomendaciones, _ = recomendar_juegos_word2vec_con_nombres_unificado(modelo, steam_id, appids_usuario)
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
    steam_id = request.session.get("steam_id")
    appids_usuario = request.session.get("appids_usuario")

    # Si no tenemos appids guardados, los buscamos en la API y los guardamos en sesión
    if not appids_usuario:
        print("🔄 No hay appids en sesión. Llamando a la API de Steam.")
        datos = llamar_api_steam(
            "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/",
            {"steamid": steam_id, "format": "json"}
        )
        if not datos or "games" not in datos.get("response", {}):
            print("❌ Error al obtener los juegos del usuario desde la API.")
            return {"recomendaciones": []}
        
        juegos_usuario = datos["response"]["games"]
        appids_usuario = [j["appid"] for j in juegos_usuario if j.get("playtime_forever", 0) > 0]
        request.session["appids_usuario"] = appids_usuario  # guardamos para la próxima

    appids_usuario = set(appids_usuario)

    # Paso 2: Cargar matriz binaria y modelos
    matriz_binaria = pd.read_csv("matriz_binaria_filtrada.csv", index_col=0)
    pca = load("modelo_pca.joblib")
    kmeans = load("modelo_kmeans.joblib")
    columnas_appids = pd.read_pickle("appids_entrenados.pkl")

    # Paso 3: Crear vector binario
    vector_usuario = pd.Series(0, index=columnas_appids)
    vector_usuario[list(appids_usuario & set(columnas_appids))] = 1

    # Paso 4: Cluster
    vector_reducido = pca.transform([vector_usuario])
    cluster = int(kmeans.predict(vector_reducido)[0])
    print(f"🧠 Usuario asignado al cluster {cluster}")

    # Paso 5: Cargar reglas
    reglas_path = f"reglas_cluster_{cluster}.csv"
    if not Path(reglas_path).exists():
        print("⚠️ No hay reglas guardadas para este cluster.")
        return {"recomendaciones": []}

    reglas = pd.read_csv(reglas_path)

    # Paso 6: Evaluar correctamente los frozensets
    if isinstance(reglas['antecedents'].iloc[0], str):
        if reglas['antecedents'].str.startswith('frozenset').any():
            reglas['antecedents'] = reglas['antecedents'].apply(eval)
            reglas['consequents'] = reglas['consequents'].apply(eval)
        else:
            reglas['antecedents'] = reglas['antecedents'].apply(lambda x: frozenset(literal_eval(x)))
            reglas['consequents'] = reglas['consequents'].apply(lambda x: frozenset(literal_eval(x)))

    # Paso 7: Cargar metadata local
    metadata_df = pd.read_csv("data/juegos_metadata.csv").dropna()
    metadata_df["appid"] = metadata_df["appid"].astype(int)
    metadata_dict = metadata_df.set_index("appid").to_dict(orient="index")

    # Paso 8: Aplicar reglas
    recomendaciones = []
    for _, fila in reglas.iterrows():
        if fila['antecedents'].issubset(appids_usuario) and not fila['consequents'].issubset(appids_usuario):
            appid = list(fila['consequents'])[0]
            recomendaciones.append((appid, fila['confidence'], fila['lift']))

    if not recomendaciones:
        print("🤷 No se encontraron recomendaciones para este usuario.")
        return {"recomendaciones": []}

    recomendaciones = sorted(recomendaciones, key=lambda x: (-x[1], -x[2]))
    vistos = set()
    resultados = []

    for appid, _, _ in recomendaciones:
        if appid in vistos:
            continue
        vistos.add(appid)

        info = metadata_dict.get(appid)
        if not info:
            continue  # saltar si no está en el CSV

        resultados.append({
            "item_id": appid,
            "nombre": info["name"],
            "imagen": info["imagen_url"],
            "descripcion": info["descripcion"]
        })

        if len(resultados) >= 10:
            break

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

    if not appids_carrito:
        return {"recomendaciones": []}

    # 2. Obtener juegos jugados por el usuario desde la sesión o la API
    steam_id = request.session.get("steam_id")
    appids_usuario = request.session.get("appids_usuario")

    if not appids_usuario:
        print("🔄 No hay appids en sesión. Llamando a la API.")
        datos = llamar_api_steam(
            "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/",
            {"steamid": steam_id, "include_appinfo": "true"}
        )
        appids_usuario = [
            j["appid"] for j in datos.get("response", {}).get("games", [])
            if j.get("playtime_forever", 0) > 0
        ] if datos else []
        request.session["appids_usuario"] = appids_usuario

    appids_usuario = set(appids_usuario)
    appids_evitar = appids_usuario.union(appids_carrito)

    # 3. Cargar modelos
    matriz_binaria = pd.read_csv("matriz_binaria_filtrada.csv", index_col=0)
    pca = load("modelo_pca.joblib")
    kmeans = load("modelo_kmeans.joblib")
    columnas_appids = pd.read_pickle("appids_entrenados.pkl")

    # 4. Crear vector binario del usuario (basado en sus juegos jugados)
    vector_usuario = pd.Series(0, index=columnas_appids)
    vector_usuario[list(appids_usuario & set(columnas_appids))] = 1

    # 5. Reducir y predecir cluster del usuario
    vector_reducido = pca.transform([vector_usuario])
    cluster = int(kmeans.predict(vector_reducido)[0])
    print(f"🧠 Usuario asignado al cluster {cluster}")

    # 6. Cargar reglas del cluster
    reglas_path = f"reglas_cluster_{cluster}.csv"
    if not Path(reglas_path).exists():
        return {"recomendaciones": []}
    reglas = pd.read_csv(reglas_path)

    # 7. Evaluar frozensets
    if isinstance(reglas['antecedents'].iloc[0], str):
        if reglas['antecedents'].str.startswith('frozenset').any():
            reglas['antecedents'] = reglas['antecedents'].apply(eval)
            reglas['consequents'] = reglas['consequents'].apply(eval)
        else:
            reglas['antecedents'] = reglas['antecedents'].apply(lambda x: frozenset(literal_eval(x)))
            reglas['consequents'] = reglas['consequents'].apply(lambda x: frozenset(literal_eval(x)))

    # 8. Cargar metadata local
    metadata_df = pd.read_csv("data/juegos_metadata.csv").dropna()
    metadata_df["appid"] = metadata_df["appid"].astype(int)
    metadata_dict = metadata_df.set_index("appid").to_dict(orient="index")

    # 9. Aplicar reglas con base en los juegos del carrito
    recomendaciones = []
    for _, fila in reglas.iterrows():
        if fila['antecedents'].issubset(appids_carrito) and not fila['consequents'].issubset(appids_evitar):
            appid = list(fila['consequents'])[0]
            if appid not in appids_evitar:
                recomendaciones.append((appid, fila['confidence'], fila['lift']))

    if not recomendaciones:
        return {"recomendaciones": []}

    # 10. Construir respuesta
    recomendaciones = sorted(recomendaciones, key=lambda x: (-x[1], -x[2]))
    vistos = set()
    resultados = []

    for appid, _, _ in recomendaciones:
        if appid in vistos or appid in appids_evitar:
            continue
        vistos.add(appid)

        info = metadata_dict.get(appid)
        if not info:
            continue  # saltar si no está en el CSV

        resultados.append({
            "item_id": appid,
            "nombre": info["name"],
            "imagen": info["imagen_url"],
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



from pathlib import Path
import pandas as pd
import csv
import time
import requests

METADATA_PATH = Path("data/juegos_metadata.csv")
OMITIDOS_PATH = Path("data/juegos_omitidos.csv")

def obtener_info_completa_desde_store(appid):
    try:
        url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=spanish"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json().get(str(appid), {}).get("data", {})
        descripcion = data.get("short_description", "").strip()
        categorias = [c["description"] for c in data.get("categories", [])]
        return descripcion, "|".join(categorias)
    except requests.exceptions.HTTPError as e:
        if res.status_code == 429:
            print(f"⏳ Petición a {appid} bloqueada por exceso de llamadas (429). Reintenta más tarde.")
        else:
            print(f"⚠️ No se pudo obtener info de {appid}: {e}")
        return None, None
    except Exception as e:
        print(f"⚠️ Error inesperado con {appid}: {e}")
        return None, None

def cargar_omitidos():
    if OMITIDOS_PATH.exists():
        df_omitidos = pd.read_csv(OMITIDOS_PATH, dtype=str)
        return set(df_omitidos["appid"].values)
    return set()

def guardar_juego_omitido(appid, name):
    if OMITIDOS_PATH.exists():
        df_omitidos = pd.read_csv(OMITIDOS_PATH, dtype=str)
    else:
        df_omitidos = pd.DataFrame(columns=["appid", "name"])

    if appid not in df_omitidos["appid"].values:
        nuevo = pd.DataFrame([{"appid": appid, "name": name}])
        df_omitidos = pd.concat([df_omitidos, nuevo], ignore_index=True)
        df_omitidos.to_csv(OMITIDOS_PATH, index=False, quoting=csv.QUOTE_ALL)

def actualizar_metadata_juegos(juegos):
    columnas = ["appid", "name", "imagen_url", "descripcion", "categorias"]

    if METADATA_PATH.exists():
        df_existente = pd.read_csv(METADATA_PATH, dtype=str, quoting=csv.QUOTE_ALL).drop_duplicates("appid", keep="last")
        for col in columnas:
            if col not in df_existente.columns:
                df_existente[col] = ""
    else:
        df_existente = pd.DataFrame(columns=columnas)

    omitidos = cargar_omitidos()
    nuevos_registros = []

    for juego in juegos:
        appid = str(juego["appid"])

        if appid in omitidos:
            print(f"⛔ AppID {appid} ya estaba omitido. No se intenta de nuevo.")
            continue

        if appid in df_existente["appid"].values:
            fila = df_existente[df_existente["appid"] == appid].iloc[0]
            if not str(fila.get("descripcion", "")).strip():
                descripcion, categorias = obtener_info_completa_desde_store(appid)
                if descripcion:
                    df_existente.loc[df_existente["appid"] == appid, "descripcion"] = descripcion
                    df_existente.loc[df_existente["appid"] == appid, "categorias"] = categorias
                time.sleep(0.5)
            continue

        descripcion, categorias = obtener_info_completa_desde_store(appid)
        if descripcion is None:
            continue  # Error temporal → ignorar

        if not descripcion.strip():
            print(f"⏭️ Juego {appid} omitido por no tener descripción.")
            guardar_juego_omitido(appid, juego["name"])
            continue

        nuevos_registros.append({
            "appid": appid,
            "name": juego["name"],
            "imagen_url": f"https://cdn.cloudflare.steamstatic.com/steam/apps/{appid}/capsule_616x353.jpg",
            "descripcion": descripcion,
            "categorias": categorias
        })
        time.sleep(0.5)

    if nuevos_registros:
        df_nuevos = pd.DataFrame(nuevos_registros)
        df_actualizado = pd.concat([df_existente, df_nuevos], ignore_index=True)
        df_actualizado = df_actualizado.drop_duplicates("appid", keep="last")
    else:
        df_actualizado = df_existente

    df_actualizado.to_csv(METADATA_PATH, index=False, quoting=csv.QUOTE_ALL)






def obtener_juegos_usuario_desde_csv(appids_usuario):
    df = pd.read_csv("datos/juegos_metadata.csv")
    return df[df["appid"].isin(appids_usuario)].to_dict(orient="records")
