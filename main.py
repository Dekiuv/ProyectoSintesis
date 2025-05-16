from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os, requests, json
from gensim.models import Word2Vec
import pandas as pd

app = FastAPI()

# Archivos estáticos y plantillas
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Cargar modelo Word2Vec
modelo = Word2Vec.load("modelo_word2vec_mejorado.model")

# Cargar metadatos
df_meta = pd.read_csv("steam_juegos_metadata.csv").fillna("")
df_meta["appid"] = df_meta["appid"].astype(str)
id_to_name = dict(zip(df_meta["appid"], df_meta["name"]))
metadata_dict = df_meta.set_index("appid").to_dict(orient="index")  # claves ya son str

# Cargar popularidad
with open("popularidad.json") as f:
    popularidad = json.load(f)

# API keys desde .env
load_dotenv()
STEAM_KEYS = [
    ("DIEGO", os.getenv("STEAM_API_KEY_DIEGO")),
    ("ALVARO", os.getenv("STEAM_API_KEY_ALVARO")),
    ("ARITZ", os.getenv("STEAM_API_KEY_ARITZ")),
    ("VICTOR", os.getenv("STEAM_API_KEY_VICTOR")),
]

# === Funciones auxiliares ===

def resolver_steam_id(entrada, api_key):
    if entrada.isdigit() and len(entrada) >= 16:
        return entrada
    url = "https://api.steampowered.com/ISteamUser/ResolveVanityURL/v1/"
    try:
        r = requests.get(url, params={"key": api_key, "vanityurl": entrada}, timeout=5)
        data = r.json()
        if data.get("response", {}).get("success") == 1:
            return data["response"]["steamid"]
    except:
        pass
    return None

def obtener_juegos_steam(steam_id, api_key):
    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
    try:
        r = requests.get(url, params={
            "key": api_key,
            "steamid": steam_id,
            "include_played_free_games": True
        }, timeout=5)
        data = r.json()
        juegos = data.get("response", {}).get("games", [])
        return {
            str(j["appid"]): j.get("playtime_forever", 0)
            for j in juegos if j.get("playtime_forever", 0) > 0
        }
    except:
        return None


def recomendar_desde_juegos(juegos_dict, topn=20, alpha=0.7):
    similares = {}
    for item_id, playtime in juegos_dict.items():
        if item_id in modelo.wv:
            peso = min(playtime / 60, 10)  # Ponderación máxima de 10
            try:
                similares_raw = modelo.wv.most_similar(item_id, topn=50)
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



# === Página principal HTML ===
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/recomendar/{usuario_input}")
def recomendar(usuario_input: str):
    for nombre, api_key in STEAM_KEYS:
        steam_id = resolver_steam_id(usuario_input, api_key)
        if not steam_id:
            continue
        juegos_dict = obtener_juegos_steam(steam_id, api_key)
        if juegos_dict is not None:
            juegos_validos = {j: p for j, p in juegos_dict.items() if j in modelo.wv}
            if not juegos_validos:
                return {"mensaje": f"⚠️ El usuario no tiene juegos válidos para el modelo."}
            recomendaciones = recomendar_desde_juegos(juegos_validos, 50, 0.8)
            return {
                "steam_id": steam_id,
                "juegos_validos": len(juegos_validos),
                "api_key_usada": nombre,
                "recomendaciones": recomendaciones
            }
    return {"error": "❌ No se pudo obtener el perfil o está vacío/privado."}

