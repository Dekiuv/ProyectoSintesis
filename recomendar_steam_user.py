import requests
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from lightfm import LightFM
import joblib

# Configuración
API_KEYS = [
    "F5E52AD27E9DC7006A2068AA05B6EE04",
    "F8A4003EBB12D2357E82A7D7ED29F694"
]
URL_API = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
RUTA_MODELO = "modelo_lightfm.pkl"
RUTA_JUEGOS = "juegos.csv"
TOP_N = 10

# --- Paso 1: Obtener juegos de Steam ---
steam_id = input("🎮 Introduce el SteamID64 del usuario: ").strip()

for i, key in enumerate(API_KEYS, 1):
    print(f"🔎 Probando con API Key {i}...")
    params = {
        'key': key,
        'steamid': steam_id,
        'format': 'json',
        'include_appinfo': False,
        'include_played_free_games': True
    }
    response = requests.get(URL_API, params=params)

    if response.status_code == 200 and response.text.strip():
        try:
            data = response.json()
            break
        except Exception:
            print(f"❌ Error al parsear JSON con API Key {i}")
            continue
    else:
        print(f"⚠️ API Key {i} falló. Código: {response.status_code}")

else:
    print("❌ No se pudo obtener una respuesta válida con ninguna API Key.")
    exit()

if "response" not in data or "games" not in data["response"]:
    print("❌ No se encontraron juegos para este usuario.")
    exit()

# Juegos jugados por el usuario
juegos_steam = [str(game["appid"]) for game in data["response"]["games"]]
print(f"✅ Juegos obtenidos: {len(juegos_steam)}")

# --- Paso 2: Filtrar juegos conocidos y generar user_features ---
df_items = pd.read_csv(RUTA_JUEGOS, dtype=str)
item_ids = df_items["item_id"].tolist()

juegos_filtrados = [j for j in juegos_steam if j in item_ids]
print(f"🎯 Juegos válidos para el modelo: {len(juegos_filtrados)}")

vector = np.zeros((1, len(item_ids)))
for idx, item_id in enumerate(item_ids):
    if item_id in juegos_filtrados:
        vector[0, idx] = 1

user_features = csr_matrix(vector)

# --- Paso 3: Cargar modelo y predecir ---
print("🤖 Generando recomendaciones...")
modelo = joblib.load(RUTA_MODELO)
scores = modelo.predict(0, np.arange(len(item_ids)), user_features=user_features)

top_idx = np.argsort(-scores)[:TOP_N]

print("\n🎯 Recomendaciones:")
for i, idx in enumerate(top_idx, 1):
    appid = item_ids[idx]
    nombre = df_items.loc[df_items['item_id'] == appid, 'item_name'].values
    nombre = nombre[0] if len(nombre) > 0 else "Nombre desconocido"
    print(f"{i}. {nombre} (AppID: {appid}) → Score: {scores[idx]:.4f}")
