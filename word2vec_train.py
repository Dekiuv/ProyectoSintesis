import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords

# === Descargar stopwords ===
nltk.download("stopwords")
stopwords_es = set(stopwords.words("spanish"))
stopwords_en = set(stopwords.words("english"))

# === Parámetros de limpieza ===
TIEMPO_MIN_JUGADO = 120
TIEMPO_ULTIMAS_2_SEMANAS = 120
MIN_USUARIOS_POR_JUEGO = 100
MIN_JUEGOS_POR_USUARIO = 3

# === AppIDs equivalentes ===
EQUIVALENCIAS_APPIDS = {
    "3240220": "271590",  # GTA V Enhanced → Legacy
}

# === Cargar datasets ===
df_usuarios = pd.read_csv("usuarios_steam_detallado.csv")
df_meta = pd.read_csv("data/juegos_metadata.csv").dropna(subset=["descripcion"])

# Convertir a string y normalizar equivalencias
df_usuarios["appid"] = df_usuarios["appid"].astype(str).replace(EQUIVALENCIAS_APPIDS)
df_meta["appid"] = df_meta["appid"].astype(str).replace(EQUIVALENCIAS_APPIDS)

# === Eliminar juegos problemáticos ===
JUEGOS_EXCLUIDOS = {"1366800", "629520", "438100", "431960", "2507950", "1281930", "714010"}
df_usuarios = df_usuarios[~df_usuarios["appid"].isin(JUEGOS_EXCLUIDOS)].copy()

# === Filtrar por tiempo jugado ===
df = df_usuarios[df_usuarios["playtime_forever"] > TIEMPO_MIN_JUGADO].copy()

# Eliminar juegos jugados por pocos usuarios
juegos_validos = df['appid'].value_counts()
juegos_validos = juegos_validos[juegos_validos >= MIN_USUARIOS_POR_JUEGO].index
df = df[df['appid'].isin(juegos_validos)]

# Eliminar usuarios con pocos juegos
usuarios_validos = df['steam_id'].value_counts()
usuarios_validos = usuarios_validos[usuarios_validos >= MIN_JUEGOS_POR_USUARIO].index
df = df[df['steam_id'].isin(usuarios_validos)]

# === Popularidad normalizada ===
popularidad = df['appid'].value_counts(normalize=True).to_dict()
min_pop, max_pop = min(popularidad.values()), max(popularidad.values())
rango = max_pop - min_pop if max_pop != min_pop else 1
popularidad_normalizada = {
    appid: 0.5 + (valor - min_pop) / rango
    for appid, valor in popularidad.items()
}

# === Limpieza de texto con stopwords y sin números ===
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+|www.\S+", "", texto)
    texto = re.sub(r"[^\w\sáéíóúüñçàèìòùâêîôûäëïöü]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    palabras = texto.split()
    palabras_filtradas = [
        p for p in palabras
        if p not in stopwords_es and p not in stopwords_en and not p.isnumeric()
    ]
    return " ".join(palabras_filtradas)

df_meta["descripcion"] = df_meta["descripcion"].astype(str).apply(limpiar_texto)
df_meta["categorias"] = df_meta["categorias"].astype(str).apply(
    lambda x: [c.strip().lower() for c in x.split("|") if c.strip() and not c.strip().isnumeric()]
)

df_meta = df_meta.drop_duplicates(subset="appid", keep="first")
metadata_dict = df_meta.set_index("appid")[["descripcion", "categorias"]].to_dict(orient="index")

# === Detectar juegos gratuitos ===
FREE_KEYWORDS = {"free to play", "free-to-play", "f2p", "juego gratuito", "gratuito", "gratis"}

def es_juego_gratuito(appid, meta):
    if appid not in meta:
        return False
    texto = meta[appid]["descripcion"] + " " + " ".join(meta[appid]["categorias"])
    return any(k in texto.lower() for k in FREE_KEYWORDS)

# === Crear secuencias para Word2Vec (excluyendo juegos gratuitos) ===
user_sequences_weighted = []

for steam_id, group in tqdm(df.groupby("steam_id"), desc="🔁 Generando secuencias sin juegos gratuitos"):
    juegos = []

    for _, row in group.iterrows():
        appid = row["appid"]

        # ❌ Excluir juegos gratuitos por completo
        if es_juego_gratuito(appid, metadata_dict):
            continue

        playtime = row["playtime_forever"]
        playtime_2w = row["playtime_2weeks"]

        base_peso = min(max(int(playtime // 60), 1), 10)
        pop_factor = popularidad_normalizada.get(appid, 1.0)
        recent_factor = 1.0 + min(playtime_2w / 1800, 0.5)
        peso = int(base_peso * pop_factor * recent_factor)

        contexto = []
        if appid in metadata_dict:
            contexto += metadata_dict[appid]["descripcion"].split()
            contexto += metadata_dict[appid]["categorias"]

        juegos.extend([appid] * peso)
        juegos.extend(contexto)

    if len(juegos) >= 5:
        user_sequences_weighted.append(juegos)

print(f"📦 Secuencias finales (sin juegos gratuitos): {len(user_sequences_weighted)}")

# === Entrenar el modelo Word2Vec ===
modelo = Word2Vec(
    sentences=user_sequences_weighted,
    vector_size=100,
    window=10,
    min_count=2,
    workers=4,
    sg=1,
    epochs=25
)

modelo.save("word2vec_steam_semantico.model")
print("✅ Modelo guardado como 'word2vec_steam_semantico.model'")
