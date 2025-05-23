from pathlib import Path
import pandas as pd
import csv
import time
import requests
from tqdm import tqdm

METADATA_PATH = Path("data/juegos_metadata.csv")
OMITIDOS_PATH = Path("data/juegos_omitidos.csv")

def obtener_info_completa_desde_store(appid, max_reintentos=10):
    intentos = 0
    while intentos < max_reintentos:
        try:
            url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=spanish"
            res = requests.get(url, timeout=5)
            if res.status_code == 429:
                espera = 60 + intentos * 10
                print(f"⏳ Petición a {appid} bloqueada (429). Esperando {espera} segundos...")
                time.sleep(espera)
                intentos += 1
                continue

            res.raise_for_status()
            data = res.json().get(str(appid), {}).get("data", {})

            name = data.get("name", "").strip()
            descripcion = data.get("short_description", "").strip()
            categorias = [c["description"] for c in data.get("categories", [])]
            release_date = data.get("release_date", {}).get("date", "").strip()

            # Obtener precio
            precio_raw = data.get("price_overview", {}).get("final_formatted", "")
            if "gratis" in precio_raw.lower():
                precio = 0.0
            elif precio_raw:
                try:
                    precio = float(precio_raw.replace("€", "").replace(",", ".").strip())
                except Exception:
                    precio = None
            else:
                precio = None

            return name, descripcion, "|".join(categorias), precio, release_date

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error en appid {appid}: {e}. Reintentando...")
            time.sleep(5)
            intentos += 1
        except Exception as e:
            print(f"⚠️ Error inesperado con {appid}: {e}")
            break

    print(f"❌ No se pudo obtener info de {appid} tras {max_reintentos} intentos.")
    return None, None, None, None, None

def cargar_omitidos():
    if OMITIDOS_PATH.exists():
        df_omitidos = pd.read_csv(OMITIDOS_PATH, dtype=str)
        return set(df_omitidos["appid"].values)
    return set()

def guardar_juego_omitido(appid, name):
    nuevo = pd.DataFrame([{"appid": appid, "name": name}])
    if OMITIDOS_PATH.exists():
        nuevo.to_csv(OMITIDOS_PATH, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL)
    else:
        nuevo.to_csv(OMITIDOS_PATH, mode='w', header=True, index=False, quoting=csv.QUOTE_ALL)

def actualizar_metadata_juegos(juegos):
    columnas = ["appid", "name", "imagen_url", "descripcion", "categorias", "precio", "fecha_lanzamiento"]

    if METADATA_PATH.exists():
        df_existente = pd.read_csv(METADATA_PATH, dtype=str, quoting=csv.QUOTE_ALL)
        appids_existentes = set(df_existente["appid"].astype(str))
    else:
        df_existente = pd.DataFrame(columns=columnas)
        appids_existentes = set()

    omitidos = cargar_omitidos()

    for juego in tqdm(juegos, desc="📥 Descargando metadata"):
        appid = str(juego["appid"])
        if appid in omitidos or appid in appids_existentes:
            continue

        nombre_real, descripcion, categorias, precio, fecha_lanzamiento = obtener_info_completa_desde_store(appid)
        if descripcion is None:
            continue

        if not descripcion.strip():
            print(f"⏭️ Juego {appid} omitido por no tener descripción.")
            guardar_juego_omitido(appid, juego["name"])
            continue

        nuevo_registro = pd.DataFrame([{
            "appid": appid,
            "name": nombre_real,
            "imagen_url": f"https://cdn.cloudflare.steamstatic.com/steam/apps/{appid}/capsule_616x353.jpg",
            "descripcion": descripcion,
            "categorias": categorias,
            "precio": precio,
            "fecha_lanzamiento": fecha_lanzamiento
        }])

        nuevo_registro.to_csv(
            METADATA_PATH,
            mode='a',
            header=not METADATA_PATH.exists(),
            index=False,
            quoting=csv.QUOTE_ALL
        )
        time.sleep(0.5)

if __name__ == "__main__":
    usuarios_df = pd.read_csv("usuarios_steam_detallado.csv")

    popularidad = usuarios_df.groupby("appid")["user_id"].nunique()
    juegos_populares = popularidad[popularidad >= 5].index

    try:
        metadata_df = pd.read_csv(METADATA_PATH).dropna(subset=["appid"])
        metadata_df["appid"] = metadata_df["appid"].astype(int)
    except FileNotFoundError:
        metadata_df = pd.DataFrame(columns=["appid"])

    appids_usuarios = set(juegos_populares)
    appids_metadata = set(metadata_df["appid"].unique())
    appids_faltantes = list(appids_usuarios - appids_metadata)
    print(f"🔍 Juegos populares sin metadata: {len(appids_faltantes)}")

    juegos_a_completar = [{"appid": appid, "name": f"Juego {appid}"} for appid in appids_faltantes]
    actualizar_metadata_juegos(juegos_a_completar)
