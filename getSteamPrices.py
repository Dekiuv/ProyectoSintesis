import pandas as pd
import requests
import time
from pathlib import Path
from tqdm import tqdm
import csv

METADATA_PATH = Path("data/juegos_metadata.csv")
OMITIDOS_PATH = Path("data/juegos_omitidos.csv")

COLUMNAS = ["appid", "name", "imagen_url", "descripcion", "categorias", "precio", "fecha_lanzamiento"]

def obtener_info_completa_desde_store(appid, max_reintentos=10):
    intentos = 0
    while intentos < max_reintentos:
        try:
            url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=spanish"
            res = requests.get(url, timeout=5)
            if res.status_code == 429:
                espera = 60 + intentos * 10
                print(f"⏳ 429 para {appid}. Esperando {espera} segundos...")
                time.sleep(espera)
                intentos += 1
                continue

            res.raise_for_status()
            data = res.json().get(str(appid), {}).get("data", {})
            if not data:
                return None

            nombre = data.get("name", "").strip()
            descripcion = data.get("short_description", "").strip()
            categorias = [c["description"] for c in data.get("categories", [])]
            imagen_url = data.get("header_image", "")
            fecha = data.get("release_date", {}).get("date", "").strip()

            # Extraer precio
            precio_info = data.get("price_overview")
            if precio_info:
                precio_str = precio_info.get("final_formatted", "0")
                if "gratis" in precio_str.lower():
                    precio = 0.0
                else:
                    precio = float(precio_str.replace("€", "").replace(",", ".").strip())
            else:
                precio = 0.0  # si no hay info, consideramos gratis

            return {
                "appid": str(appid),
                "name": nombre,
                "imagen_url": imagen_url,
                "descripcion": descripcion,
                "categorias": "|".join(categorias),
                "precio": precio,
                "fecha_lanzamiento": fecha
            }

        except requests.RequestException as e:
            print(f"⚠️ Error en {appid}: {e}. Reintentando...")
            time.sleep(5)
            intentos += 1
        except Exception as e:
            print(f"⚠️ Error inesperado en {appid}: {e}")
            break

    print(f"❌ No se pudo obtener info de {appid} tras {max_reintentos} intentos.")
    return None

def actualizar_metadata_juegos(juegos):
    # Cargar metadata actual
    if METADATA_PATH.exists():
        df_existente = pd.read_csv(METADATA_PATH, dtype=str)
    else:
        df_existente = pd.DataFrame(columns=COLUMNAS)

    df_existente["appid"] = df_existente["appid"].astype(str)
    df_existente = df_existente.set_index("appid")

    for juego in tqdm(juegos, desc="📥 Actualizando metadata"):
        appid = str(juego["appid"])
        nueva_info = obtener_info_completa_desde_store(appid)
        if nueva_info is None:
            continue

        # Añadir o actualizar la fila
        df_existente.loc[appid] = nueva_info

        # Guardar tras cada actualización (seguridad)
        df_existente.reset_index().to_csv(METADATA_PATH, index=False, quoting=csv.QUOTE_ALL)

        time.sleep(0.5)  # evitar bloqueos por abuso

if __name__ == "__main__":
    usuarios_df = pd.read_csv("usuarios_steam_detallado.csv")
    appids = usuarios_df["appid"].astype(str).unique()
    juegos_a_actualizar = [{"appid": a, "name": f"Juego {a}"} for a in appids]

    actualizar_metadata_juegos(juegos_a_actualizar)
