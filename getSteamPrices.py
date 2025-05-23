import pandas as pd
import requests
import time
from pathlib import Path
import csv
from tqdm import tqdm

METADATA_PATH = Path("data/juegos_metadata.csv")
COLUMNAS = ["appid", "name", "imagen_url", "descripcion", "categorias", "precio", "fecha_lanzamiento"]

def obtener_info_completa_desde_store(appid, max_reintentos=10):
    intentos = 0
    while intentos < max_reintentos:
        try:
            url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=spanish"
            res = requests.get(url, timeout=5)

            if res.status_code == 429:
                espera = 60 + intentos * 10
                print(f"⏳ 429 bloqueado ({appid}). Esperando {espera} segundos...")
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

            # Precio
            precio_info = data.get("price_overview")
            if precio_info:
                precio_str = precio_info.get("final_formatted", "0")
                if "gratis" in precio_str.lower():
                    precio = 0.0
                else:
                    precio = float(precio_str.replace("€", "").replace(",", ".").strip())
            else:
                precio = 0.0

            return {
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
            print(f"⚠️ Error inesperado con {appid}: {e}")
            break

    print(f"❌ No se pudo obtener info de {appid} tras {max_reintentos} intentos.")
    return None

def actualizar_solo_existentes():
    if not METADATA_PATH.exists():
        print("❌ El archivo metadata no existe.")
        return

    df = pd.read_csv(METADATA_PATH, dtype=str)
    df["appid"] = df["appid"].astype(str)
    df = df.set_index("appid")

    for appid in tqdm(df.index, desc="🔄 Actualizando juegos ya existentes"):
        nueva_info = obtener_info_completa_desde_store(appid)
        if nueva_info is None:
            continue

        for col, valor in nueva_info.items():
            df.at[appid, col] = valor

        # Guardar tras cada juego
        df.reset_index().to_csv(METADATA_PATH, index=False, quoting=csv.QUOTE_ALL)
        time.sleep(0.5)

if __name__ == "__main__":
    actualizar_solo_existentes()
