from dotenv import load_dotenv
import os
import requests
import time
import csv
import json
from collections import deque

# === Cargar API Keys ===
load_dotenv()
STEAM_KEYS = [
    ("DIEGO", os.getenv("STEAM_API_KEY_DIEGO")),
    ("ALVARO", os.getenv("STEAM_API_KEY_ALVARO")),
    ("ARITZ", os.getenv("STEAM_API_KEY_ARITZ")),
    ("VICTOR", os.getenv("STEAM_API_KEY_VICTOR")),
    ("RAUL", os.getenv("STEAM_API_KEY_RAUL")),
]

# === Función para usar múltiples claves de API ===
def llamar_api(url, params, timeout=5):
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

# === Funciones para obtener datos ===
def obtener_juegos(steam_id):
    data = llamar_api(
        "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/",
        {"steamid": steam_id, "include_appinfo": True}
    )
    if not data or not data.get("response"):
        return []
    return data["response"].get("games", [])

def obtener_amigos(steam_id):
    data = llamar_api(
        "https://api.steampowered.com/ISteamUser/GetFriendList/v1/",
        {"steamid": steam_id, "relationship": "friend"}
    )
    if not data or not data.get("friendslist"):
        return []
    return [amigo["steamid"] for amigo in data["friendslist"].get("friends", [])]

# === Función principal ===
def recolectar_datos_steam(steam_id_inicial, max_usuarios=1000, output_csv="usuarios_steam_detallado.csv", estado_guardado="estado_progreso.json"):
    visitados = set()
    cola = deque([steam_id_inicial])
    total = 0

    # Cargar estado previo si existe
    if os.path.exists(estado_guardado):
        with open(estado_guardado, "r", encoding="utf-8") as f:
            estado = json.load(f)
            visitados = set(estado.get("visitados", []))
            cola = deque(estado.get("cola", []))
            total = estado.get("total", 0)
            print(f"🔄 Reanudando desde el estado guardado: {total} usuarios ya procesados.")

    existe = os.path.isfile(output_csv)
    with open(output_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(["steam_id", "appid", "name", "playtime_forever", "playtime_2weeks", "img_icon_url", "has_stats", "has_leaderboards", "content_descriptorids"])

        while cola and total < max_usuarios:
            steam_id = cola.popleft()
            if steam_id in visitados:
                continue

            print(f"🔍 Analizando usuario {steam_id} ({total + 1}/{max_usuarios})...")
            juegos = obtener_juegos(steam_id)
            if juegos:
                for j in juegos:
                    writer.writerow([
                        steam_id,
                        j.get("appid"),
                        j.get("name"),
                        j.get("playtime_forever", 0),
                        j.get("playtime_2weeks", 0),
                        j.get("img_icon_url", ""),
                        j.get("has_community_visible_stats", False),
                        j.get("has_leaderboards", False),
                        ";".join(map(str, j.get("content_descriptorids", [])))
                    ])
                total += 1
            else:
                print(f"⚠️ Usuario {steam_id} sin juegos públicos.")

            amigos = obtener_amigos(steam_id)
            for amigo_id in amigos:
                if amigo_id not in visitados:
                    cola.append(amigo_id)

            visitados.add(steam_id)

            # Guardar estado
            with open(estado_guardado, "w", encoding="utf-8") as ef:
                json.dump({"visitados": list(visitados), "cola": list(cola), "total": total}, ef)

            time.sleep(1.5)  # Para evitar bloqueos por parte de la API

    print(f"✅ Recolección completada. Total usuarios: {total}. Datos guardados en '{output_csv}'.")

# === Ejecutar ===
if __name__ == "__main__":
    steam_id_inicial = input("Introduce el Steam ID inicial: ").strip()
    recolectar_datos_steam(steam_id_inicial, max_usuarios=20000)
