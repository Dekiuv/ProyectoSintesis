import requests
import time

API_KEY = "F5E52AD27E9DC7006A2068AA05B6EE04"
STEAMID = "76561198373413082"

# Función para obtener resumen del perfil
def get_profile(steamid):
    url = "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/"
    params = {"key": API_KEY, "steamids": steamid}
    res = requests.get(url, params=params).json()
    return res["response"]["players"][0]

# Función para obtener juegos
def get_owned_games(steamid):
    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    params = {
        "key": API_KEY,
        "steamid": steamid,
        "include_appinfo": True,
        "include_played_free_games": True,
        "format": "json"
    }
    res = requests.get(url, params=params).json()
    return res["response"].get("games", [])

# Función para obtener amigos
def get_friends(steamid):
    url = "https://api.steampowered.com/ISteamUser/GetFriendList/v0001/"
    params = {"key": API_KEY, "steamid": steamid, "relationship": "friend"}
    res = requests.get(url, params=params).json()
    return [f["steamid"] for f in res.get("friendslist", {}).get("friends", [])]

# Función para obtener nombres de amigos
def get_friend_names(friend_ids):
    names = []
    for i in range(0, len(friend_ids), 100):
        ids_chunk = friend_ids[i:i+100]
        url = "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/"
        params = {"key": API_KEY, "steamids": ",".join(ids_chunk)}
        res = requests.get(url, params=params).json()
        for p in res["response"]["players"]:
            names.append(p["personaname"])
        time.sleep(0.5)
    return names

# Función para obtener logros de un juego
def get_achievements(steamid, appid):
    url = "https://api.steampowered.com/ISteamUserStats/GetPlayerAchievements/v1/"
    params = {"key": API_KEY, "steamid": steamid, "appid": appid}
    res = requests.get(url, params=params)
    if res.status_code != 200:
        return 0
    data = res.json()
    achievements = data.get("playerstats", {}).get("achievements", [])
    return sum(1 for a in achievements if a["achieved"] == 1)

# === EJECUCIÓN DEL PROGRAMA ===

# 1. Perfil del usuario
profile = get_profile(STEAMID)
print(f"\n👤 Nombre: {profile['personaname']}")
print(f"📸 Avatar: {profile['avatarfull']}")
print(f"🌐 Perfil: {profile['profileurl']}")

# 2. Top 5 juegos más jugados
games = get_owned_games(STEAMID)
if not games:
    print("\n⚠️ No se encontraron juegos (perfil privado o sin datos).")
    exit()

top_played = sorted(games, key=lambda g: g["playtime_forever"], reverse=True)[:5]
print("\n🎮 Top 5 juegos más jugados:")
for game in top_played:
    print(f"🔹 {game['name']} {game['appid']} - {game['playtime_forever'] // 60} horas")

# 3. Lista de amigos
friend_ids = get_friends(STEAMID)
if friend_ids:
    print(f"\n👥 Amigos ({len(friend_ids)}):")
    friend_names = get_friend_names(friend_ids)
    for name in friend_names:
        print(f"   • {name}")
else:
    print("\n👥 No se pudieron obtener amigos (perfil privado o sin amigos visibles).")

# 4. Juegos con más logros desbloqueados
print("\n🏆 Juegos con más logros desbloqueados:")
top_logros = []
for game in top_played:
    count = get_achievements(STEAMID, game["appid"])
    top_logros.append((game["name"], count))
top_logros = sorted(top_logros, key=lambda x: x[1], reverse=True)

for name, count in top_logros:
    print(f"🏅 {name}: {count} logros desbloqueados")
