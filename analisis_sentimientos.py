import requests
import re
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ========== CONFIGURACIÓN ==========
API_KEY = "F5E52AD27E9DC7006A2068AA05B6EE04"
RUTA_MODELO = "./modelo_sentimiento"

# ========== LIMPIAR TEXTO ==========
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+|www.\S+", "", texto)
    texto = re.sub(r"[^\w\sáéíóúüñçàèìòùâêîôûäëïöü]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# ========== CARGAR MODELO ==========
print("⏳ Cargando modelo desde disco...")
tokenizer = AutoTokenizer.from_pretrained(RUTA_MODELO)
model = AutoModelForSequenceClassification.from_pretrained(RUTA_MODELO)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print("✅ Modelo cargado.\n")

# ========== BUSCAR APPID POR NOMBRE ==========
def obtener_appid(nombre_juego):
    url = f"https://store.steampowered.com/api/storesearch/?term={nombre_juego}&cc=us&l=en"
    respuesta = requests.get(url).json()
    if respuesta.get("items"):
        return respuesta["items"][0]["id"]
    return None

# ========== OBTENER REVIEWS ==========
def obtener_reviews(appid, cantidad=10):
    url = f"https://store.steampowered.com/appreviews/{appid}?json=1&num_per_page={cantidad}&language=all&filter=recent"
    r = requests.get(url).json()
    comentarios = [review["review"] for review in r.get("reviews", [])]
    return comentarios

# ========== MAIN ==========
while True:
    nombre_juego = input("🎮 Escribe el nombre del juego (o 'salir'): ")
    if nombre_juego.lower() == "salir":
        break

    appid = obtener_appid(nombre_juego)
    if not appid:
        print("❌ No se encontró el juego.\n")
        continue

    comentarios = obtener_reviews(appid, cantidad=10)
    if not comentarios:
        print("❌ No se encontraron comentarios.\n")
        continue

    print(f"\n📝 Mostrando análisis de sentimiento para los 10 comentarios más relevantes de '{nombre_juego}':\n")

    for comentario in comentarios:
        limpio = limpiar_texto(comentario)
        if not limpio.strip():
            continue
        resultado = sentiment_analyzer(limpio, truncation=True)[0]
        estrellas = int(resultado['label'][0])
        sentimiento = "Muy malo" if estrellas == 1 else "Malo" if estrellas == 2 else "Neutral" if estrellas == 3 else "Bueno" if estrellas == 4 else "Muy bueno"
        print(f"🗨️  Comentario: {comentario.strip()[:150]}...")
        print(f"📊 Sentimiento: {sentimiento} ({resultado['label']}) | Confianza: {resultado['score']:.2f}\n")