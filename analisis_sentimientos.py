import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Función de limpieza (mantiene acentos y caracteres multilingües)
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+|www.\S+", "", texto)
    texto = re.sub(r"[^\w\sáéíóúüñçàèìòùâêîôûäëïöü]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# Cargar desde carpeta local
ruta_modelo = "modelo_sentimiento"
print("⏳ Cargando modelo desde disco...")
tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
model = AutoModelForSequenceClassification.from_pretrained(ruta_modelo)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print("✅ Modelo cargado correctamente.\n")

# Interfaz en directo
while True:
    texto = input("📝 Escribe una reseña (o 'salir'): ")
    if texto.lower() == "salir":
        break

    texto_limpio = limpiar_texto(texto)
    resultado = sentiment_analyzer(texto_limpio, truncation=True)[0]
    estrellas = int(resultado['label'][0])
    sentimiento = "Muy malo" if estrellas == 1 else "Malo" if estrellas == 2 else "Neutral" if estrellas == 3 else "Bueno" if estrellas == 4 else "Muy bueno"
    print(f"\n📊 Sentimiento: {sentimiento} ({resultado['label']}) | Confianza: {resultado['score']:.2f}\n")
