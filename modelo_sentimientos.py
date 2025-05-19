from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Nombre del modelo y ruta donde guardarlo
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
save_path = "modelo_sentimiento"

print("⏳ Descargando modelo...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print("💾 Guardando modelo en disco...")
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print("✅ Modelo descargado y guardado en:", save_path)