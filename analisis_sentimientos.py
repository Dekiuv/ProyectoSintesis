import pandas as pd
import re
from transformers import pipeline

# Cargar dataset
df = pd.read_csv("data/User_reviews.csv")
df = df.dropna(subset=['review_text'])

# Limpiar texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+|www.\S+", "", texto)
    texto = re.sub(r"[^a-zA-Z\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

df['review_text'] = df['review_text'].apply(limpiar_texto)

# Cargar modelo
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Procesar en lote
reviews = df['review_text'].tolist()
results = sentiment_analyzer(reviews, truncation=True, batch_size=16)

# Añadir resultados
df['sentiment'] = [r['label'] for r in results]
df['confidence'] = [r['score'] for r in results]

# Guardar resultados
df.to_csv("user_reviews_sentiment.csv", index=False)
print("✅ Análisis completado. Resultados guardados en 'user_reviews_sentiment.csv'")

# Opcional: mostrar resumen
print(df['sentiment'].value_counts())
