import pandas as pd
from transformers import pipeline

# Cargar dataset
df = pd.read_csv("australian_user_reviews.csv")

# Eliminar nulos en la columna de texto
df = df.dropna(subset=['review_text'])

# Inicializar pipeline de análisis de sentimientos
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Aplicar análisis a cada reseña
def obtener_sentimiento(texto):
    resultado = sentiment_analyzer(texto, truncation=True)[0]
    return resultado['label'], resultado['score']

# Aplicar a todas las reseñas
df[['sentiment', 'confidence']] = df['review_text'].apply(lambda x: pd.Series(obtener_sentimiento(x)))

# Guardar resultados en un nuevo archivo
df.to_csv("user_reviews_sentiment.csv", index=False)

print("✅ Análisis completado. Resultados guardados en 'user_reviews_sentiment.csv'")
