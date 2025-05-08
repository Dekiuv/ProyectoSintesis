import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k
import joblib

# Config
RUTA_MATRIZ = "matriz_usuario_juego.csv"
RUTA_MODELO = "modelo_lightfm.pkl"
RUTA_JUEGOS = "item_ids.csv"
EPOCHS = 30  # más entrenamiento

print("📊 Cargando matriz...")
df = pd.read_csv(RUTA_MATRIZ, index_col=0, dtype=str)
df = df.fillna(0).astype(float).astype(int)
matriz = csr_matrix(df.values)

# user_features = matriz
user_features = matriz

# Entrenar el modelo con mejoras
print("🚀 Entrenando modelo LightFM mejorado...")
model = LightFM(
    loss='warp',
    no_components=100,
    item_alpha=1e-6,
    user_alpha=1e-6
)
model.fit(matriz, user_features=user_features, epochs=EPOCHS, num_threads=4)

# Evaluación
print(f"📈 Precision@10: {precision_at_k(model, matriz, user_features=user_features, k=10).mean():.4f}")
print(f"📈 Recall@10:    {recall_at_k(model, matriz, user_features=user_features, k=10).mean():.4f}")

# Guardar modelo y columnas
joblib.dump(model, RUTA_MODELO)
df.columns.to_series().to_csv(RUTA_JUEGOS, index=False)
print("✅ Modelo y juegos guardados correctamente.")
