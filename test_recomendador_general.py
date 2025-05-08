import pandas as pd
from surprise import dump
import os

# ---------------------------
# CONFIGURACIÓN
# ---------------------------
RUTA_MODELO = "svd_modelo_general.pkl"
RUTA_DATASET = "usuarios_items_cluster.csv"
TOP_N = 10

# ---------------------------
# Pedir ID de usuario
# ---------------------------
USER_ID = input("🔎 Introduce el user_id para obtener recomendaciones: ").strip()

# ---------------------------
# Cargar modelo entrenado
# ---------------------------
print("📦 Cargando modelo...")
_, model = dump.load(RUTA_MODELO)

# ---------------------------
# Cargar datos
# ---------------------------
print("📊 Cargando datos...")
df = pd.read_csv(RUTA_DATASET)

# Asegurar formato correcto
df['user_id'] = df['user_id'].astype(str)
df['item_id'] = df['item_id'].astype(str)

# Obtener juegos únicos con nombre
juegos = df[['item_id', 'item_name']].drop_duplicates()

# Obtener juegos ya jugados por el usuario
juegos_usuario = df[df['user_id'] == USER_ID]['item_id'].tolist()

if not juegos_usuario:
    print(f"⚠️ El usuario {USER_ID} no tiene juegos registrados en el dataset.")
    exit()

# Filtrar juegos que no ha jugado
juegos_no_jugados = juegos[~juegos['item_id'].isin(juegos_usuario)]

# Predecir para juegos no jugados
print(f"🤖 Generando recomendaciones para el usuario {USER_ID}...")
predicciones = []
for _, row in juegos_no_jugados.iterrows():
    item_id = row['item_id']
    item_name = row['item_name']
    pred = model.predict(USER_ID, item_id)
    predicciones.append((item_id, item_name, pred.est))

# Ordenar y mostrar top N
top_recomendaciones = sorted(predicciones, key=lambda x: x[2], reverse=True)[:TOP_N]

print(f"\n🎮 Recomendaciones para el usuario {USER_ID}:")
for i, (item_id, item_name, score) in enumerate(top_recomendaciones, 1):
    print(f"{i}. {item_name} (id: {item_id}) → Predicción: {score:.4f}")
