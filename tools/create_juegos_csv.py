import pandas as pd

# Ruta del dataset original
RUTA_ORIGINAL = "usuarios_items_cluster.csv"
RUTA_SALIDA = "juegos.csv"

# Cargar el dataset
df = pd.read_csv(RUTA_ORIGINAL, dtype=str)

# Asegurarse de que las columnas necesarias están presentes
if 'item_id' not in df.columns or 'item_name' not in df.columns:
    print("❌ El archivo debe tener columnas 'item_id' y 'item_name'.")
    exit()

# Eliminar duplicados para tener cada juego solo una vez
df_juegos = df[['item_id', 'item_name']].drop_duplicates()

# Guardar como CSV
df_juegos.to_csv(RUTA_SALIDA, index=False)
print(f"✅ Archivo '{RUTA_SALIDA}' generado con {len(df_juegos)} juegos únicos.")
