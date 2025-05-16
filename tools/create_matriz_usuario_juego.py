import pandas as pd

# Configuración
RUTA_ENTRADA = "usuarios_items_balanceado.csv"
RUTA_SALIDA = "matriz_usuario_juego.csv"

print("📥 Cargando datos...")
df = pd.read_csv(RUTA_ENTRADA, dtype={"user_id": str, "item_id": str})

# Pivot a matriz usuario × item
print("🔄 Generando matriz usuario-juego (binaria)...")
matriz = df.pivot_table(index="user_id", columns="item_id", values="valor", fill_value=0)

# Guardar resultado
matriz.to_csv(RUTA_SALIDA)
print(f"✅ Matriz guardada en {RUTA_SALIDA} con shape {matriz.shape}")
