import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy, dump
from surprise.model_selection import train_test_split
import os

# Configuración
CSV_GENERAL = "usuarios_items_cluster.csv"  # Puedes usar el dataset sin clusters si tienes uno original
MODELO_GENERAL_PATH = "svd_modelo_general.pkl"

# Cargar datos
df = pd.read_csv(CSV_GENERAL)
df['user_id'] = df['user_id'].astype(str)
df['item_id'] = df['item_id'].astype(str)

# Asumimos que 'rating' es 1 (si el usuario tiene el juego)
if 'rating' not in df.columns:
    df['rating'] = 1

# Definir Reader para Surprise (valor binario entre 0 y 1)
reader = Reader(rating_scale=(0, 1))

# Cargar datos en el dataset de Surprise
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Entrenar modelo SVD general
modelo_general = SVD()
modelo_general.fit(trainset)

# Calcular métricas
predictions = modelo_general.test(testset)
mae = accuracy.mae(predictions, verbose=False)
rmse = accuracy.rmse(predictions, verbose=False)

def precision_recall_at_k(predictions, k=10, threshold=0.5):
    from collections import defaultdict
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    precisions, recalls = dict(), dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in top_k)
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in top_k)
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k > 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel > 0 else 0
    return sum(precisions.values()) / len(precisions), sum(recalls.values()) / len(recalls)

precision, recall = precision_recall_at_k(predictions, k=10)

print(f"\nModelo General: MAE={mae:.4f} | RMSE={rmse:.4f} | Precision@10={precision:.4f} | Recall@10={recall:.4f}")

# Guardar modelo
dump.dump(MODELO_GENERAL_PATH, algo=modelo_general)
print(f"Modelo general guardado en: {MODELO_GENERAL_PATH}")
