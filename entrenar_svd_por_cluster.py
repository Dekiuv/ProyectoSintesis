import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy, dump
from surprise.model_selection import train_test_split
from collections import defaultdict
import os

RUTA_CSV = "usuarios_items_cluster.csv"
DIR_MODELOS = "modelos_svd"
RUTA_METRICAS = "metricas_svd_clusters.txt"
os.makedirs(DIR_MODELOS, exist_ok=True)

def precision_recall_at_k(predictions, k=10, threshold=0.5):
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

print("📥 Cargando datos...")
df = pd.read_csv(RUTA_CSV)
df['user_id'] = df['user_id'].astype(str)
df['item_id'] = df['item_id'].astype(str)
dfs_por_cluster = {
    cluster: g[['user_id', 'item_id']].assign(rating=1)
    for cluster, g in df.groupby('cluster')
}

reader = Reader(rating_scale=(0, 1))
resultados = []

for cluster_id, df_cluster in dfs_por_cluster.items():
    print(f"\n🚀 Entrenando SVD para cluster {cluster_id} ({len(df_cluster)} interacciones)...")
    
    data = Dataset.load_from_df(df_cluster[['user_id', 'item_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    model = SVD()
    model.fit(trainset)

    predictions = model.test(testset)

    mae = accuracy.mae(predictions, verbose=False)
    rmse = accuracy.rmse(predictions, verbose=False)
    precision, recall = precision_recall_at_k(predictions, k=10)

    print(f"📊 Cluster {cluster_id}: MAE={mae:.4f} | RMSE={rmse:.4f} | Precision@10={precision:.4f} | Recall@10={recall:.4f}")
    ruta_modelo = os.path.join(DIR_MODELOS, f"svd_cluster_{cluster_id}.pkl")
    dump.dump(ruta_modelo, algo=model)

    resultados.append({
        "cluster": cluster_id,
        "interacciones": len(df_cluster),
        "MAE": mae,
        "RMSE": rmse,
        "Precision@10": precision,
        "Recall@10": recall
    })

# Guardar resultados en un .txt
with open(RUTA_METRICAS, "w") as f:
    for r in resultados:
        f.write(
            f"Cluster {r['cluster']}: {r['interacciones']} interacciones | "
            f"MAE: {r['MAE']:.4f} | RMSE: {r['RMSE']:.4f} | "
            f"Precision@10: {r['Precision@10']:.4f} | Recall@10: {r['Recall@10']:.4f}\n"
        )

print(f"\n✅ Métricas guardadas en {RUTA_METRICAS}")
