from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from scipy.sparse import csr_matrix

# Cargar modelo entrenado
model = joblib.load("modelo_lightfm_sampled.pkl")

# Supongamos que tienes una lista fija de item_ids (juegos) en el mismo orden que se entrenó el modelo
item_ids = joblib.load("item_ids.pkl")  # Lista de item_ids en el mismo orden que el modelo espera
item_id_to_index = {str(item): idx for idx, item in enumerate(item_ids)}

app = FastAPI(title="Recomendador Steam")

# Esquema para entrada
class RecomendacionRequest(BaseModel):
    juegos_usuario: list[str]
    top_n: int = 10

@app.post("/recomendar/")
def recomendar(req: RecomendacionRequest):
    juegos_usuario = req.juegos_usuario
    top_n = req.top_n

    vector = np.zeros((1, len(item_ids)))
    for j in juegos_usuario:
        if j in item_id_to_index:
            vector[0, item_id_to_index[j]] = 1

    user_features = csr_matrix(vector)
    scores = model.predict(0, np.arange(len(item_ids)), user_features=user_features)

    top_indices = np.argsort(-scores)[:top_n]
    recomendaciones = [
        {"item_id": item_ids[i], "score": float(scores[i])}
        for i in top_indices
        if str(item_ids[i]) not in juegos_usuario
    ]

    return {"recomendaciones": recomendaciones}
