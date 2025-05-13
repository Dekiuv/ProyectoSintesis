from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import random

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Cargar CSV y modelo
chatbot_df = pd.read_csv("data/ChatbotSteam.csv", sep=';').dropna()
modelo = SentenceTransformer("all-MiniLM-L6-v2")
preguntas_codificadas = modelo.encode(chatbot_df['Question'].tolist(), convert_to_tensor=True)

@app.get("/", response_class=HTMLResponse)
async def soporte(request: Request):
    temas = ["Steam", "Juegos", "Amigos", "Perfil", "Devolución"]
    return templates.TemplateResponse("soporte.html", {"request": request, "preguntas": temas})

@app.post("/preguntar")
async def preguntar(data: dict):
    pregunta = data.get("pregunta", "")
    pregunta_vec = modelo.encode(pregunta, convert_to_tensor=True)
    similitudes = util.pytorch_cos_sim(pregunta_vec, preguntas_codificadas)[0]
    top_indices = similitudes.topk(k=5).indices.tolist()
    sugerencias = [chatbot_df.iloc[i]["Question"] for i in top_indices]
    return JSONResponse({"sugerencias": sugerencias})

@app.post("/respuesta")
async def respuesta(data: dict):
    pregunta = data.get("pregunta", "")
    pregunta_vec = modelo.encode(pregunta, convert_to_tensor=True)
    similitudes = util.pytorch_cos_sim(pregunta_vec, preguntas_codificadas)[0]
    idx = similitudes.argmax().item()
    respuesta = chatbot_df.iloc[idx]["Answer"]
    return JSONResponse({"respuesta": respuesta})

@app.post("/sugerencias")
async def sugerencias(data: dict):
    pregunta = data.get("pregunta", "")
    pregunta_vec = modelo.encode(pregunta, convert_to_tensor=True)
    similitudes = util.pytorch_cos_sim(pregunta_vec, preguntas_codificadas)[0]
    top_indices = similitudes.topk(k=10).indices.tolist()
    random.shuffle(top_indices)
    top5 = top_indices[:5]
    sugerencias = [chatbot_df.iloc[i]["Question"] for i in top5]
    return JSONResponse({"sugerencias": sugerencias})