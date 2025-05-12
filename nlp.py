import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Cargar dataset (ajusta si tu archivo se llama distinto)
df = pd.read_csv("juegos.csv")
df = df.dropna(subset=['item_name'])

# Cargar modelo preentrenado
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# Codificar nombres de juegos
print("🔁 Codificando juegos, espera unos segundos...")
nombres_juegos = df['item_name'].tolist()
embeddings_juegos = modelo.encode(nombres_juegos, convert_to_tensor=True)
print("✅ Juegos cargados y codificados.")

def buscar_juegos(texto, top_k=5):
    emb_usuario = modelo.encode(texto, convert_to_tensor=True)
    similitudes = util.pytorch_cos_sim(emb_usuario, embeddings_juegos)[0]
    top_indices = similitudes.topk(k=top_k).indices.tolist()

    print(f"\n🎮 Resultados más parecidos a: \"{texto}\"\n")
    for i, idx in enumerate(top_indices):
        nombre = df.iloc[idx]['item_name']
        app_id = df.iloc[idx]['item_id']
        score = similitudes[idx].item()
        print(f"{i+1}. {nombre}")
    print()

# Bucle principal
if __name__ == "__main__":
    print("🧠 Buscador de juegos Steam con NLP (Escribe 'salir' para terminar)\n")
    while True:
        consulta = input("🔍 Escribe algo para buscar juegos: ")
        if consulta.lower() == "salir":
            print("👋 ¡Hasta luego!")
            break
        buscar_juegos(consulta)
