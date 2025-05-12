import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Cargar dataset con separador ;
chatbot_df = pd.read_csv("data\ChatbotSteam.csv", sep=';').dropna()

# Cargar modelo
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# Codificar preguntas
preguntas_codificadas = modelo.encode(chatbot_df['Question'].tolist(), convert_to_tensor=True)

def mostrar_opciones(pregunta_usuario, top_k=5):
    pregunta_vec = modelo.encode(pregunta_usuario, convert_to_tensor=True)
    similitudes = util.pytorch_cos_sim(pregunta_vec, preguntas_codificadas)[0]

    # Obtener top_k más similares
    top_indices = similitudes.topk(k=top_k).indices.tolist()

    print("\n🤖 ¿Quizás quisiste preguntar una de estas?")
    for i, idx in enumerate(top_indices):
        pregunta = chatbot_df.iloc[idx]['Question']
        score = similitudes[idx].item()
        print(f"{i+1}. {pregunta}")

    return top_indices

def responder_seleccion(indices):
    while True:
        seleccion = input("\n🔢 Elige el número de la pregunta que prefieras (1-5) o '0' para cancelar: ")
        if seleccion == '0':
            print("❌ Consulta cancelada.")
            return
        if seleccion.isdigit() and 1 <= int(seleccion) <= len(indices):
            idx = indices[int(seleccion) - 1]
            respuesta = chatbot_df.iloc[idx]['Answer']
            print(f"\n🗨️ Respuesta: {respuesta}\n")
            return
        else:
            print("⚠️ Entrada inválida. Intenta con un número válido.")

# Menú principal
if __name__ == "__main__":
    print("🤖 Asistente Steam (Escribe 'salir' para terminar)\n")
    while True:
        pregunta = input("💬 Tú: ")
        if pregunta.lower() == 'salir':
            print("👋 ¡Hasta luego!")
            break
        indices = mostrar_opciones(pregunta)
        responder_seleccion(indices)
