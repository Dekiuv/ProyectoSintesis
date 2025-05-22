# 🎮 Steamia - Sistema de Recomendación de Videojuegos

**Steamia** es una plataforma inteligente de recomendación de videojuegos basada en datos reales de Steam. Integra múltiples técnicas de Inteligencia Artificial para personalizar la experiencia de los usuarios, incluyendo recomendaciones por historial de juego, análisis de sentimiento de reseñas, sistema de carrito de compra, chatbot de soporte y más.

## 🚀 Funcionalidades

- 🔑 Login con Steam ID para obtener juegos propios.
- 🛍️ Sistema de tienda con recomendaciones personalizadas (Word2Vec).
- 🧠 Sistema de recomendación híbrido:
  - Recomendaciones basadas en juegos jugados (Word2Vec)
  - Reglas de asociación (Market Basket Analysis por clúster)
  - Reseñas con análisis de sentimiento (modelo transformers)
- 💬 Chatbot de soporte entrenado con preguntas frecuentes (Sentence Transformers).
- 📬 Confirmación por correo electrónico tras la compra.
- 📊 Recomendaciones visuales en la página de carrito con modelo de reglas.
- 🧾 Página de biblioteca y gestión de amigos integrada.

## 🧠 Tecnologías utilizadas

| Lenguaje / Herramienta        | Descripción breve                                                   |
|------------------------------|---------------------------------------------------------------------|
| **Python**                   | Lógica del backend y procesamiento de datos                         |
| **FastAPI**                  | Framework web para construir la API                                 |
| **HTML / CSS / JS**          | Interfaz de usuario interactiva                                     |
| **Gensim (Word2Vec)**        | Recomendaciones de juegos similares por historial                   |
| **Transformers (HuggingFace)**| Análisis de sentimiento de reseñas                                 |
| **Sentence Transformers**    | Embeddings para el chatbot inteligente                              |
| **Scikit-learn (PCA, KMeans)**| Clustering y reducción de dimensionalidad                           |
| **pandas / numpy**           | Análisis y manipulación de datos                                    |
| **SMTP (smtplib)**           | Envío automático de correos de confirmación                         |
| **Steam API**                | Obtención en tiempo real del perfil y biblioteca del usuario        |
| **BeautifulSoup**            | Web scraping de los juegos más vendidos de Steam                    |

## 📁 Estructura del proyecto


## ⚙️ Instalación y ejecución

1. Clona el repositorio:

```bash
git clone https://github.com/tuusuario/steamia.git
cd steamia
```
2. Instala las librerias necesarias
```bash
pip install -r requirements.txt
```

3. Ejecutar programa
```bash
python -m uvicorn app:app --reload
```
