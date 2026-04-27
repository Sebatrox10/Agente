# 🤖 AI Worker & RAG Engine (Python)

Worker especializado en Inteligencia Artificial y procesamiento de lenguaje natural. Se encarga de procesar los documentos que le envía el Orquestador Java, realizar la vectorización matemática y ejecutar sistemas RAG (Retrieval-Augmented Generation) para automatizar la gestión de conocimiento.

## 🚀 Arquitectura y Tecnologías
* **Core:** Python, FastAPI / Flask (Endpoints REST).
* **IA:** API de Google Gemini (Generación de resúmenes y extracción de entidades).
* **Vectorización:** PostgreSQL con extensión `pgvector` para búsqueda semántica.
* **Despliegue:** Contenerización con Docker.

## ⚙️ Funcionalidades Principales
1. **Procesamiento de PDFs:** Extracción de texto limpio y segmentación de documentos complejos en fragmentos procesables (chunks).
2. **Resumen Inteligente:** Integración con LLMs (Gemini) para generar síntesis estructuradas del contenido y detección de temas principales.
3. **Búsqueda Semántica (RAG):** Conversión del texto en embeddings matemáticos y almacenamiento en `pgvector`, permitiendo consultas de contexto súper precisas.
4. **Exportación a Obsidian:** Formateo automático de los resúmenes y datos extraídos en archivos Markdown compatibles, integrando la IA directamente en el flujo de toma de notas personal.

## 🛠️ Instalación y Configuración Local
1. Clona este repositorio: `git clone [URL_DEL_REPO]`
2. Crea tu entorno virtual: `python -m venv env` y actívalo.
3. Instala las dependencias: `pip install -r requirements.txt`
4. Crea un archivo `.env` en la raíz copiando las variables de `.env.example` y agrega tu `GEMINI_API_KEY` y credenciales de base de datos.
5. Ejecuta el servidor (o levanta el contenedor con Docker si usas Compose).
