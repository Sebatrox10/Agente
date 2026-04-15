from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import fitz
import os
from dotenv import load_dotenv

import arxiv

# Cargamos las variables del archivo .env que ya tienes
load_dotenv()

# 1. Configuración de Gemini
genai.configure(api_key="AIzaSyD3ZWEg608EVge7g1qsmRovy_-jAcoqzMs")
model = genai.GenerativeModel('gemini-flash-latest') 

# 2. Inicialización de API y Modelo de Vectores
app = FastAPI(title="Worker IA Vectorial")

print("Cargando modelo cerebral... 🧠")
modelo = SentenceTransformer('all-MiniLM-L6-v2')
print("Modelo cargado. Listo para vectorizar.")

# --- ESTRUCTURAS DE DATOS ---

class PeticionVector(BaseModel):
    texto: str

class PeticionSintesis(BaseModel):
    pregunta: str
    contexto: str

# --- ENDPOINTS ---

@app.post("/vectorizar")
def vectorizar_texto(peticion: PeticionVector):
    """Convierte texto en vectores para Postgres."""
    vector_matematico = modelo.encode(peticion.texto).tolist()
    return {"vector": vector_matematico}

@app.post("/extraer-pdf")
async def extraer_pdf(file: UploadFile = File(...)):
    """Extrae texto de un archivo PDF."""
    contenido = await file.read()
    doc = fitz.open(stream=contenido, filetype="pdf")
    texto_completo = "".join([pagina.get_text() for pagina in doc])
    return {"texto": texto_completo}

@app.post("/investigar-arxiv")
async def investigar_arxiv(peticion: PeticionVector): 
    search = arxiv.Search(
        query = peticion.texto,
        max_results = 5,
        sort_by = arxiv.SortCriterion.Relevance
    )
    
    opciones = []
    for res in search.results():
        opciones.append({
            "titulo": res.title,
            "resumen": res.summary[:500] + "...", 
            "url": res.pdf_url,
            "id_arxiv": res.entry_id.split('/')[-1]
        })
    return {"opciones": opciones}


@app.post("/generar-respuesta")
async def generar_respuesta(peticion: PeticionSintesis):
    # 1. Gemini decide si el contexto de tus PDFs basta
    prompt_decision = f"""
    CONTEXTO LOCAL: {peticion.contexto}
    PREGUNTA: {peticion.pregunta}
    ¿Es posible responder la pregunta de forma académica solo con el contexto local? Responde 'SI' o 'NO'.
    """
    decision = model.generate_content(prompt_decision).text.strip()

    contexto_final = peticion.contexto
    fuente_extra = ""

    # 2. Si no basta, vamos a la biblioteca científica (ArXiv)
    if "NO" in decision.upper():
        fuente_extra = buscar_en_arxiv(peticion.pregunta)
        contexto_final += "\n\nINFORMACIÓN ACADÉMICA (ArXiv):\n" + fuente_extra

    # 3. Generación final con "Sustento Científico"
    prompt_final = f"""
    Eres Troxi, el Agente Académico de Sebastián. Tu tarea es crear una nota para Obsidian.

    TEMA: {peticion.pregunta}
    CONTEXTO: {contexto_final}

    INSTRUCCIONES DE FORMATO:
    1. Si se solicita un MENTEFACTO, usa bloques de código 'mermaid' tipo 'graph TD'.
    2. Usa vínculos de Obsidian [[Concepto]] para términos importantes.
    3. Si el concepto ya existe en el contexto, vincúlalo.
    4. Estructura: 
    # {peticion.pregunta}
    - **Concepto Central**: ...
    - **Mentefacto**: [Código Mermaid aquí]
    - **Resumen Crítico**: ...
    """
    
    respuesta = model.generate_content(prompt_final)
    return {"respuesta": respuesta.text}
