from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import fitz
import os
from dotenv import load_dotenv
from duckduckgo_search import DDGS
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

def buscar_en_web(query: str):
    """Busca en internet y devuelve un resumen de los mejores resultados."""
    print(f"🌐 Buscando en la red sobre: {query}...")
    with DDGS() as ddgs:
        resultados = [r['body'] for r in ddgs.text(query, max_results=5)]
    return "\n".join(resultados)

def buscar_en_arxiv(query: str):
    """Busca los 3 artículos más relevantes en ArXiv y extrae sus resúmenes."""
    print(f"📚 Investigando en ArXiv sobre: {query}...")
    search = arxiv.Search(
        query = query,
        max_results = 3,
        sort_by = arxiv.SortCriterion.Relevance
    )
    
    resultados_academicos = []
    for result in search.results():
        info = f"Título: {result.title}\nAutores: {result.authors}\nResumen: {result.summary}\nURL: {result.pdf_url}"
        resultados_academicos.append(info)
        
    return "\n\n---\n\n".join(resultados_academicos)

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
    Eres Troxi, el Agente Académico de Sebastián.
    Tu tarea es responder basándote en el contexto local y la investigación científica adjunta.
    
    {contexto_final}
    
    Pregunta: {peticion.pregunta}
    
    REGLA DE ORO: Si usaste información de ArXiv, menciona el título del artículo al final de la respuesta para que Sebastián pueda citarlo. Usa LaTeX para las fórmulas.
    """
    
    respuesta = model.generate_content(prompt_final)
    return {"respuesta": respuesta.text}