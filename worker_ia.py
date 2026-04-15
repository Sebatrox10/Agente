from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import fitz
import os
from dotenv import load_dotenv
import json

from sentence_transformers import SentenceTransformer
import arxiv

# Cargamos las variables del archivo .env que ya tienes
load_dotenv()

# 1. Configuración de Gemini
genai.configure(api_key="AIzaSyD3ZWEg608EVge7g1qsmRovy_-jAcoqzMs")
model = genai.GenerativeModel('gemini-flash-latest') 

# Inicializamos el modelo matemático que convierte texto a vectores
embedder = SentenceTransformer('all-MiniLM-L6-v2')

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

class PeticionEdicion(BaseModel):
    contenido: str
    instruccion: str

# --- ENDPOINTS ---

# 1. RECIBIR Y LEER PDF (Para tus clases)
@app.post("/extraer-pdf")
async def extraer_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    doc = fitz.open(stream=contents, filetype="pdf")
    texto_completo = ""
    for pagina in doc:
        texto_completo += pagina.get_text()

    # Le pedimos a Gemini que haga el trabajo duro
    prompt_analisis = f"""
    Eres un asistente académico. Analiza el siguiente texto de un documento universitario.
    1. Identifica el TEMA PRINCIPAL en 1 a 3 palabras (ej. "Logica_Matematica", "Machine_Learning"). Usa guiones bajos en vez de espacios.
    2. Escribe un RESUMEN INTELIGENTE con los 3 a 5 puntos o descubrimientos más importantes.

    Devuelve tu respuesta ESTRICTAMENTE en formato JSON, con las claves "tema" y "resumen". 
    IMPORTANTE: El valor de "resumen" DEBE SER UNA SOLA CADENA DE TEXTO (String) usando Markdown, NO un array ni una lista.

    TEXTO DEL DOCUMENTO (Extracto):
    {texto_completo[:60000]} 
    """
    
    try:
        respuesta_llm = model.generate_content(prompt_analisis).text
        respuesta_limpia = respuesta_llm.replace("```json", "").replace("```", "").strip()
        datos_ia = json.loads(respuesta_limpia)
        
        tema_detectado = datos_ia.get("tema", "Investigaciones_Varias")
        resumen_bruto = datos_ia.get("resumen", "No se pudo generar el resumen.")

        # ¡EL ESCUDO! Si Gemini desobedece y manda una lista, la convertimos a texto
        if isinstance(resumen_bruto, list):
            resumen_inteligente = "\n- ".join(str(item) for item in resumen_bruto)
            resumen_inteligente = "- " + resumen_inteligente # Añade viñeta al primero
        else:
            resumen_inteligente = str(resumen_bruto)

    except Exception as e:
        tema_detectado = "Clasificacion_Pendiente"
        resumen_inteligente = "Hubo un problema al pedirle el resumen a la IA: " + str(e)

    return {
        "texto": texto_completo,
        "tema": str(tema_detectado).replace(" ", "_"), # Forzamos que no tenga espacios para la carpeta
        "resumen": resumen_inteligente
    }

# 2. VECTORIZAR TEXTO (Para guardar en PostgreSQL)
@app.post("/vectorizar")
async def vectorizar(peticion: PeticionVector):
    # Asume que ya tienes tu modelo de embeddings cargado arriba
    vector = embedder.encode(peticion.texto).tolist()
    return {"vector": vector}

# 3. BUSCAR EN INTERNET (Comando /arxiv)
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

# 4. CHAT Y RAG (Responder preguntas normales)
@app.post("/generar-respuesta")
async def generar_respuesta(peticion: PeticionSintesis):
    prompt_sistema = f"""
    Eres Troxi, el asistente personal académico e investigador del usuario.

    CONTEXTO DE LA BASE DE DATOS (Trozos de PDFs): 
    {peticion.contexto}

    REGLAS DE COMPORTAMIENTO:
    1. CHARLA CASUAL: Si el usuario te saluda o hace charla normal, responde con naturalidad.
    2. RESPONDER PREGUNTAS: Si el usuario hace una pregunta, utiliza PRINCIPALMENTE el CONTEXTO DE LA BASE DE DATOS para responder.
    3. IGNORANCIA HONESTA: Si la respuesta no está en el CONTEXTO, no la inventes. Dile al usuario que no lo sabes y sugiérele usar el comando /arxiv.
    4. FORMATO OBLIGATORIO: ESTÁ ESTRICTAMENTE PROHIBIDO usar sintaxis LaTeX (símbolos $ o $$). Debes escribir toda la matemática, lógica y símbolos usando caracteres Unicode normales (ej. p ∨ q, p → q, ¬p) para que se rendericen correctamente en Telegram.

    MENSAJE DEL USUARIO: 
    {peticion.pregunta}
    """
    respuesta_final = model.generate_content(prompt_sistema).text.strip()
    return {"respuesta": respuesta_final}


@app.post("/editar-documento")
async def editar_documento(peticion: PeticionEdicion):
    prompt_edicion = f"""
    Eres un asistente experto en Obsidian. 
    A continuación tienes el CONTENIDO ACTUAL de una nota académica, y una INSTRUCCIÓN del usuario sobre cómo modificarla o completarla.
    
    INSTRUCCIÓN DEL USUARIO: {peticion.instruccion}
    
    CONTENIDO ACTUAL:
    {peticion.contenido}
    
    Debes devolver ÚNICAMENTE el texto completo modificado en formato Markdown, listo para sobreescribir el archivo.
    Respeta las etiquetas (tags) y la estructura inicial. No agregues texto introductorio ni explicaciones, solo el documento final.
    """
    
    try:
        nuevo_contenido = model.generate_content(prompt_edicion).text.strip()
        
        # El escudo elegante: Si Gemini envuelve su respuesta en un bloque de código markdown,
        # simplemente separamos por líneas y descartamos la primera (```markdown) y la última (```)
        if nuevo_contenido.startswith("```") and nuevo_contenido.endswith("```"):
            lineas = nuevo_contenido.split('\n')
            # Verificamos que tenga más de 2 líneas para no borrar un texto vacío por error
            if len(lineas) > 2:
                nuevo_contenido = '\n'.join(lineas[1:-1])
            
        return {"texto_editado": nuevo_contenido.strip()}
        
    except Exception as e:
        return {"texto_editado": "ERROR", "detalle": str(e)}