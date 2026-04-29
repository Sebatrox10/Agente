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
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash') 

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
    formato_cita: str = "APA 7"

class PeticionEdicion(BaseModel):
    contenido: str
    instruccion: str

class PeticionTextoCompleto(BaseModel):
    texto: str
    url_origen: str

# --- ENDPOINTS ---

@app.post("/extraer-texto")
async def extraer_texto(peticion: PeticionTextoCompleto):
    # Usamos un prompt casi idéntico al del PDF, pero adaptado para web
    prompt_analisis = f"""
    Eres un asistente académico avanzado. Analiza el siguiente texto extraído de la web ({peticion.url_origen}).
    
    1. TEMA: Identifica el tema principal en 1 a 3 palabras. Usa guiones bajos.
    2. RESUMEN: Escribe los 3 a 5 puntos más importantes en una sola cadena de texto usando Markdown.
    3. METADATOS: Extrae el título original, el autor (si se menciona) y el año de publicación. Si no los encuentras, usa "Desconocido" o "s.f.".

    Devuelve tu respuesta ESTRICTAMENTE en formato JSON con la siguiente estructura exacta:
    {{
      "tema": "Tema_Detectado",
      "resumen": "- Punto 1\\n- Punto 2...",
      "metadata": {{
        "titulo": "Título de la web",
        "autor": "Autor",
        "anio": "2026"
      }}
    }}

    TEXTO WEB (Extracto):
    {peticion.texto[:15000]} 
    """
    
    try:
        respuesta_llm = model.generate_content(prompt_analisis).text
        respuesta_limpia = respuesta_llm.replace("```json", "").replace("```", "").strip()
        datos_ia = json.loads(respuesta_limpia)
        
        tema_detectado = datos_ia.get("tema", "Investigacion_Web")
        resumen_bruto = datos_ia.get("resumen", "No se pudo generar el resumen.")
        metadata = datos_ia.get("metadata", {"titulo": peticion.url_origen, "autor": "Desconocido", "anio": "s.f."})

        if isinstance(resumen_bruto, list):
            resumen_inteligente = "\n- ".join(str(item) for item in resumen_bruto)
            resumen_inteligente = "- " + resumen_inteligente
        else:
            resumen_inteligente = str(resumen_bruto)

    except Exception as e:
        return {"error": str(e)}

    return {
        "texto": peticion.texto,
        "tema": str(tema_detectado).replace(" ", "_"),
        "resumen": resumen_inteligente,
        "metadata": metadata
    }

# 1. RECIBIR Y LEER PDF (Para tus clases)
@app.post("/extraer-pdf")
async def extraer_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    doc = fitz.open(stream=contents, filetype="pdf")
    texto_completo = ""
    for pagina in doc:
        texto_completo += pagina.get_text()

    # NUEVO PROMPT: Le pedimos los metadatos explícitamente
    prompt_analisis = f"""
    Eres un asistente académico avanzado. Analiza el siguiente texto de un documento universitario o paper científico.
    
    1. TEMA: Identifica el tema principal en 1 a 3 palabras (ej. "Machine_Learning"). Usa guiones bajos.
    2. RESUMEN: Escribe los 3 a 5 puntos más importantes en una sola cadena de texto usando Markdown.
    3. METADATOS: Extrae el título original, el autor (o autores) y el año de publicación. Si no los encuentras, usa "Desconocido" o "s.f.".

    Devuelve tu respuesta ESTRICTAMENTE en formato JSON con la siguiente estructura exacta:
    {{
      "tema": "Tema_Detectado",
      "resumen": "- Punto 1\\n- Punto 2...",
      "metadata": {{
        "titulo": "Título del documento",
        "autor": "Nombre del Autor",
        "anio": "2024"
      }}
    }}

    TEXTO DEL DOCUMENTO (Extracto inicial para metadatos):
    {texto_completo[:15000]} 
    """
    
    try:
        respuesta_llm = model.generate_content(prompt_analisis).text
        respuesta_limpia = respuesta_llm.replace("```json", "").replace("```", "").strip()
        datos_ia = json.loads(respuesta_limpia)
        
        tema_detectado = datos_ia.get("tema", "Investigaciones_Varias")
        resumen_bruto = datos_ia.get("resumen", "No se pudo generar el resumen.")
        metadata = datos_ia.get("metadata", {"titulo": "Desconocido", "autor": "Desconocido", "anio": "s.f."})

        if isinstance(resumen_bruto, list):
            resumen_inteligente = "\n- ".join(str(item) for item in resumen_bruto)
            resumen_inteligente = "- " + resumen_inteligente
        else:
            resumen_inteligente = str(resumen_bruto)

    except Exception as e:
        tema_detectado = "Clasificacion_Pendiente"
        resumen_inteligente = "Hubo un problema de IA: " + str(e)
        metadata = {"titulo": "Desconocido", "autor": "Desconocido", "anio": "s.f."}

    return {
        "texto": texto_completo,
        "tema": str(tema_detectado).replace(" ", "_"),
        "resumen": resumen_inteligente,
        "metadata": metadata # ¡Enviamos el JSON anidado a Java!
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
    Eres Troxi, un Agente de Investigación Académica de nivel universitario.

    CONTEXTO DE LA BASE DE DATOS (Trozos de documentos etiquetados): 
    {peticion.contexto}

    REGLAS DE COMPORTAMIENTO:
    1. CITA RIGUROSA: Utiliza la información de las fuentes proporcionadas para responder. CITA SIEMPRE en formato {peticion.formato_cita}.
    2. BIBLIOGRAFÍA: Al final genera una sección "### Referencias".
    3. IGNORANCIA HONESTA: Si no sabes, no lo inventes.
    4. SINTAXIS: Prohibido LaTeX ($ o $$). Usa Unicode normal.

    MENSAJE DEL USUARIO: 
    {peticion.pregunta}
    """
    
    # --- ESCUDO PARA LA IA ---
    try:
        respuesta_final = model.generate_content(prompt_sistema).text.strip()
        return {"respuesta": respuesta_final}
    except Exception as e:
        # En lugar de colapsar, le avisamos amablemente a Java y a Telegram
        return {"respuesta": f"⚠️ Mi cerebro (Gemini) tuvo un bloqueo al pensar la respuesta: {str(e)}"}


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