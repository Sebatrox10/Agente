import os
import glob
import pickle
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# Cargar las claves
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

OBSIDIAN_PATH = "/home/sebatrox/Obsidian_Vault"
INDEX_PATH = "indice_puro.pkl"

def dividir_texto(texto, max_chars=1000, overlap=100):
    """Función nativa y ligera para dividir notas largas en fragmentos."""
    fragmentos = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + max_chars
        fragmentos.append(texto[inicio:fin])
        inicio += max_chars - overlap
    return fragmentos

def construir_indice():
    """Lee las notas y usa la API nativa de Google para los vectores."""
    print("📚 Buscando notas en Obsidian...")
    archivos_md = glob.glob(os.path.join(OBSIDIAN_PATH, "**/*.md"), recursive=True)
    
    fragmentos_totales = []
    origenes = []

    for ruta in archivos_md:
        try:
            with open(ruta, 'r', encoding='utf-8') as f:
                contenido = f.read()
                if contenido.strip():
                    nombre_archivo = os.path.basename(ruta)
                    # Dividimos el texto nosotros mismos sin Langchain
                    trozos = dividir_texto(contenido)
                    fragmentos_totales.extend(trozos)
                    origenes.extend([nombre_archivo] * len(trozos))
        except Exception as e:
            print(f"⚠️ Error leyendo {ruta}: {e}")

    if not fragmentos_totales:
        print("❌ No se encontraron archivos .md. Revisa la ruta.")
        return

    print(f"✅ Se encontraron {len(archivos_md)} notas, divididas en {len(fragmentos_totales)} fragmentos.")
    print("🧠 Conectando con Google para generar vectores (esto toma unos segundos)...")
    
    vectores = []
    # Pedimos los vectores uno por uno a Gemini usando el modelo más actual de embeddings
    for frag in fragmentos_totales:
        respuesta = genai.embed_content(
            model="models/text-embedding-004",
            content=frag,
            task_type="retrieval_document"
        )
        vectores.append(respuesta['embedding'])
    
    # Guardar en nuestro archivo super ligero
    datos_indice = {
        "fragmentos": fragmentos_totales,
        "origenes": origenes,
        "vectores": np.array(vectores) # Usamos el numpy del sistema
    }
    
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(datos_indice, f)
        
    print("🚀 ¡Índice construido y guardado con éxito en 'indice_puro.pkl'!")

def buscar_en_notas(consulta, k=3):
    """Busca en el índice usando el producto punto matemático."""
    if not os.path.exists(INDEX_PATH):
        return ""
    
    with open(INDEX_PATH, "rb") as f:
        datos = pickle.load(f)
        
    # Convertir la pregunta a vector
    respuesta = genai.embed_content(
        model="models/text-embedding-004",
        content=consulta,
        task_type="retrieval_query"
    )
    vector_pregunta = np.array(respuesta['embedding'])
    
    # Matemáticas puras: Similitud del coseno (Producto Punto)
    similitudes = np.dot(datos["vectores"], vector_pregunta)
    
    # Obtener los mejores resultados
    mejores_indices = np.argsort(similitudes)[-k:][::-1]
    
    contexto = "INFORMACIÓN DE MIS NOTAS:\n\n"
    for i in mejores_indices:
        contexto += f"--- Extraído de: {datos['origenes'][i]} ---\n{datos['fragmentos'][i]}\n\n"
    
    return contexto

if __name__ == "__main__":
    construir_indice()
