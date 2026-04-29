import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from ddgs import DDGS
import datetime
import pytz
from dotenv import load_dotenv

# Importar la base de datos de historial y nuestro nuevo Bibliotecario
from database import init_db, save_message, get_history
from bibliotecario import buscar_en_notas


# Cargar variables de entorno
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- HERRAMIENTAS QUE EL AGENTE PUEDE USAR ---

def buscar_en_internet(consulta: str): 
    """Busca información actualizada en tiempo real en la web."""
    print(f"🌐 Buscando en internet: {consulta}")
    try:
        with DDGS() as ddgs:
            # Buscamos los 3 primeros resultados
            results = [r for r in ddgs.text(consulta, max_results=3)]
            
        if not results:
            return "El buscador no devolvió resultados."
        return str(results)
    except Exception as e:
        print(f"⚠️ Error buscando en internet: {e}")
        return "No me pude conectar a internet en este momento para buscar esta información."

def consultar_mis_notas(consulta: str):
    """Busca en mi bóveda de Obsidian información personal, de estudio o proyectos."""
    print(f"📂 Consultando Obsidian para: {consulta}")
    return buscar_en_notas(consulta)

# Configurar el modelo con las herramientas
# 1. Obtener la fecha exacta del servidor
zona_ecuador = pytz.timezone('America/Guayaquil')
fecha_hoy = datetime.datetime.now(zona_ecuador).strftime("%d de %B de %Y")

# 2. Inyectarle el contexto de tiempo al agente
instrucciones_base = f"Eres mi agente personal. El día de hoy es {fecha_hoy}. Usa obligatoriamente este año y mes como referencia para buscar cualquier noticia o dato actual en internet."

# 3. Configurar el modelo con el calendario incluido
tools = [buscar_en_internet, consultar_mis_notas]
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    tools=tools,
    system_instruction=instrucciones_base
)

# Configurar logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Responde al comando /start."""
    await update.message.reply_text("¡Hola, Sebastián! Mi módulo de lectura de notas (RAG) está activado. ¿Qué quieres buscar en tu Obsidian?")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Procesa los mensajes interceptándolos para inyectar notas de Obsidian."""
    user_id = update.message.from_user.id
    user_text = update.message.text

    # 1. Recuperar el historial válido para Gemini
    history = get_history(user_id, limit=10)
    chat = model.start_chat(history=history, enable_automatic_function_calling=True)

    # 2. RAG: Buscar en tus notas de Obsidian
    contexto_obsidian = buscar_en_notas(user_text)

    try:
        # 3. Le enviamos solo tu texto crudo. Gemini decide si usa herramientas o responde normal.
        response = chat.send_message(user_text)
        bot_reply = response.text
        
        # 4. Guardar en la base de datos
        save_message(user_id, "user", user_text)
        save_message(user_id, "model", bot_reply)
        
        await update.message.reply_text(bot_reply)

    except Exception as e:
        logger.error(f"Error con Gemini: {e}")
        await update.message.reply_text("Lo siento, tuve un problema al procesar tu solicitud.")

async def tarea_proactiva(context: ContextTypes.DEFAULT_TYPE):
    """El agente te habla sin que tú le hables primero."""
    print("⏰ Ejecutando tarea proactiva...")
    query = "Últimas noticias sobre Inteligencia Artificial"
    noticias = buscar_en_internet(query)
    
    prompt = f"Aquí tienes noticias recientes: {noticias}. Hazme un resumen corto."
    resumen = model.generate_content(prompt).text
    
    MI_CHAT_ID = 1516951121
    await context.bot.send_message(chat_id=MI_CHAT_ID, text=f"🤖 REPORTE PROACTIVO:\n\n{resumen}")

def main():
    """Inicia el bot."""
    init_db()

    # Tiempos de espera extendidos para evitar Timeouts en tu servidor
    application = Application.builder().token(TELEGRAM_TOKEN).connect_timeout(30.0).read_timeout(30.0).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Iniciando el bot con RAG activado...")

    zona_ecuador = pytz.timezone('America/Guayaquil')
    hora_ejecucion = datetime.time(hour=8, minute=0, tzinfo=zona_ecuador)
    application.job_queue.run_daily(tarea_proactiva, time=hora_ejecucion)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Iniciando el bot con RAG y Tareas Autónomas...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
