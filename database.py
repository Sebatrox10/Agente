import sqlite3
import os

DB_NAME = "chat_history.db"

def init_db():
    """Crea la tabla si no existe."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_message(user_id: int, role: str, content: str):
    """Guarda un mensaje en la base de datos."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO messages (user_id, role, content)
        VALUES (?, ?, ?)
    ''', (user_id, role, content))
    conn.commit()
    conn.close()

def get_history(user_id: int, limit: int = 10):
    """Recupera los últimos 'limit' mensajes de un usuario."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Obtenemos los últimos mensajes y los ordenamos cronológicamente
    cursor.execute('''
        SELECT role, content FROM (
            SELECT role, content, timestamp 
            FROM messages 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ) ORDER BY timestamp ASC
    ''', (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    
    # Formatear para Gemini
    history = []
    for role, content in rows:
        history.append({
            "role": role,
            "parts": [content]
        })
    return history
