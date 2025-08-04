import sqlite3
from datetime import datetime

DB_PATH = 'webapp/app.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_verified INTEGER DEFAULT 0,
            is_approved INTEGER DEFAULT 0,
            is_admin INTEGER DEFAULT 0,
            registration_date TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Usage logs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            action TEXT,
            details TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print('Database initialized.') 