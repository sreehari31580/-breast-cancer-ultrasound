import sqlite3
import bcrypt
import uuid
from webapp.database import DB_PATH, init_db
from datetime import datetime

# Helper to hash passwords
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# Registration: create user with is_verified=0, is_approved=0
def register_user(username, email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    password_hash = hash_password(password)
    verification_token = str(uuid.uuid4())
    try:
        c.execute('''INSERT INTO users (username, email, password_hash, is_verified, is_approved, is_admin, registration_date) VALUES (?, ?, ?, 0, 0, 0, ?)''',
                  (username, email, password_hash, datetime.now().isoformat()))
        conn.commit()
        user_id = c.lastrowid
        # Store token in a simple way (for demo, production should use a separate table)
        c.execute('''CREATE TABLE IF NOT EXISTS email_tokens (user_id INTEGER, token TEXT, FOREIGN KEY(user_id) REFERENCES users(id))''')
        c.execute('''INSERT INTO email_tokens (user_id, token) VALUES (?, ?)''', (user_id, verification_token))
        conn.commit()
        return user_id, verification_token
    except sqlite3.IntegrityError:
        return None, None
    finally:
        conn.close()

# Email verification
def verify_email(token):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT user_id FROM email_tokens WHERE token=?''', (token,))
    row = c.fetchone()
    if row:
        user_id = row[0]
        c.execute('''UPDATE users SET is_verified=1 WHERE id=?''', (user_id,))
        c.execute('''DELETE FROM email_tokens WHERE user_id=?''', (user_id,))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

# Login
def authenticate_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT id, password_hash, is_verified, is_approved, is_admin FROM users WHERE username=?''', (username,))
    row = c.fetchone()
    conn.close()
    if row:
        user_id, password_hash, is_verified, is_approved, is_admin = row
        if check_password(password, password_hash):
            return {
                'id': user_id,
                'is_verified': is_verified,
                'is_approved': is_approved,
                'is_admin': is_admin
            }
    return None

# Admin: approve/ban users
def set_user_approval(user_id, approve=True):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''UPDATE users SET is_approved=? WHERE id=?''', (1 if approve else 0, user_id))
    conn.commit()
    conn.close()

def get_pending_users():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT id, username, email, registration_date FROM users WHERE is_verified=1 AND is_approved=0''')
    users = c.fetchall()
    conn.close()
    return users

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT id, username, email, is_verified, is_approved, is_admin, registration_date FROM users''')
    users = c.fetchall()
    conn.close()
    return users

def log_user_action(user_id, action, details=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO usage_logs (user_id, action, details) VALUES (?, ?, ?)''', (user_id, action, details))
    conn.commit()
    conn.close()

def get_usage_logs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT usage_logs.id, users.username, usage_logs.timestamp, usage_logs.action, usage_logs.details FROM usage_logs LEFT JOIN users ON usage_logs.user_id = users.id ORDER BY usage_logs.timestamp DESC''')
    logs = c.fetchall()
    conn.close()
    return logs 