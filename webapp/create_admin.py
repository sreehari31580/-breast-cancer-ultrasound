import sqlite3
import getpass
from webapp.database import DB_PATH, init_db
import bcrypt
from datetime import datetime

init_db()

print("=== Create Admin User ===")
username = input("Username: ")
email = input("Email: ")
while True:
    password = getpass.getpass("Password: ")
    password2 = getpass.getpass("Confirm Password: ")
    if password == password2:
        break
    print("Passwords do not match. Try again.")

password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
try:
    c.execute('''INSERT INTO users (username, email, password_hash, is_verified, is_approved, is_admin, registration_date) VALUES (?, ?, ?, 1, 1, 1, ?)''',
              (username, email, password_hash, datetime.now().isoformat()))
    conn.commit()
    print(f"Admin user '{username}' created successfully.")
except sqlite3.IntegrityError:
    print("Username or email already exists.")
finally:
    conn.close() 