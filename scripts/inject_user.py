import sqlite3
import bcrypt

DB_PATH = "/Users/apple/Mini Projects/posture new/posture_monitoring.db"
PHONE = "3254769809"
PASSWORD = "Admin@123"

def add_user():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Ensure users table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone_number VARCHAR UNIQUE,
            hashed_password VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # (Actually the table is created by SQLAlchemy, so just insert)
    
    # Hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(PASSWORD.encode('utf-8'), salt).decode('utf-8')
    
    try:
        cursor.execute("INSERT INTO users (phone_number, hashed_password) VALUES (?, ?)", (PHONE, hashed))
        conn.commit()
        print(f"User {PHONE} successfully injected into database.")
    except Exception as e:
        print(f"Error (maybe already exists?): {e}")
        # Try update if exists
        cursor.execute("UPDATE users SET hashed_password = ? WHERE phone_number = ?", (hashed, PHONE))
        conn.commit()
        print(f"Password UPDATED for {PHONE}.")
    
    conn.close()

if __name__ == "__main__":
    add_user()
