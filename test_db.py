from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()  # loads .env from current working directory

url = os.getenv("DATABASE_URL")
print("DATABASE_URL =", url)

engine = create_engine(url)

with engine.connect() as conn:
    print("✅ Connected as:", conn.execute(text("SELECT current_user")).scalar())
    print("✅ DB:", conn.execute(text("SELECT current_database()")).scalar())
