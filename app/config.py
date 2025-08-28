import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # API settings
    APP_NAME: str = "OveloAI API"
    VERSION: str = "1.0.0"

    # SMTP settings for email notifications
    SMTP_SERVER: str = os.getenv("SMTP_SERVER")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", 587))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD")
    RECEIVER_EMAIL: str = os.getenv("RECEIVER_EMAIL")

    # AI settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434") # New: Default to localhost, will be changed to VM IP
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "phi3:mini")
    FAISS_DB_PATH: str = os.getenv("FAISS_DB_PATH", "faiss_db")
    # --- REVERTED TO ORIGINAL EMBEDDING MODEL FOR QUALITY ---
    HUGGINGFACE_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Lead capture phrases
    LEAD_CAPTURE_PHRASES: list = [
        "price", "quote", "cost", "contact", "talk to", "meeting", "schedule"
    ]

settings = Settings()
