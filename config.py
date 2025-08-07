import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Gemini API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyC7IFzwAaG5WGcSjIJhKbeaQ")
    
    # Flask Configuration
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "1")
    
    # Chat Configuration
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default
    MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", "20"))
    MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "5"))
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO") 