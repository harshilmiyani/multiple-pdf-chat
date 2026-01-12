"""
Configuration management for RAG PDF Chatbot.
Application settings (API keys are entered by users in the UI).
"""
import os
from typing import Optional
from pathlib import Path


class Config:
    """Application configuration class."""
    
    # Paths
    BASE_DIR = Path(__file__).parent
    FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
    
    # Note: API keys are entered by users in the Streamlit UI (client-side only)
    
    # Model Configuration
    DEFAULT_MODEL = "gemini-2.0-flash"
    DEFAULT_TEMPERATURE = 0.1
    
    # Embedding Configuration
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE = "cpu"
    
    # Chunking Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
    
    # Retrieval Configuration
    MAX_DOCUMENTS_RETRIEVED = 8
    SIMILARITY_SCORE_THRESHOLD = 1.5
    MIN_DOCUMENTS_FALLBACK = 3
    MAX_CONTEXT_LENGTH = 8000
    
    # Streamlit Configuration
    PAGE_TITLE = "Chat with multiple PDFs"
    PAGE_ICON = ":books:"
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure required directories exist."""
        cls.FAISS_INDEX_DIR.mkdir(exist_ok=True)
