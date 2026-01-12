"""
Utility functions for the RAG PDF Chatbot.
"""
import logging
from typing import List, Optional, Tuple
from PyPDF2 import PdfReader
import streamlit as st

from config import Config
from constants import ERROR_PDF_READING, ERROR_TEXT_SPLITTING, ERROR_VECTOR_STORE

# Setup logging - console only (works on Streamlit Cloud)
# File logging removed for cloud compatibility
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Console logging only
    ]
)
logger = logging.getLogger(__name__)


def get_pdf_text(pdf_docs) -> Optional[str]:
    """
    Extract text from uploaded PDF documents.
    
    Args:
        pdf_docs: List of uploaded PDF file objects
        
    Returns:
        Extracted text as string, or None if error occurs
    """
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        logger.info(f"Successfully extracted text from {len(pdf_docs)} PDF(s)")
        return text
    except Exception as e:
        error_msg = ERROR_PDF_READING.format(error=str(e))
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return None


def get_text_chunks(text: str, model_name: str) -> List[str]:
    """
    Split text into chunks for processing.
    
    Args:
        text: Text to split
        model_name: Model name (for future model-specific chunking)
        
    Returns:
        List of text chunks
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=Config.CHUNK_SEPARATORS
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    except Exception as e:
        error_msg = ERROR_TEXT_SPLITTING.format(error=str(e))
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return []


def get_vector_store(text_chunks: List[str], model_name: str, api_key: Optional[str] = None):
    """
    Create FAISS vector store from text chunks.
    
    Args:
        text_chunks: List of text chunks to embed
        model_name: Model name
        api_key: API key (not used for embeddings, kept for consistency)
        
    Returns:
        FAISS vector store object, or None if error occurs
    """
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': Config.EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(str(Config.FAISS_INDEX_DIR))
        logger.info(f"Created vector store with {len(text_chunks)} chunks")
        return vector_store
    except Exception as e:
        error_msg = ERROR_VECTOR_STORE.format(error=str(e))
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return None
