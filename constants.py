"""
Application constants.
"""
from typing import Final

# Prompt Templates
PROMPT_TEMPLATE: Final[str] = """You are an expert assistant that answers questions based on the provided context from PDF documents.

INSTRUCTIONS:
1. Read the context carefully and extract relevant information
2. Answer the question comprehensively using ONLY the information from the context
3. If the answer is not in the context, explicitly state "The answer is not available in the provided context"
4. Do NOT make up information or use knowledge outside the provided context
5. Provide detailed explanations when available
6. Cite specific details from the context when relevant
7. If multiple relevant points exist, list them clearly

CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}

ANSWER:"""

# Error Messages
ERROR_PDF_READING: Final[str] = "Error reading PDF: {error}"
ERROR_TEXT_SPLITTING: Final[str] = "Error splitting text: {error}"
ERROR_VECTOR_STORE: Final[str] = "Error creating vector store: {error}"
ERROR_PROCESSING_QUESTION: Final[str] = "Error processing question: {error}"

# User Messages
MSG_UPLOAD_PDFS: Final[str] = "Please upload PDF files before processing."
MSG_PROCESS_PDFS: Final[str] = "Please upload and process PDF files first."
MSG_API_KEY_REQUIRED: Final[str] = "Please provide API key before processing."
MSG_PDFS_PROCESSED: Final[str] = "PDFs processed successfully!"
MSG_NO_TEXT_EXTRACTED: Final[str] = "Could not extract text from PDFs. Please check if PDFs are valid."
MSG_CHUNKS_FAILED: Final[str] = "Failed to create text chunks."
MSG_VECTOR_STORE_FAILED: Final[str] = "Failed to create vector store."

# UI Constants
AVATAR_USER_URL: Final[str] = "https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png"
AVATAR_BOT_URL: Final[str] = "https://i.ibb.co/wNmYHsx/langchain-logo.webp"
