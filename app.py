"""
RAG PDF Chatbot - Main Application
A Streamlit-based application for chatting with PDF documents using RAG (Retrieval-Augmented Generation).
"""
import streamlit as st
import pandas as pd
import base64
from typing import Optional, List, Tuple
from datetime import datetime

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Local imports
from config import Config
from constants import (
    PROMPT_TEMPLATE,
    ERROR_PROCESSING_QUESTION,
    MSG_UPLOAD_PDFS,
    MSG_PROCESS_PDFS,
    MSG_API_KEY_REQUIRED,
    MSG_PDFS_PROCESSED,
    MSG_NO_TEXT_EXTRACTED,
    MSG_CHUNKS_FAILED,
    MSG_VECTOR_STORE_FAILED
)
from utils import get_pdf_text, get_text_chunks, get_vector_store, logger

# Ensure directories exist
Config.ensure_directories()


def get_conversational_chain(model_name: str, api_key: str) -> dict:
    """
    Create conversational chain components for question answering.
    
    Args:
        model_name: Name of the model to use
        api_key: API key for the model (required)
        
    Returns:
        Dictionary containing model and prompt components
    """
    if model_name == "Google AI":
        model = ChatGoogleGenerativeAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.DEFAULT_TEMPERATURE,
            google_api_key=api_key
        )
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        return {"model": model, "prompt": prompt}
    return {}


def process_question(
    user_question: str,
    model_name: str,
    api_key: str,
    vector_store: FAISS
) -> Optional[str]:
    """
    Process a user question and generate an answer.
    
    Args:
        user_question: The question to answer
        model_name: Name of the model to use
        api_key: API key for the model
        vector_store: FAISS vector store containing document embeddings
        
    Returns:
        Generated answer as string, or None if error occurs
    """
    try:
        # Retrieve relevant documents with similarity scores
        docs_with_scores = vector_store.similarity_search_with_score(
            user_question,
            k=Config.MAX_DOCUMENTS_RETRIEVED
        )
        
        # Filter documents by similarity score
        filtered_docs = []
        for doc, score in docs_with_scores:
            if score < Config.SIMILARITY_SCORE_THRESHOLD:
                filtered_docs.append(doc)
        
        # Fallback to top documents if threshold too strict
        if not filtered_docs:
            filtered_docs = [doc for doc, _ in docs_with_scores[:Config.MIN_DOCUMENTS_FALLBACK]]
        
        # Format context with document markers
        context_parts = []
        for i, doc in enumerate(filtered_docs, 1):
            context_parts.append(f"[Document {i}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Limit context length to avoid token limits
        if len(context) > Config.MAX_CONTEXT_LENGTH:
            truncated_context = ""
            for part in context_parts:
                if len(truncated_context + part) < Config.MAX_CONTEXT_LENGTH:
                    truncated_context += part + "\n\n---\n\n"
                else:
                    break
            context = truncated_context
        
        # Get chain components
        chain_components = get_conversational_chain(model_name, api_key)
        if not chain_components:
            st.error("Unsupported model")
            return None
            
        model = chain_components["model"]
        prompt = chain_components["prompt"]
        
        # Format prompt with context and question
        formatted_prompt = prompt.format(context=context, question=user_question)
        
        # Generate response
        messages = [HumanMessage(content=formatted_prompt)]
        response = model.invoke(messages)
        
        # Extract answer from response
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
            
    except Exception as e:
        error_msg = ERROR_PROCESSING_QUESTION.format(error=str(e))
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return None


def display_chat_message(question: str, answer: str) -> None:
    """Display a chat message in the UI - simple and clean."""
    # User message
    with st.chat_message("user"):
        st.write(question)
    
    # Bot message
    with st.chat_message("assistant"):
        st.write(answer)


def display_conversation_history(conversation_history: List[Tuple]) -> None:
    """Display conversation history."""
    for question, answer, _, _, _ in conversation_history:
        display_chat_message(question, answer)


def handle_user_input(
    user_question: str,
    model_name: str,
    api_key: str,
    conversation_history: List[Tuple]
) -> None:
    """
    Handle user input and generate response.
    
    Args:
        user_question: The question to answer
        model_name: Name of the model to use
        api_key: API key for the model (required)
        conversation_history: List to append conversation to
    """
    if not api_key:
        st.warning(MSG_API_KEY_REQUIRED)
        return
    
    if 'vector_store' not in st.session_state or st.session_state.vector_store is None:
        st.warning(MSG_PROCESS_PDFS)
        return
    
    # Process question with loading indicator
    with st.spinner("🤔 Processing your question..."):
        response_output = process_question(
            user_question,
            model_name,
            api_key,
            st.session_state.vector_store
        )
    
    if response_output:
        # Add to conversation history
        pdf_names = st.session_state.pdf_names if 'pdf_names' in st.session_state else []
        conversation_history.append((
            user_question,
            response_output,
            model_name,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ", ".join(pdf_names)
        ))
        
        # Download conversation as CSV
        if len(conversation_history) > 0:
            df = pd.DataFrame(
                conversation_history,
                columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"]
            )
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv">📥 Download CSV</a>'
            st.sidebar.markdown(href, unsafe_allow_html=True)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'pdf_names' not in st.session_state:
        st.session_state.pdf_names = []


def render_sidebar() -> Tuple[str, Optional[str]]:
    """Render sidebar UI and return model name and API key."""
    # Logo/Name header - clickable to website (prominent logo)
    st.sidebar.markdown(
        """
        <style>
            .logo-link {
                text-decoration: none;
                color: inherit;
                display: block;
            }
            .logo-link:hover h1 {
                transform: scale(1.05);
            }
        </style>
        <div style="text-align: center; margin-bottom: 25px; padding: 20px 0; border-bottom: 2px solid #e0e0e0;">
            <a href="https://harshilmiyani.com" target="_blank" class="logo-link">
                <h1 style="margin: 0; font-size: 32px; font-weight: 800; 
                           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           -webkit-background-clip: text;
                           -webkit-text-fill-color: transparent;
                           background-clip: text;
                           cursor: pointer;
                           transition: all 0.3s ease;
                           letter-spacing: -1px;
                           line-height: 1.2;">
                    Harshil Miyani
                </h1>
                <p style="margin: 8px 0 0 0; font-size: 11px; color: #888; font-weight: 400; text-transform: uppercase; letter-spacing: 1px;">
                    Click to visit website
                </p>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Source code link - prominent
    st.sidebar.markdown(
        """
        <div style="margin-bottom: 15px;">
            <a href="https://github.com/harshilmiyani/multiple-pdf-chat" target="_blank" 
               style="display: block; padding: 12px 15px; background: linear-gradient(135deg, #24292e 0%, #181717 100%); 
                      border-radius: 8px; text-decoration: none; color: white !important; font-weight: 600;
                      text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: all 0.3s ease;">
                <span style="font-size: 20px; margin-right: 8px;">📂</span>View Source Code
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Professional social links with badges
    st.sidebar.markdown(
        """
        <style>
            .profile-link {
                display: block;
                padding: 10px 15px;
                margin: 8px 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 8px;
                text-decoration: none;
                color: white !important;
                font-weight: 500;
                transition: all 0.3s ease;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .profile-link:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            .profile-link.github {
                background: linear-gradient(135deg, #24292e 0%, #181717 100%);
            }
            .profile-link.linkedin {
                background: linear-gradient(135deg, #0077b5 0%, #005885 100%);
            }
            .profile-link.website {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .profile-link span {
                margin-right: 8px;
                font-size: 18px;
            }
        </style>
        <div style="margin: 15px 0;">
            <a href="https://github.com/harshilmiyani" target="_blank" class="profile-link github">
                <span>🔗</span>GitHub Profile
            </a>
            <a href="https://www.linkedin.com/in/harshilmiyani/" target="_blank" class="profile-link linkedin">
                <span>💼</span>LinkedIn Profile
            </a>
            <a href="https://harshilmiyani.com" target="_blank" class="profile-link website">
                <span>🌐</span>Personal Website
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔑 API Key")
    
    api_key = st.sidebar.text_input(
        "Google API Key:",
        type="password",
        help="Get your API key from ai.google.dev"
    )
    
    if not api_key:
        st.sidebar.info("👆 Enter your API key")
        return "Google AI", None
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Upload PDFs")
    
    return "Google AI", api_key


def process_pdf_uploads(pdf_docs, model_name: str, api_key: str) -> None:
    """Process uploaded PDF files."""
    if st.button("✅ Process PDFs", type="primary"):
        if pdf_docs:
            with st.spinner("⏳ Processing..."):
                try:
                    # Extract text
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text is None or raw_text == "":
                        st.error("❌ " + MSG_NO_TEXT_EXTRACTED)
                        return
                    
                    # Create chunks
                    text_chunks = get_text_chunks(raw_text, model_name)
                    if not text_chunks:
                        st.error("❌ " + MSG_CHUNKS_FAILED)
                        return
                    
                    # Create vector store
                    vector_store = get_vector_store(text_chunks, model_name, api_key)
                    if vector_store is None:
                        st.error("❌ " + MSG_VECTOR_STORE_FAILED)
                        return
                    
                    # Store in session state
                    st.session_state.vector_store = vector_store
                    st.session_state.pdf_names = [pdf.name for pdf in pdf_docs]
                    st.success("✅ " + MSG_PDFS_PROCESSED)
                    logger.info(f"Successfully processed {len(pdf_docs)} PDF(s)")
                except Exception as e:
                    logger.error(f"Error processing PDFs: {str(e)}", exc_info=True)
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ " + MSG_UPLOAD_PDFS)


def main() -> None:
    """Main application function."""
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout="centered"
    )
    
    st.title(f"{Config.PAGE_ICON} {Config.PAGE_TITLE}")
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    model_name, api_key = render_sidebar()
    if model_name is None or api_key is None:
        st.info("👆 Please enter your Google API key in the sidebar to get started.")
        return
    
    # PDF upload section
    with st.sidebar:
        pdf_docs = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf']
        )
        process_pdf_uploads(pdf_docs, model_name, api_key)
        
        # Reset button
        if st.button("🔄 Reset Chat"):
            st.session_state.conversation_history = []
            st.session_state.vector_store = None
            st.session_state.pdf_names = []
            st.rerun()
    
    # Display chat history
    if len(st.session_state.conversation_history) > 0:
        display_conversation_history(st.session_state.conversation_history)
    
    # Question input
    if prompt := st.chat_input("💬 Ask a question about your PDFs..."):
        handle_user_input(
            prompt,
            model_name,
            api_key,
            st.session_state.conversation_history
        )
        st.rerun()


if __name__ == "__main__":
    main()
