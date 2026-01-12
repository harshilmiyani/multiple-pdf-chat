# 📚 RAG PDF Chatbot

<div align="center">

**A production-ready Streamlit application for intelligent document Q&A using Retrieval-Augmented Generation (RAG)**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2+-green.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/harshilmiyani/multiple-pdf-chat)

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Configuration](#-configuration)

**[🚀 Live Demo](https://multiple-pdf-chat.harshilmiyani.com/)** • **[📂 View Source Code](https://github.com/harshilmiyani/multiple-pdf-chat)**

</div>

---

## 📖 Introduction

**RAG PDF Chatbot** is an intelligent document question-answering system that allows users to upload PDF documents and ask natural language questions about their content. Built with state-of-the-art RAG (Retrieval-Augmented Generation) technology, it combines:

- **Semantic Search**: FAISS vector database for accurate document retrieval
- **LLM Integration**: Google Gemini 2.0 Flash for intelligent answer generation
- **Smart Chunking**: Optimized text processing for better context understanding
- **Client-Side Security**: Your API keys stay in your browser - never on the server

### What is RAG?

RAG (Retrieval-Augmented Generation) enhances LLM responses by:
1. **Retrieving** relevant context from your documents
2. **Augmenting** the LLM prompt with this context
3. **Generating** accurate, context-aware answers

This approach ensures answers are grounded in your actual documents, reducing hallucinations and improving accuracy.

---

## 🚀 Features

### Core Features

- ✅ **Multi-PDF Support**: Upload and process multiple PDF files simultaneously
- ✅ **Intelligent Chunking**: Optimized 1000-character chunks with 200-character overlap for better context retention
- ✅ **Semantic Search**: FAISS-based vector store with HuggingFace embeddings for accurate document retrieval
- ✅ **Quality Filtering**: Similarity score-based filtering (threshold < 1.5) to ensure relevant results
- ✅ **Conversation History**: Track all conversations with timestamps and download as CSV
- ✅ **Client-Side Only**: Your API key stays in your browser - completely secure

### Technical Features

- 🔧 **Production Ready**: Comprehensive error handling, logging, and configuration management
- 📊 **Performance Optimized**: Efficient vector storage and retrieval with configurable parameters
- 🎨 **Clean UI**: Simple, intuitive Streamlit interface with emoji indicators
- 🔒 **Secure**: No server-side API key storage - each user provides their own key
- 📝 **Well Documented**: Type hints, docstrings, and clear code structure
- 🛠️ **Developer Friendly**: Modular architecture, easy to extend and customize

---

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed ([Download Python](https://www.python.org/downloads/))
- **Google API Key** ([Get one here](https://ai.google.dev/))
- **4GB+ RAM** recommended for processing large PDFs
- **Git** (optional, for cloning the repository)

---

## 🛠️ Installation

### Method 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/harshilmiyani/multiple-pdf-chat.git
cd multiple-pdf-chat

# Run the setup script
chmod +x setup.sh
./setup.sh

# Start the application
streamlit run app.py
```

### Method 2: Manual Setup

#### Step 1: Clone the Repository

```bash
git clone https://github.com/harshilmiyani/multiple-pdf-chat.git
cd multiple-pdf-chat
```

#### Step 2: Create Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

#### Step 4: Verify Installation

```bash
# Check if Streamlit is installed
streamlit --version

# Check if LangChain is installed
python -c "import langchain; print(f'LangChain {langchain.__version__}')"
```

---

## 🎯 Usage

### Try the Live Demo

🌐 **[Try it online](https://multiple-pdf-chat.harshilmiyani.com/)** - No installation required! Just upload your PDFs and start asking questions.

### Quick Start

1. **Start the Application**
   ```bash
   streamlit run app.py
   ```
   The app will open at `http://localhost:8501`

2. **Get Your API Key**
   - Visit [Google AI Studio](https://ai.google.dev/)
   - Sign in with your Google account
   - Create a new API key
   - Copy the key

3. **Enter API Key**
   - Open the sidebar in the app
   - Paste your Google API key in the "API Key" field
   - Your key is stored only in your browser session

4. **Upload PDFs**
   - Click "Choose PDF files" in the sidebar
   - Select one or more PDF files
   - Click "✅ Process PDFs" button
   - Wait for "PDFs processed successfully!" message

5. **Ask Questions**
   - Type your question in the chat input at the bottom
   - Press Enter or click Send
   - Get instant answers based on your PDF content!

### Example Questions

- "What is the main topic of this document?"
- "Summarize the key points"
- "What are the conclusions?"
- "Explain the methodology used"
- "List all the important dates mentioned"

---

## 📁 Project Structure

```
multiple-pdf-chat/
├── app.py                 # Main Streamlit application (UI & logic)
├── config.py              # Configuration management (settings)
├── constants.py           # Application constants (messages, templates)
├── utils.py               # Utility functions (PDF processing, chunking)
├── requirements.txt       # Python dependencies with version ranges
├── setup.sh              # Automated setup script
├── .gitignore            # Git ignore rules
├── README.md             # This file
│
└── faiss_index/          # FAISS vector store (auto-generated, gitignored)
    ├── index.faiss
    └── index.pkl
```

### File Descriptions

- **`app.py`**: Main application entry point, handles UI and user interactions
- **`config.py`**: Centralized configuration (chunk sizes, model settings, paths)
- **`constants.py`**: All string constants (prompts, error messages, UI text)
- **`utils.py`**: Core functions (PDF text extraction, chunking, vector store creation)
- **`requirements.txt`**: Python package dependencies with version constraints

---

## ⚙️ Configuration

### API Key Setup

**This is a client-side only application.** Each user provides their own Google API key:

1. Get your API key from [Google AI Studio](https://ai.google.dev/)
2. Enter it in the sidebar when running the application
3. Your API key is stored only in your browser session (never sent to servers)

### Application Settings

Edit `config.py` to customize the application:

#### Chunking Configuration
```python
CHUNK_SIZE = 1000              # Size of text chunks (characters)
CHUNK_OVERLAP = 200            # Overlap between chunks (characters)
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # Splitting strategy
```

#### Retrieval Configuration
```python
MAX_DOCUMENTS_RETRIEVED = 8           # Number of documents to retrieve
SIMILARITY_SCORE_THRESHOLD = 1.5      # Quality filter threshold
MIN_DOCUMENTS_FALLBACK = 3            # Minimum documents if threshold too strict
MAX_CONTEXT_LENGTH = 8000             # Max context length (characters)
```

#### Model Configuration
```python
DEFAULT_MODEL = "gemini-2.0-flash"    # Google Gemini model
DEFAULT_TEMPERATURE = 0.1             # Lower = more deterministic
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
```

---

## 🔧 Architecture

### System Overview

```
┌─────────────┐
│   PDF File  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Text Extraction│ (PyPDF2)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Text Chunking  │ (LangChain RecursiveCharacterTextSplitter)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   Embeddings    │ (HuggingFace sentence-transformers)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Vector Store   │ (FAISS)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐      ┌──────────────┐
│  User Question  │─────▶│  Similarity  │
└─────────────────┘      │    Search     │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │   Retrieve   │
                         │  Top Chunks │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │   Context    │
                         │  + Question  │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │   Gemini     │
                         │   Generate   │
                         │    Answer    │
                         └──────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | Streamlit | Web UI framework |
| **LLM** | Google Gemini 2.0 Flash | Answer generation |
| **Embeddings** | HuggingFace sentence-transformers | Text vectorization |
| **Vector Store** | FAISS (CPU) | Fast similarity search |
| **PDF Processing** | PyPDF2 | Text extraction |
| **RAG Pipeline** | LangChain | Orchestration |

### Key Components

1. **PDF Processing** (`utils.py`)
   - Extracts text from PDF files using PyPDF2
   - Handles multiple PDFs and error cases

2. **Text Chunking** (`utils.py`)
   - Splits text into optimized chunks (1000 chars, 200 overlap)
   - Uses LangChain's RecursiveCharacterTextSplitter
   - Smart separators for natural boundaries

3. **Embedding** (`utils.py`)
   - Creates vector embeddings using HuggingFace models
   - Normalizes embeddings for better cosine similarity
   - Uses CPU for compatibility

4. **Vector Store** (`utils.py`)
   - Stores embeddings in FAISS for fast retrieval
   - Saves/loads index locally
   - Efficient similarity search

5. **Retrieval** (`app.py`)
   - Finds relevant document chunks using similarity search
   - Filters by quality threshold
   - Combines top results into context

6. **Generation** (`app.py`)
   - Formats prompt with context and question
   - Uses Google Gemini to generate answers
   - Extracts and displays response

---

## 📊 Performance Optimizations

### Current Settings

- **Chunk Size**: 1000 characters (optimal for context retention)
- **Chunk Overlap**: 200 characters (20% overlap for continuity)
- **Document Retrieval**: Top 8 documents with quality filtering
- **Similarity Threshold**: Score < 1.5 (filters irrelevant chunks)
- **Context Limit**: 8000 characters (prevents token overflow)
- **Temperature**: 0.1 (more deterministic, accurate answers)

### Performance Tips

- **Large PDFs**: Process in batches or increase chunk size
- **Memory Issues**: Reduce `MAX_DOCUMENTS_RETRIEVED` in `config.py`
- **Speed**: Use GPU for embeddings (modify `EMBEDDING_DEVICE` in `config.py`)

---

## 🐛 Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError

**Problem**: Missing dependencies

**Solution**:
```bash
pip install -r requirements.txt
```

#### 2. API Key Error

**Problem**: Invalid or missing API key

**Solutions**:
- Verify your API key at [Google AI Studio](https://ai.google.dev/)
- Check API key permissions and quotas
- Ensure you've entered the key in the sidebar
- Try creating a new API key

#### 3. PDF Reading Error

**Problem**: Cannot extract text from PDF

**Solutions**:
- Ensure PDFs are not corrupted
- Check if PDFs contain extractable text (not just images)
- Try with a different PDF file
- For scanned PDFs, use OCR tools first

#### 4. Memory Issues

**Problem**: Out of memory when processing large PDFs

**Solutions**:
- Reduce `MAX_DOCUMENTS_RETRIEVED` in `config.py`
- Process PDFs one at a time
- Increase chunk size to reduce number of chunks
- Close other applications to free memory

#### 5. Dependency Conflicts

**Problem**: Version conflicts during installation

**Solution**:
```bash
# Create fresh virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### Debugging

Logs are output to the console. Check Streamlit Cloud logs or your terminal for detailed error information.

---

## 🔒 Security

### Client-Side Only Architecture

- ✅ **No Server Storage**: API keys are never stored on the server
- ✅ **Browser Session**: Keys stored only in Streamlit session state
- ✅ **Individual Keys**: Each user provides their own API key
- ✅ **No Data Collection**: No user data is collected or stored

### Best Practices

- Keep your API keys secure
- Rotate API keys regularly
- Never commit API keys to version control
- Review dependencies for security vulnerabilities
- Use environment variables for local development (optional)

---

## 💻 Development

### Code Structure

The codebase follows clean architecture principles:

- **Separation of Concerns**: UI, business logic, and utilities are separated
- **Type Hints**: Full type annotations for better IDE support
- **Error Handling**: Comprehensive try/except blocks with logging
- **Configuration**: Centralized in `config.py`
- **Constants**: All strings in `constants.py`

### Adding New Features

1. **Add Configuration** (`config.py`)
   ```python
   NEW_FEATURE_SETTING = "default_value"
   ```

2. **Add Constants** (`constants.py`)
   ```python
   NEW_MESSAGE: Final[str] = "Your message here"
   ```

3. **Add Utilities** (`utils.py`)
   ```python
   def new_utility_function(param: str) -> str:
       """Docstring here."""
       # Implementation
   ```

4. **Update UI** (`app.py`)
   ```python
   # Add new UI elements and integrate utilities
   ```

### Running in Development Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with auto-reload
streamlit run app.py --server.runOnSave=true
```

### Code Quality

- ✅ Type hints throughout
- ✅ Docstrings for all functions
- ✅ Error handling with logging
- ✅ Consistent naming conventions
- ✅ Modular architecture

---

## 🧪 Testing

### Manual Testing Checklist

- [ ] PDF upload and processing
- [ ] Question answering with single PDF
- [ ] Question answering with multiple PDFs
- [ ] Conversation history tracking
- [ ] CSV download functionality
- [ ] Error handling (invalid PDF, missing API key)
- [ ] Reset functionality
- [ ] API key validation

### Testing Different Scenarios

1. **Small PDFs** (< 10 pages)
2. **Large PDFs** (> 100 pages)
3. **Multiple PDFs** (5+ files)
4. **Complex Questions** (multi-part questions)
5. **Edge Cases** (no answer in context, empty PDFs)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the Repository**
   ```bash
   git clone https://github.com/harshilmiyani/multiple-pdf-chat.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Follow the code style
   - Add type hints
   - Update documentation
   - Add error handling

4. **Test Your Changes**
   - Test all functionality
   - Check for errors
   - Verify UI works correctly

5. **Submit a Pull Request**
   - Describe your changes
   - Reference any issues
   - Include screenshots if UI changes

### Contribution Guidelines

- Write clean, documented code
- Follow existing code style
- Add tests for new features
- Update README if needed
- Keep commits atomic and meaningful

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Developer

**Harshil Miyani**

- 🔗 [GitHub](https://github.com/harshilmiyani)
- 💼 [LinkedIn](https://www.linkedin.com/in/harshilmiyani/)
- 🌐 [Website](https://harshilmiyani.com)

---

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) - RAG pipeline tools
- [HuggingFace](https://huggingface.co/) - Embedding models
- [Google AI](https://ai.google.dev/) - Gemini LLM
- [Streamlit](https://streamlit.io/) - Web framework
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search

---

## 🆘 Support

- **Issues**: [Open an issue](https://github.com/harshilmiyani/multiple-pdf-chat/issues)
- **Questions**: Contact via [LinkedIn](https://www.linkedin.com/in/harshilmiyani/)
- **Documentation**: Check this README and code comments

---

<div align="center">

**Made with ❤️ using Streamlit, LangChain, and Google Gemini**

⭐ Star this repo if you find it useful!

</div>
