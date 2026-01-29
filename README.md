# 🧠 RAG-Based Question Answering System

A production-ready Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask questions using natural language. Built with FastAPI, Pinecone, Groq, and Sentence Transformers.

![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green)

## ✨ Features

- 📄 **Document Upload**: Support for PDF and TXT files (up to 10MB)
- 🧩 **Intelligent Chunking**: Context-aware text splitting with overlap
- 🔍 **Semantic Search**: Vector similarity search using Pinecone
- 🤖 **AI-Powered Answers**: Fast answer generation using Groq (Llama 3.1)
- 📊 **Real-time Metrics**: Track latency, similarity scores, and performance
- 🎨 **Modern UI**: Sleek glassmorphism design with smooth animations
- ⚡ **Blazing Fast**: Sub-second query responses with Groq
- 🛡️ **Rate Limiting**: Built-in protection against abuse

## 🏗️ Architecture

```
User → Upload Document → Document Processing → Chunking → Embedding
                                                              ↓
User ← LLM Generation ← Context Building ← Retrieval ← Vector Store (Pinecone)
```

## 🚀 Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | FastAPI |
| **Vector Store** | Pinecone (Cloud) |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **LLM** | Groq (Llama 3.1 8B Instant) |
| **Document Parsing** | PyPDF2 |
| **Rate Limiting** | SlowAPI |
| **Deployment** | Vercel (Serverless) |

## 📋 Prerequisites

- Python 3.12+
- Pinecone Account (free tier)
- Groq Account (free tier)

## 🔑 Getting API Keys

### Pinecone

1. Visit <https://www.pinecone.io/>
2. Sign up for free
3. Create a new project
4. Copy your API key from the dashboard

### Groq

1. Visit <https://console.groq.com/>
2. Sign up for free
3. Navigate to API Keys
4. Create and copy your API key

## 💻 Local Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd task
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1

# Groq Configuration
GROQ_API_KEY=your_groq_api_key_here

# Application Settings (Optional)
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_FILE_SIZE_MB=10
```

### 4. Run the Application

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at:

- **Frontend**: <http://localhost:8000>
- **API Docs**: <http://localhost:8000/api/docs>
- **Health Check**: <http://localhost:8000/api/health>

## 📡 API Usage

### Upload Document

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

Response:

```json
{
  "status": "success",
  "document_id": "doc_abc123",
  "filename": "document.pdf",
  "chunks_processed": 45,
  "processing_time_ms": 1234.56
}
```

### Query Documents

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic?",
    "top_k": 5,
    "score_threshold": 0.5
  }'
```

Response:

```json
{
  "answer": "The main topic is...",
  "sources": [
    {
      "chunk_text": "...",
      "document_id": "doc_abc123",
      "similarity_score": 0.89,
      "metadata": {...}
    }
  ],
  "metadata": {
    "retrieval_time_ms": 234.5,
    "generation_time_ms": 567.8,
    "total_time_ms": 802.3,
    "chunks_retrieved": 5
  }
}
```

## 🚢 Deployment (Vercel)

### 1. Install Vercel CLI

```bash
npm install -g vercel
```

### 2. Login to Vercel

```bash
vercel login
```

### 3. Deploy

```bash
vercel --prod
```

### 4. Set Environment Variables

In Vercel dashboard:

1. Go to Project Settings → Environment Variables
2. Add:
   - `PINECONE_API_KEY`
   - `GROQ_API_KEY`
   - `PINECONE_ENVIRONMENT`

## 📊 Performance Metrics

The system tracks and displays:

- **Retrieval Time**: Time to search vector store
- **Generation Time**: LLM answer generation time
- **Total Query Time**: End-to-end latency
- **Similarity Scores**: Relevance of retrieved chunks
- **Chunks Retrieved**: Number of context pieces used

Target Performance:

- Document Upload: <5s for 1MB PDF
- Query Response: <3s end-to-end
- Similarity Threshold: >0.7 for high relevance

## 🧪 Testing

### Manual Testing

1. Start the server: `uvicorn app.main:app --reload`
2. Open <http://localhost:8000>
3. Upload a test document (PDF or TXT)
4. Ask questions about the document
5. Verify answers and sources

### Health Check

```bash
curl http://localhost:8000/api/health
```

## 🎨 UI Features

- **Drag & Drop**: Intuitive file upload
- **Real-time Status**: Live processing feedback
- **Glassmorphism Design**: Modern, sleek aesthetics
- **Smooth Animations**: Enhanced user experience
- **Responsive Layout**: Works on all devices
- **Dark Mode**: Easy on the eyes

## 📁 Project Structure

```
task/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── config.py                  # Configuration settings
│   ├── models.py                  # Pydantic models
│   ├── api/
│   │   ├── upload.py              # Upload endpoint
│   │   └── query.py               # Query endpoint
│   ├── services/
│   │   ├── document_processor.py  # PDF/TXT parsing & chunking
│   │   ├── embeddings.py          # Embedding generation
│   │   ├── vector_store.py        # Pinecone operations
│   │   └── llm.py                 # Groq LLM integration
│   └── utils/
├── static/
│   ├── index.html                 # Frontend UI
│   ├── style.css                  # Styling
│   └── script.js                  # JavaScript logic
├── docs/
│   └── EXPLANATIONS.md            # Technical documentation
├── requirements.txt               # Python dependencies
├── vercel.json                    # Vercel deployment config
├── .env.example                   # Environment template
└── README.md                      # This file
```

## 🔧 Configuration

All settings can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 512 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `MAX_FILE_SIZE_MB` | 10 | Maximum upload size |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `UPLOAD_RATE_LIMIT` | 10/hour | Upload rate limit |
| `QUERY_RATE_LIMIT` | 60/hour | Query rate limit |

## 🐛 Troubleshooting

### Pinecone Connection Issues

- Verify API key is correct
- Check your Pinecone environment/region
- Ensure index exists (auto-created on first run)

### Groq API Errors

- Verify API key is valid
- Check rate limits (free tier has generous limits)
- Ensure model `llama-3.1-8b-instant` is available

### Upload Failures

- Check file size (<10MB)
- Verify file format (PDF or TXT only)
- Ensure file contains extractable text

## 📄 License

MIT License - Feel free to use this project for your interview!

## 👨‍💻 Author

Built as an AI internship assignment demonstrating:

- RAG system architecture
- Production-ready code
- Modern UI/UX design
- API development best practices

---

**Note**: This is a demonstration project built for an AI internship interview. All technologies used are free tier and suitable for learning and evaluation purposes.
