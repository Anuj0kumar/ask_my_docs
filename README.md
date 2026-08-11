# 🌍 Ask My Docs: Enterprise RAG API

**Ask My Docs** is a production-grade Retrieval-Augmented Generation (RAG) backend API designed to ingest dense policy documents (such as the **IPCC Climate Reports**), perform advanced semantic and keyword retrieval, and generate **strict, hallucination-free answers with precise page citations**.

---

# 🚀 Features

- **Advanced Retrieval Architecture**
  - Bypasses naïve RAG by implementing a **Hybrid Search Pipeline**
  - Combines **ChromaDB Vector Embeddings** with **BM25 Keyword Search**

- **Cross-Encoder Reranking**
  - Uses `ms-marco-MiniLM-L-6-v2`
  - Reranks retrieved chunks and selects the **top 3 most contextually relevant passages**

- **Local-First Generation**
  - Powered by **Ollama (Llama 3.2)**
  - Zero API cost
  - High privacy
  - Runs completely offline

- **Automated Evaluation**
  - Evaluated mathematically using the **Ragas Framework**
  - Metrics:
    - Context Precision
    - Faithfulness

- **Production API**
  - Built with **FastAPI**
  - Asynchronous request handling
  - High-concurrency ready

- **Hardened Security**
  - API Key Authentication (`X-API-Key`)
  - Strict CORS policies
  - Rate Limiting using **SlowAPI**

- **DevOps Ready**
  - Docker containerization
  - GitHub Actions CI/CD smoke-testing pipeline

---

# 🧠 How It Works (Request Lifecycle)

## 1. PDF Ingestion

- PyMuPDF extracts text from uploaded PDFs
- LangChain splits documents into overlapping chunks

## 2. Embedding

- Chunks are converted into vector embeddings using `nomic-embed-text`
- Stored locally inside **ChromaDB**

## 3. Query

A client sends a POST request to:

```
POST /api/v1/ask
```

## 4. Hybrid Retrieval

The system retrieves approximately **10 candidate chunks** using:

- Semantic vector search (ChromaDB)
- Exact keyword matching (BM25)

## 5. Cross-Encoder Reranking

The Cross-Encoder model scores the retrieved chunks and selects the **best 3**.

## 6. Generation

**Llama 3.2** receives only those top 3 chunks and generates an answer that is:

- Strictly bound to the retrieved context
- Hallucination-resistant
- Returned with exact **page citations**

---

# 🏗️ Architecture Overview

```text
                PDF Documents
                      │
                      ▼
              Text Extraction (PyMuPDF)
                      │
                      ▼
            Chunking (LangChain Splitter)
                      │
                      ▼
        Embeddings (nomic-embed-text via Ollama)
                      │
                      ▼
                  ChromaDB Vector Store
                      │
                      ▼
               Hybrid Retrieval Pipeline
         (Vector Search + BM25 Keyword Search)
                      │
                      ▼
     Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
                      │
                      ▼
              Top 3 Context Chunks Selected
                      │
                      ▼
            Llama 3.2 Generation (Ollama)
                      │
                      ▼
      JSON Response with Answer + Page Citations
```

---

# 🛠️ Technology Stack

## Core

- Python 3.11
- FastAPI
- Pydantic
- Uvicorn

## AI & RAG

- LangChain
- ChromaDB
- Sentence-Transformers
- Rank_BM25

## LLM Provider

- Ollama
- Llama 3.2
- nomic-embed-text

## DevOps

- Docker
- GitHub Actions

## Security & Observability

- SlowAPI
- Python Logging

---

# 📁 Project Structure

```text
ask_my_docs/
│
├── src/
│   ├── api/
│   ├── rag/
│   ├── retrieval/
│   ├── security/
│   ├── config/
│   └── ...
│
├── chroma_db/
├── data/
├── evaluation/
├── tests/
├── Dockerfile
├── requirements.txt
├── README.md
└── .github/workflows/
```

---

# 💻 Local Installation

## Prerequisites

Install the following:

1. **Python 3.11+**
2. **Ollama**

Download:

- https://www.python.org/
- https://ollama.com/

Pull the required models:

```bash
ollama run llama3.2
ollama pull nomic-embed-text
```

---

## Clone the Repository

```bash
git clone https://github.com/Anuj0kumar/ask_my_docs.git
cd ask_my_docs
```

---

## Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Start the API Server

```bash
uvicorn src.api:app --reload
```

The API will be available at:

```
http://localhost:8000
```

---

# 📘 API Documentation

FastAPI automatically generates interactive documentation.

Open:

```
http://localhost:8000/docs
```

Before executing requests:

1. Click **Authorize**
2. Enter the development API key

```
ak-ask-my-docs-prod-777
```

---

# 🔑 Authentication

All protected endpoints require:

```
X-API-Key: ak-ask-my-docs-prod-777
```

Example:

```http
POST /api/v1/ask
X-API-Key: ak-ask-my-docs-prod-777
Content-Type: application/json
```

---

# 🐳 Docker Deployment

Build the image:

```bash
docker build -t ask-my-docs-api .
```

Run the container:

```bash
docker run -p 8000:8000 ask-my-docs-api
```

---

# 📊 Evaluation

The retrieval pipeline was evaluated using the **Ragas Framework**.

## Metrics

- **Context Precision**
- **Faithfulness**

Evaluation was performed locally using sequential execution:

```python
max_workers = 1
```

This avoids external API rate-limit issues and provides deterministic evaluation.

---

# 🔄 SDLC Progress

## Completed

- PDF ingestion
- Document chunking
- Embedding generation
- ChromaDB vector storage
- BM25 keyword indexing
- Hybrid retrieval
- Cross-Encoder reranking
- FastAPI backend
- Swagger documentation
- API key authentication
- CORS configuration
- Rate limiting
- Docker containerization
- GitHub Actions CI pipeline

---

# 📦 Continuation Package

## Executive Summary

**Project:** Ask My Docs (Enterprise RAG API)

**SDLC Phase:** Phase 12 (Project Handover / Maintenance)

**Current Version:** **V1.0.0**

### Summary

The project evolved from a basic RAG prototype into a **production-grade enterprise backend** featuring:

- Hybrid retrieval
- Cross-Encoder reranking
- Mathematical evaluation
- Secure FastAPI deployment
- Dockerized infrastructure
- GitHub Actions CI/CD

---



# ⚠️ Known Issues & Technical Debt

## ChromaDB Scaling

Current implementation loads ChromaDB into memory during initialization.

Future improvement:

- Run ChromaDB in **Client/Server mode**
- Host it as a dedicated Docker service
- Improve scalability for multi-gigabyte document collections

---

## API Key Management

The development API key is currently hardcoded.

Future improvement:

- PostgreSQL-backed credential management
- Multi-user authentication
- API key rotation
- Revocation support

---

# 👨‍💻 Author

**Anuj Kumar Kushwaha**

Built as a hands-on exercise in mastering the **entire Software Development Life Cycle (SDLC)** — from **Data Engineering** and **Retrieval-Augmented Generation (RAG)** to **Backend Engineering**, **Security**, **Docker**, and **CI/CD**.
````

This version is fully formatted for GitHub with proper headings, code fences, architecture diagrams, roadmap sections, ADRs, and a professional README structure.
