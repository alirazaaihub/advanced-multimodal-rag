# 🔬 Advanced Multi-Modal RAG System

A production-grade Retrieval-Augmented Generation pipeline that goes beyond basic RAG — handling **text, tables, and images** from PDFs with advanced retrieval techniques including Hybrid Search, RRF Fusion, Query Expansion, and Cross-Encoder Reranking.

---

## 🚀 What Makes This Different

Basic RAG systems just split text and search. This system:

- Extracts **text, tables, and images** separately from PDFs
- Uses **AI summaries** to make tables and images searchable
- Expands your query into **3 variations** for better recall
- Combines **dense + sparse retrieval** (Hybrid Search)
- Fuses results using **Reciprocal Rank Fusion (RRF)**
- **Reranks** final results with a Cross-Encoder for precision
- Maintains **short-term, entity, and long-term memory** across conversations

**Example:**
> Upload "Attention Is All You Need" paper  
> Ask "What is the architecture of the Transformer model?"  
> System expands query → retrieves from text + tables + figures → reranks → gives precise answer

---

## ✨ Features

- 📄 **Multi-Modal PDF Parsing** — extracts text, tables (HTML), and images separately
- 🤖 **AI Summarization** — LLM generates searchable descriptions for tables and images
- 🔍 **Hybrid Search** — Dense (ChromaDB) + Sparse (BM25) retrieval combined
- 🔀 **RRF Fusion** — Reciprocal Rank Fusion merges results from multiple queries and retrievers
- 📝 **Query Expansion** — LLM generates 3 query variations to improve recall
- 🏆 **Cross-Encoder Reranking** — `ms-marco-MiniLM-L-6-v2` reranks top results for precision
- 🧠 **Three Memory Layers** — Summary memory + Entity memory + Vector long-term memory
- ⚡ **LangGraph Pipeline** — clean, modular node-based agent flow

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | LangGraph |
| LLM | Groq (llama-3.1-8b-instant) |
| Embeddings | HuggingFace (BAAI/bge-base-en-v1.5) |
| Vector Store | ChromaDB |
| PDF Parsing | Unstructured (hi-res strategy) |
| Sparse Retrieval | BM25Retriever (LangChain) |
| Reranker | CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| Short-Term Memory | ConversationSummaryBufferMemory |
| Entity Memory | ConversationEntityMemory |
| Language | Python 3.11+ |

---

## ⚙️ System Architecture

### Ingestion Pipeline

```
PDF File
    ↓
Unstructured Partition (hi-res)
    ↓
Chunk by Title
    ↓
Separate: Text | Tables | Images
    ↓
AI Summary (for tables & images)
    ↓
ChromaDB Vector Store
```

### Retrieval & Generation Pipeline

```
User Query
    ↓
┌──────────────────┐
│  Query Expansion │  → LLM generates 3 query variations
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Hybrid Retrieve  │  → Dense (ChromaDB) + Sparse (BM25) per query
│   + RRF Fusion   │  → Fuse all results with Reciprocal Rank Fusion
└────────┬─────────┘
         ↓
┌──────────────────┐
│    Reranking     │  → CrossEncoder scores & reranks top 10 → top 5
└────────┬─────────┘
         ↓
┌──────────────────┐
│    Generate      │  → LLM answers using context + memory
└────────┬─────────┘
         ↓
      Response + Memory Update
```

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/alirazaaihub/advanced-multimodal-rag
cd advanced-multimodal-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at: https://console.groq.com

---

## ▶️ Usage

**Step 1 — Ingest your PDF:**

```bash
python ingest.py
```

> Edit `ingest.py` and set your PDF path:
> ```python
> run_pipeline("your_document.pdf")
> ```

**Step 2 — Start the chatbot:**

```bash
python agent.py
```

**Step 3 — Chat:**

```
🚀 Advanced Memory + RAG Chatbot Started

User: What is the Transformer architecture?
Bot: The Transformer model consists of an encoder-decoder structure...

User: What do the attention score tables show?
Bot: Based on the extracted tables, the attention scores indicate...

User: exit
```

---

## 📁 Project Structure

```
advanced-multimodal-rag/
│
├── ingest.py           # PDF parsing, chunking, AI summarization, vector store
├── agent.py            # LangGraph agent with hybrid retrieval + reranking
├── requirements.txt    # Project dependencies
├── .env                # API keys (do not commit)
├── .env.example        # Template for environment variables
├── db/                 # ChromaDB vector store for documents (auto-created)
├── memory_db/          # ChromaDB vector store for long-term memory (auto-created)
└── README.md
```

---

## 📋 Dependencies

```
langchain
langchain-groq
langchain-huggingface
langchain-chroma
langchain-community
langgraph
sentence-transformers
unstructured[all-docs]
chromadb
pydantic
python-dotenv
```

Install all:
```bash
pip install langchain langchain-groq langchain-huggingface langchain-chroma langchain-community langgraph sentence-transformers "unstructured[all-docs]" chromadb pydantic python-dotenv
```

> **Note:** `unstructured[all-docs]` requires `poppler` and `tesseract` for hi-res PDF parsing.  
> Install on Windows: [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) | [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

---

## 🧩 Key Concepts Used

| Concept | How It's Used |
|--------|----------------|
| **Query Expansion** | LLM generates 3 variants of user query → better recall |
| **Dense Retrieval** | ChromaDB semantic similarity search (top 10 per query) |
| **Sparse Retrieval** | BM25 keyword-based search (top 10 per query) |
| **RRF Fusion** | Combines dense + sparse results using rank-based scoring |
| **Cross-Encoder Reranking** | Scores query-document pairs for final precision ranking |
| **Multi-Modal Extraction** | Unstructured separates text, tables, images from PDF |
| **AI Table/Image Summary** | LLM converts tables and images into searchable text |
| **Summary Memory** | Keeps compressed conversation history (max 200 tokens) |
| **Entity Memory** | Tracks named entities mentioned in conversation |
| **Vector Long-Term Memory** | Stores full conversation turns in ChromaDB |

---

## 🙋 About

Built by **Ali raza** — an 18-year-old self-taught AI developer from Pakistan.  
This is part of my Agentic AI portfolio built using LangChain, LangGraph, and Groq.

📌 [LinkedIn](https://www.linkedin.com/in/ali-raza-7124a0403/) • [GitHub](https://github.com/alirazaaihub)

---

## 📄 License

MIT License — feel free to use and modify.
