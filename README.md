# RAG System

A lightweight Retrieval-Augmented Generation (RAG) pipeline built using local embeddings, ChromaDB, and Groq-hosted LLM inference.

The project is focused on building a simple but scalable document question-answering workflow where textual documents are embedded locally, stored in a vector database, and retrieved dynamically during inference.

## Current Progress

Current implementation supports:

* Loading `.txt` documents from a local directory
* Splitting documents into overlapping chunks
* Local embedding generation using `sentence-transformers`
* Persistent vector storage using ChromaDB
* Semantic retrieval using vector similarity search
* Response generation using Groq API with Llama 3.3 70B
* Environment variable handling through `.env`

The system is currently operating as a CLI-based prototype.

## Project Architecture

```text
Documents -> Chunking -> Embeddings -> ChromaDB
                                          |
                                          v
User Query -> Query Embedding -> Retrieval -> LLM Response
```

## Tech Stack

* Python
* ChromaDB
* Sentence Transformers
* Groq API
* OpenAI Python SDK
* dotenv

## Embedding Model

```python
all-MiniLM-L6-v2
```

Used for lightweight local embedding generation.

## LLM

```python
llama-3.3-70b-versatile
```

Served through Groq API.

## Project Structure

```text
Rag_system/
│
├── app.py
├── .gitignore
├── .env
├── content_dict/
├── chroma_storage/
└── README.md
```

## Setup

### Clone Repository

```bash
git clone https://github.com/Arpansharma7/Rag_system.git
cd Rag_system
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

## Run

```bash
python app.py
```

## Current Limitations

* Only `.txt` files are supported
* No frontend interface yet
* No streaming responses
* No metadata filtering
* Duplicate embeddings may occur if documents are reinserted repeatedly
* Retrieval pipeline is still unoptimized for larger datasets

## Planned Improvements

* PDF and DOCX ingestion
* Web interface
* Better chunking strategies
* Hybrid search
* Context reranking
* Multi-document querying
* Conversation memory
* Docker deployment
* API endpoint integration

## Notes

Large generated vector storage and secret environment files are excluded from version control using `.gitignore`.

## License

This project is currently intended for learning and experimentation purposes.
