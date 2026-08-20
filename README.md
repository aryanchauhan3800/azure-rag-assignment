# Azure RAG Assignment

A production-ready Retrieval-Augmented Generation (RAG) pipeline built with Python, Azure OpenAI, and Azure AI Search.

## Architecture
- **Ingestion**: Parses PDF, DOCX, and XLSX files, chunking them into semantic segments while retaining row, sheet, and page metadata.
- **Embedding & Indexing**: Generates vector embeddings for chunks using Azure OpenAI and indexes them into Azure AI Search using a custom HNSW schema.
- **Retrieval**: Performs HNSW vector similarity search to find relevant context.
- **Generation**: Uses an Azure OpenAI Chat model to generate strictly grounded answers with source attribution.

## Directory Structure
- `data/`: Sample raw documents organized by department.
- `src/ingestion/`: Modules to parse and chunk diverse document types.
- `src/retrieval/`: Search index creation, embedding generation, and vector retrieval.
- `src/generation/`: LLM integration with strict grounding prompts.
- `src/rag.py`: Main CLI for end-to-end question answering.
- `tests/`: Fully mocked unit test suite.

## Setup Instructions
1. Setup a Python virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Configure your Azure variables in a `.env` file (see `.env.example` for reference). DO NOT commit `.env`.
   ```env
   AZURE_SEARCH_ENDPOINT="https://<your-service>.search.windows.net"
   AZURE_SEARCH_ADMIN_KEY="<your-admin-key>"
   AZURE_SEARCH_INDEX_NAME="rag-index"

   AZURE_OPENAI_ENDPOINT="https://<your-resource>.services.ai.azure.com/openai/v1"
   AZURE_OPENAI_API_KEY="<your-api-key>"
   AZURE_OPENAI_API_VERSION="2024-10-21"
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
   AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4.1-mini" 
   ```

## Usage

### 1. Ingestion & Indexing (Run when data changes)
Parses the `data/` directory and synchronizes chunks with Azure AI Search:
```bash
.venv/bin/python src/retrieval/indexer.py
```

### 2. End-to-End RAG CLI
Ask questions against your documents. It retrieves chunks, builds a context, and prompts the LLM for a grounded answer.
```bash
PYTHONPATH=. .venv/bin/python src/rag.py "What is the travel meal limit?"
```

## Testing
Unit tests execute locally and are fully mocked to prevent unexpected Azure costs.
```bash
PYTHONPATH=. .venv/bin/python -m pytest
```

## Security
- The `.env` file containing API keys is ignored by Git.
- Code defensively handles missing configuration via `ValueError`.
- Avoid hardcoding any secrets in Python logic.
