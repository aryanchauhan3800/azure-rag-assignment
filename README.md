# Azure RAG Assignment

A production-ready Retrieval-Augmented Generation (RAG) pipeline built with Python, Azure OpenAI, and Azure AI Search. Features Streamlit UI, hybrid search, automated evaluation, and CI/CD via GitHub Actions.

## Architecture
```text
Documents
 ↓
Ingestion (PDF/DOCX/XLSX to Chunks)
 ↓
Azure OpenAI Embeddings (text-embedding-3-small)
 ↓
Azure AI Search (rag-index)
 ↓
Vector/Hybrid Retrieval (HNSW + BM25)
 ↓
Optional Reranking (Future Enhancement)
 ↓
Azure OpenAI Chat (gpt-4.1-mini)
 ↓
Grounded Answer & Sources
```

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
   SEARCH_MODE="hybrid"
   ```

## Usage

### 1. Ingestion & Indexing
Run this when data changes to synchronize chunks with Azure AI Search:
```bash
.venv/bin/python src/retrieval/indexer.py
```

### 2. End-to-End CLI
Ask questions via the terminal:
```bash
PYTHONPATH=. .venv/bin/python src/rag.py "What is the travel meal limit?"
```
*Example Out-of-Scope:*
```bash
PYTHONPATH=. .venv/bin/python src/rag.py "What is the company policy on bringing pets to the office?"
```
*(The system is prompted strictly to refuse hallucination).*

### 3. Streamlit UI
Launch the interactive web interface:
```bash
.venv/bin/streamlit run app.py
```

## Testing & CI/CD
Unit tests execute locally and are fully mocked to prevent unexpected Azure costs.
```bash
PYTHONPATH=. .venv/bin/python -m pytest
```
GitHub Actions is configured (`.github/workflows/tests.yml`) to automatically run the test suite on pushes and PRs to `main`.

## Evaluation
A lightweight evaluation framework tests retrieval accuracy and groundedness on real Azure services.
```bash
PYTHONPATH=. .venv/bin/python evaluation/evaluate.py
```

## Advanced Features
- **Hybrid Search**: Enabled by default via `SEARCH_MODE="hybrid"`. Queries are run both semantically (embeddings) and lexically (keyword) simultaneously for higher recall.
- **Reranking**: Azure AI Search offers a native Semantic Ranker. This is deferred as a **Future Enhancement** because it requires specific Azure service tiers (Standard) and enabling semantic configurations on the index, which incurs extra costs.

## Security
- The `.env` file containing API keys is ignored by Git.
- Code defensively handles missing configuration.
