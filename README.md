# Azure RAG Assignment
[![Tests](https://github.com/aryanchauhan3800/azure-rag-assignment/actions/workflows/tests.yml/badge.svg)](https://github.com/aryanchauhan3800/azure-rag-assignment/actions/workflows/tests.yml)
A portfolio-quality Retrieval-Augmented Generation (RAG) pipeline built with Python, Azure OpenAI, and Azure AI Search. Features Streamlit UI, hybrid search query routing, automated evaluation, Docker support, and CI/CD via GitHub Actions.

## Live Demo
**Live Demo:** [https://9oqbb9a5drtgxbc6qeghsp.streamlit.app/](https://9oqbb9a5drtgxbc6qeghsp.streamlit.app/) *(Placeholder if deployment URL is not available yet)*

## Screenshots
### Streamlit UI
![Streamlit RAG UI](docs/images/streamlit-ui.png)

### RAG CLI Output
![RAG CLI](docs/images/rag-cli.png)

### Automated Evaluation & Tests
![Evaluation and Tests](docs/images/evaluation-tests.png)

## 1. Project Overview
This project ingests company documents (PDFs, DOCX, XLSX) and builds a fully functional RAG system allowing users to query information conversationally. It incorporates strong anti-hallucination guardrails, table extraction, and intelligent query routing.

## 2. Architecture
```text
Documents (PDFs, DOCX, XLSX)
 ↓
Ingestion (Semantic Chunking & Table Extraction)
 ↓
Azure OpenAI Embeddings (text-embedding-3-small)
 ↓
Azure AI Search (rag-index)
 ↓
Query Routing (Conversational Bypass)
 ↓
Vector/Hybrid Retrieval (HNSW + BM25)
 ↓
Azure OpenAI Chat (gpt-4.1-mini)
 ↓
Grounded Answer & Source Citations
```

## 3. Features
- **Document Ingestion**: Seamless extraction and semantic chunking of text, preserving page and row-level metadata, including full extraction of DOCX tables.
- **Hybrid Retrieval**: Employs both dense vector search and keyword BM25 search to maximize context recall.
- **Query Routing**: Intercepts casual conversational queries (e.g., "Hi", "Thank you") to save API costs and improve UX without unnecessary RAG retrieval.
- **Strict Grounding**: System prompts enforce absolute reliance on retrieved context; out-of-scope questions are explicitly met with "I could not find this information...".
- **Source Grouping**: Answers include clear, grouped citations of specific source documents to ensure transparency.
- **Dockerized Deployment**: Fully containerized Streamlit application for production readiness.

## 4. Technology Stack
- **Language**: Python 3.10+
- **AI Services**: Azure OpenAI (Embeddings & Chat Completions)
- **Search**: Azure AI Search (Vector + Hybrid)
- **UI**: Streamlit
- **Deployment**: Docker
- **Testing & CI**: Pytest, GitHub Actions

## 5. Key Engineering Decisions
- **Hybrid Retrieval**: Combines vector similarity (HNSW) with BM25 keyword search to maximize recall.
- **Query Routing**: A lightweight router intercepts conversational queries (e.g., "Hello") to avoid unnecessary Azure Search calls and reduce latency.
- **Strict Grounding**: System prompts explicitly constrain the LLM to only answer based on retrieved context, ensuring it refuses unsupported queries rather than hallucinating limits from unrelated policies.
- **DOCX Table Extraction**: Standard `python-docx` extraction skips tables, so a custom routine parses DOCX tables natively to ensure highly structured data (like travel per-diems) makes it into the search index.
- **Mocked Unit Tests**: The Pytest suite entirely mocks Azure AI SDKs, allowing GitHub Actions CI to validate logic without requiring actual cloud credentials or incurring costs.
- **Live Evaluation**: A deterministic evaluation suite uses actual Azure resources to validate end-to-end RAG retrieval, factual generation, and semantic validation.

## 6. Setup Instructions
1. Setup a Python virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Configure your Azure variables in a `.env` file (see `.env.example` for reference). **DO NOT commit `.env`.**

### Environment Variables
Requires an Azure AI Search instance and an Azure OpenAI resource.
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

## 7. Streamlit Instructions
Launch the interactive web interface locally:
```bash
.venv/bin/streamlit run app.py
```

## 8. Docker Instructions
The application is fully containerized for production deployment. To build and run the Docker image:

1. **Build the image:**
   ```bash
   docker build -t azure-rag-app .
   ```
2. **Run the container:**
   ```bash
   docker run --env-file .env -p 8501:8501 -d --name rag-app azure-rag-app
   ```
3. Access the Streamlit application at `http://localhost:8501`.

## 9. Testing Results
Unit tests execute locally and are fully mocked to prevent unexpected Azure costs. GitHub Actions is configured to automatically run the test suite on pushes to `main`.
- **Verified Results**: 15/15 unit tests passed.

Run tests:
```bash
PYTHONPATH=. .venv/bin/python -m pytest
```

## 10. Evaluation Results
A lightweight evaluation framework tests retrieval accuracy and groundedness on real Azure services against a ground-truth dataset.
- **Verified Results**: 5/5 live evaluation cases passed.

Run evaluation:
```bash
PYTHONPATH=. .venv/bin/python evaluation/evaluate.py
```

## 11. Security
- The `.env` file containing API keys is ignored by Git (`.gitignore`) and Docker (`.dockerignore`).
- The `Dockerfile` does not hardcode credentials or copy `.env` into the image.
- `.venv`, `__pycache__`, and `.pytest_cache` are fully ignored.
- Code defensively handles missing configuration.
- CI pipelines use mocked services, preventing credential leaks.

## 12. Limitations
- Azure AI Search index requires manual schema management and update scripts when metadata structures change.
- Text chunking may split cross-page tables awkwardly depending on token lengths.

## 13. Future Improvements
- **Semantic Reranking**: Azure AI Search offers a native Semantic Ranker. This is currently deferred as a future enhancement because it requires specific Azure service tiers (Standard) and enabling semantic configurations on the index, which incurs extra infrastructure setup and costs outside this project's scope.
