import os
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)
from openai import AzureOpenAI, OpenAI

# Load environment variables (e.g. from .env file)
load_dotenv()

def sanitize_key(key: str) -> str:
    """Sanitize chunk_id to be a valid Azure Search document key.
    Keys can only contain letters, numbers, dashes, and underscores.
    """
    return re.sub(r'[^a-zA-Z0-9_-]', '_', key)

def get_azure_search_index_client() -> SearchIndexClient:
    endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    admin_key = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
    if not endpoint or not admin_key:
        raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_ADMIN_KEY must be set.")
    return SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(admin_key))

def get_azure_search_client(index_name: str) -> SearchClient:
    endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    admin_key = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
    if not endpoint or not admin_key:
        raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_ADMIN_KEY must be set.")
    return SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(admin_key))

def get_openai_client():
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
    
    if not endpoint:
        raise ValueError("Configuration Error: AZURE_OPENAI_ENDPOINT must be set.")
    if not api_key:
        raise ValueError("Configuration Error: AZURE_OPENAI_API_KEY must be set.")
    if not endpoint.startswith("https://"):
        raise ValueError(f"Configuration Error: Invalid endpoint format. Must start with https://. Got: {endpoint}")
        
    if "/openai/v1" in endpoint:
        # OpenAI-compatible Foundry endpoint
        return OpenAI(base_url=endpoint, api_key=api_key)
    else:
        # Standard Azure OpenAI endpoint
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )

def create_index_if_not_exists(index_name: str, vector_dimensions: int = 1536):
    """Creates the search index with the required schema."""
    index_client = get_azure_search_index_client()
    
    # Check if index exists
    if index_name in [name for name in index_client.list_index_names()]:
        print(f"Index '{index_name}' already exists.")
        return

    # Define schema
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="original_chunk_id", type=SearchFieldDataType.String),
        SearchableField(name="text", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchField(
            name="embedding", 
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
            searchable=True, 
            vector_search_dimensions=vector_dimensions, 
            vector_search_profile_name="myHnswProfile"
        ),
        SimpleField(name="document_name", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="department", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="source_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="source_metadata", type=SearchFieldDataType.String, filterable=True),
    ]

    # Configure Vector Search profile and algorithm
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")]
    )

    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    index_client.create_index(index)
    print(f"Created index: {index_name}")

def generate_embeddings(texts: List[str], deployment_name: str) -> List[List[float]]:
    """Generate embeddings for a list of texts using Azure OpenAI."""
    if not deployment_name:
        raise ValueError("Configuration Error: Embedding deployment name must be provided (AZURE_OPENAI_EMBEDDING_DEPLOYMENT).")
        
    client = get_openai_client()
    try:
        response = client.embeddings.create(input=texts, model=deployment_name)
        return [item.embedding for item in response.data]
    except Exception as e:
        raise RuntimeError(f"API Error: Failed to generate embeddings for deployment '{deployment_name}'. Error: {str(e)}")

def map_chunks_to_documents(chunks: List[Dict[str, Any]], deployment_name: str, batch_size: int = 16) -> List[Dict[str, Any]]:
    """Maps ingestion chunks to Azure Search documents and generates embeddings in batches."""
    documents = []
    
    # Process in batches to respect rate limits and payload sizes
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [chunk["text"] for chunk in batch]
        
        embeddings = generate_embeddings(texts, deployment_name)
        
        for j, chunk in enumerate(batch):
            doc_name = chunk.get("document_name", "")
            doc_ext = os.path.splitext(doc_name)[1].lower().replace(".", "")
            
            # Form metadata string from either page_number, sheet_name, or row_range
            metadata = str(chunk.get("page_number", ""))
            if "row_range" in chunk:
                metadata += f" | rows:{chunk['row_range']}"
            
            doc = {
                "id": sanitize_key(chunk["chunk_id"]),
                "original_chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "embedding": embeddings[j],
                "document_name": doc_name,
                "department": chunk["department"],
                "source_type": doc_ext,
                "source_metadata": metadata
            }
            documents.append(doc)
            
    return documents

def index_chunks(chunks: List[Dict[str, Any]], index_name: str, deployment_name: str):
    """Orchestrates index creation, embedding generation, and document upload."""
    create_index_if_not_exists(index_name)
    
    print(f"Generating embeddings and mapping {len(chunks)} chunks...")
    documents = map_chunks_to_documents(chunks, deployment_name)
    
    search_client = get_azure_search_client(index_name)
    
    # Upload in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        results = search_client.upload_documents(documents=batch)
        print(f"Uploaded batch of {len(results)} documents.")

if __name__ == "__main__":
    # Example usage for manual execution
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.ingestion.ingest import ingest_all_departments
    
    index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME", "rag-index")
    deployment_name = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    
    data_directory = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    chunks = ingest_all_departments(data_directory)
    
    if chunks:
        print(f"Starting indexing for {len(chunks)} chunks...")
        index_chunks(chunks, index_name, deployment_name)
        print("Indexing completed successfully.")
    else:
        print("No chunks found to index.")
