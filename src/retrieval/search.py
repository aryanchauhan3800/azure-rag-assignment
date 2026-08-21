import os
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

from azure.search.documents.models import VectorizedQuery
from src.retrieval.indexer import get_openai_client, get_azure_search_client

load_dotenv()

def retrieve_documents(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieves top_k relevant documents for a given query using vector search."""
    # 1. Get embedding for the query
    deployment_name = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if not deployment_name:
        raise ValueError("Configuration Error: AZURE_OPENAI_EMBEDDING_DEPLOYMENT must be set in .env")
        
    client = get_openai_client()
    try:
        response = client.embeddings.create(input=[query], model=deployment_name)
        query_vector = response.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"Failed to generate query embedding: {e}")

    # 2. Search Azure AI Search
    index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME", "rag-index")
    search_client = get_azure_search_client(index_name)
    
    vector_query = VectorizedQuery(
        vector=query_vector, 
        k_nearest_neighbors=top_k, 
        fields="embedding"
    )
    
    # 3. Determine Search Mode
    search_mode = os.environ.get("SEARCH_MODE", "hybrid").lower()
    search_text = query if search_mode == "hybrid" else None
    
    # Execute the search
    results = search_client.search(
        search_text=search_text,
        vector_queries=[vector_query],
        select=["id", "document_name", "department", "source_metadata", "text"],
        top=top_k
    )
    
    docs = []
    for result in results:
        docs.append({
            "id": result["id"],
            "document_name": result["document_name"],
            "department": result["department"],
            "source_metadata": result["source_metadata"],
            "text": result["text"],
            "score": result["@search.score"]
        })
    return docs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Vector Retrieval")
    parser.add_argument("query", type=str, nargs="?", default="What are the travel limits?", help="The query to search for")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to retrieve")
    args = parser.parse_args()
    
    try:
        docs = retrieve_documents(args.query, top_k=args.top_k)
        print(f"\nTop {args.top_k} Results for Query: '{args.query}'\n" + "="*50)
        for i, doc in enumerate(docs, 1):
            text_snippet = doc["text"][:200].replace("\n", " ") + "..."
            print(f"[{i}] Score: {doc['score']:.4f} | Doc: {doc['document_name']} | Dept: {doc['department']} | Meta: {doc['source_metadata']}")
            print(f"    Text: {text_snippet}\n")
    except Exception as e:
        print(f"Error: {e}")
