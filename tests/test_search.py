import pytest
from unittest.mock import patch, MagicMock
from src.retrieval.search import retrieve_documents

@patch("src.retrieval.search.get_azure_search_client")
@patch("src.retrieval.search.get_openai_client")
@patch("os.environ.get")
def test_retrieve_documents(mock_env_get, mock_get_openai, mock_get_search):
    def env_side_effect(key, default=None):
        if key == "AZURE_OPENAI_EMBEDDING_DEPLOYMENT":
            return "dummy-embedding"
        if key == "AZURE_SEARCH_INDEX_NAME":
            return "dummy-index"
        return default
    mock_env_get.side_effect = env_side_effect
    
    # Mock embedding creation
    mock_openai_client = MagicMock()
    mock_get_openai.return_value = mock_openai_client
    mock_response = MagicMock()
    mock_data = MagicMock()
    mock_data.embedding = [0.1, 0.2]
    mock_response.data = [mock_data]
    mock_openai_client.embeddings.create.return_value = mock_response
    
    # Mock search client
    mock_search_client = MagicMock()
    mock_get_search.return_value = mock_search_client
    mock_search_client.search.return_value = [
        {"id": "1", "document_name": "doc1.pdf", "department": "HR", "source_metadata": "1", "text": "text1", "@search.score": 0.95}
    ]
    
    docs = retrieve_documents("query", top_k=2)
    assert len(docs) == 1
    assert docs[0]["id"] == "1"
    assert docs[0]["score"] == 0.95
    assert docs[0]["document_name"] == "doc1.pdf"
    
    # Verify OpenAI client call
    mock_openai_client.embeddings.create.assert_called_once()
    args, kwargs = mock_openai_client.embeddings.create.call_args
    assert kwargs["model"] == "dummy-embedding"
    assert kwargs["input"] == ["query"]
    
    # Verify Search client call
    mock_search_client.search.assert_called_once()
    args, kwargs = mock_search_client.search.call_args
    assert len(kwargs["vector_queries"]) == 1
    assert kwargs["vector_queries"][0].k_nearest_neighbors == 2
    assert kwargs["search_text"] == "query" # Default is hybrid
