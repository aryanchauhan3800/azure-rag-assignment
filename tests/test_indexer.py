import os
import pytest
from unittest.mock import patch, MagicMock
from src.retrieval.indexer import sanitize_key, map_chunks_to_documents, index_chunks, create_index_if_not_exists

def test_sanitize_key():
    # Azure search keys only allow letters, numbers, dashes, and underscores
    assert sanitize_key("my.doc.pdf_p1_c0") == "my_doc_pdf_p1_c0"
    assert sanitize_key("TravelPolicy.docx_Finance_p1_c0") == "TravelPolicy_docx_Finance_p1_c0"
    assert sanitize_key("good_key-123") == "good_key-123"

@patch("src.retrieval.indexer.get_openai_client")
def test_map_chunks_to_documents(mock_get_openai_client):
    # Mock OpenAI client response
    mock_client = MagicMock()
    mock_get_openai_client.return_value = mock_client
    
    mock_response = MagicMock()
    item1 = MagicMock()
    item1.embedding = [0.1] * 1536
    item2 = MagicMock()
    item2.embedding = [0.2] * 1536
    mock_response.data = [item1, item2]
    
    mock_client.embeddings.create.return_value = mock_response

    chunks = [
        {
            "chunk_id": "doc1.pdf_IT_p1_c0",
            "text": "Hello world",
            "document_name": "doc1.pdf",
            "department": "IT",
            "page_number": 1
        },
        {
            "chunk_id": "doc2.xlsx_Sales_Sheet1_r1-10",
            "text": "Sales data",
            "document_name": "doc2.xlsx",
            "department": "Sales",
            "page_number": "Sheet:Sheet1",
            "row_range": "1-10"
        }
    ]

    docs = map_chunks_to_documents(chunks, "dummy-deployment", batch_size=2)
    
    assert len(docs) == 2
    
    # Assert transformations for PDF
    assert docs[0]["id"] == "doc1_pdf_IT_p1_c0"
    assert docs[0]["original_chunk_id"] == "doc1.pdf_IT_p1_c0"
    assert docs[0]["source_type"] == "pdf"
    assert docs[0]["source_metadata"] == "1"
    assert len(docs[0]["embedding"]) == 1536
    assert docs[0]["embedding"][0] == 0.1
    
    # Assert transformations for XLSX
    assert docs[1]["id"] == "doc2_xlsx_Sales_Sheet1_r1-10"
    assert docs[1]["source_type"] == "xlsx"
    assert docs[1]["source_metadata"] == "Sheet:Sheet1 | rows:1-10"
    assert len(docs[1]["embedding"]) == 1536
    assert docs[1]["embedding"][0] == 0.2
    
    # Verify OpenAI client was called correctly
    mock_client.embeddings.create.assert_called_once()
    args, kwargs = mock_client.embeddings.create.call_args
    assert kwargs["input"] == ["Hello world", "Sales data"]
    assert kwargs["model"] == "dummy-deployment"

@patch("src.retrieval.indexer.get_azure_search_index_client")
def test_create_index_if_not_exists(mock_get_search_index_client):
    mock_client = MagicMock()
    mock_get_search_index_client.return_value = mock_client
    
    # Mocking that the index doesn't exist yet
    mock_client.list_index_names.return_value = ["other-index"]
    
    create_index_if_not_exists("my-index")
    
    mock_client.create_index.assert_called_once()
    args, kwargs = mock_client.create_index.call_args
    assert args[0].name == "my-index"

@patch("src.retrieval.indexer.get_azure_search_client")
@patch("src.retrieval.indexer.map_chunks_to_documents")
@patch("src.retrieval.indexer.create_index_if_not_exists")
def test_index_chunks(mock_create_index, mock_map, mock_get_search_client):
    mock_search_client = MagicMock()
    mock_get_search_client.return_value = mock_search_client
    
    # Mocking map output to return 1 doc
    mock_map.return_value = [{"id": "1", "text": "test"}]
    
    index_chunks([{"dummy": "chunk"}], "dummy-index", "dummy-deployment")
    
    # Check index creation is triggered
    mock_create_index.assert_called_once_with("dummy-index")
    
    # Check document upload is triggered
    mock_search_client.upload_documents.assert_called_once()
    args, kwargs = mock_search_client.upload_documents.call_args
    assert len(kwargs["documents"]) == 1
