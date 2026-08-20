import pytest
from unittest.mock import patch, MagicMock
from src.generation.chat import generate_answer

@patch("src.generation.chat.get_openai_client")
@patch("os.environ.get")
def test_generate_answer(mock_env_get, mock_get_openai_client):
    # Mock environment to have a deployment
    def env_side_effect(key, default=None):
        if key == "AZURE_OPENAI_CHAT_DEPLOYMENT":
            return "dummy-chat"
        return default
    mock_env_get.side_effect = env_side_effect

    # Mock AzureOpenAI client
    mock_client = MagicMock()
    mock_get_openai_client.return_value = mock_client
    
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a grounded answer."
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    chunks = [
        {"document_name": "TravelPolicy.docx", "department": "Finance", "source_metadata": "1", "text": "Meals are covered up to $100."}
    ]
    
    answer, sources = generate_answer("What is the meal limit?", chunks)
    
    assert answer == "This is a grounded answer."
    assert len(sources) == 1
    assert sources[0]["document_name"] == "TravelPolicy.docx"
    assert sources[0]["metadata"] == "1"
    
    # Verify API call params
    mock_client.chat.completions.create.assert_called_once()
    args, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == "dummy-chat"
    assert kwargs["temperature"] == 0.0
    assert len(kwargs["messages"]) == 2
    assert "Meals are covered up to $100." in kwargs["messages"][1]["content"]

@patch("os.environ.get")
def test_generate_answer_missing_deployment(mock_env_get):
    # Mock missing deployment
    mock_env_get.return_value = None
    with pytest.raises(ValueError, match="AZURE_OPENAI_CHAT_DEPLOYMENT must be set"):
        generate_answer("query", [])
