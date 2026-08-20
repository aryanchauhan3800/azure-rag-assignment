import pytest
from src.generation.router import is_conversational, get_conversational_response

def test_is_conversational_true():
    assert is_conversational("Hi") == True
    assert is_conversational("Hello!") == True
    assert is_conversational("What can you do?") == True
    assert is_conversational("Thanks") == True

def test_is_conversational_false():
    assert is_conversational("What is the travel meal limit?") == False
    assert is_conversational("How many days of leave do I get?") == False
    assert is_conversational("Tell me about the code of conduct") == False

def test_get_conversational_response():
    resp = get_conversational_response()
    assert "Hello!" in resp
    assert "company documents" in resp
