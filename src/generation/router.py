import re

# Casual and conversational phrases
CONVERSATIONAL_PATTERNS = [
    r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b",
    r"^(thanks|thank you|thanks a lot)\b",
    r"^who are you\b",
    r"^what can you do\b",
    r"^how are you\b"
]

def is_conversational(query: str) -> bool:
    """
    Checks if a query is a casual/conversational message that should bypass RAG retrieval.
    """
    cleaned_query = query.lower().strip().rstrip("!?.,")
    for pattern in CONVERSATIONAL_PATTERNS:
        if re.search(pattern, cleaned_query):
            return True
    return False

def get_conversational_response() -> str:
    """
    Returns a standard conversational response.
    """
    return "Hello! I can answer questions based on the company documents. What would you like to know?"
