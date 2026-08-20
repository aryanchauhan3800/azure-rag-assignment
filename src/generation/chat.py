import os
from typing import List, Dict, Any, Tuple
from src.retrieval.indexer import get_openai_client

def generate_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """
    Generates a grounded answer using Azure OpenAI, based only on the provided context chunks.
    Returns a tuple of (answer, sources).
    """
    deployment_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
    if not deployment_name:
        raise ValueError("Configuration Error: AZURE_OPENAI_CHAT_DEPLOYMENT must be set in .env")
        
    client = get_openai_client()
    
    # Build context
    context_text = ""
    sources = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_text += f"\n--- Document {i}: {chunk.get('document_name', 'Unknown')} ({chunk.get('department', 'Unknown')}) ---\n"
        context_text += chunk.get('text', '') + "\n"
        
        source_info = {
            "document_name": chunk.get('document_name', 'Unknown'),
            "metadata": chunk.get('source_metadata', 'N/A'),
            "score": chunk.get('score', 0.0)
        }
        if source_info not in sources:
            sources.append(source_info)
            
    system_prompt = (
        "You are an assistant answering questions based strictly on the provided company documents. "
        "Your instructions:\n"
        "- Answer ONLY using the provided context.\n"
        "- If the answer is not present in the context, say:\n"
        "'I could not find this information in the provided documents.'\n"
        "- Do not invent facts.\n"
        "- Do not use prior knowledge outside the provided context."
    )
    
    user_prompt = f"Context Documents:\n{context_text}\n\nUser Question: {query}\n\nAnswer the question using the context above:"
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        answer = response.choices[0].message.content
        return answer, sources
    except Exception as e:
        raise RuntimeError(f"Chat API Error: Failed to generate answer using deployment '{deployment_name}'. Error: {e}")
