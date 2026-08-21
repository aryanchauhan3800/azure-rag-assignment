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
    
    # Build context and group sources
    context_text = ""
    source_map = {}
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        doc_name = chunk.get('document_name', 'Unknown')
        context_text += f"\n--- Document {i}: {doc_name} ({chunk.get('department', 'Unknown')}) ---\n"
        context_text += chunk.get('text', '') + "\n"
        
        score = chunk.get('score', 0.0)
        meta = chunk.get('source_metadata', 'N/A')
        
        if doc_name not in source_map:
            source_map[doc_name] = {
                "document_name": doc_name,
                "metadata": meta,
                "score": score,
                "chunk_count": 1
            }
        else:
            source_map[doc_name]["chunk_count"] += 1
            # Keep the max score and its associated metadata
            if score > source_map[doc_name]["score"]:
                source_map[doc_name]["score"] = score
                source_map[doc_name]["metadata"] = meta
                
    sources = list(source_map.values())
            
    system_prompt = (
        "You are an assistant answering questions based strictly on the provided company documents. "
        "Your instructions:\n"
        "- Answer ONLY using the provided context.\n"
        "- Prioritize the document/chunk that directly answers the user's specific question.\n"
        "- Do NOT combine different policy categories (e.g. Travel Policy vs Expense/Client Policy) unless the user explicitly asks for a comparison.\n"
        "- If retrieved documents contain related but different policies, do not merge them into one answer. Answer based only on the specific policy requested.\n"
        "- Do not include limits or values from unrelated policies in your answer.\n"
        "- If the answer is not present in the context, say exactly:\n"
        "'I could not find this information in the provided documents.'\n"
        "- Do not invent facts or infer values from related policies."
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
        
        if "I could not find this information" in answer:
            sources = []
            
        return answer, sources
    except Exception as e:
        raise RuntimeError(f"Chat API Error: Failed to generate answer using deployment '{deployment_name}'. Error: {e}")
