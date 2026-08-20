import sys
import os
import json

# Add parent directory to path to import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval.search import retrieve_documents
from src.generation.chat import generate_answer

def run_evaluation(questions_file: str):
    print("Starting RAG Evaluation (Live Azure AI Search & OpenAI)...")
    
    with open(questions_file, 'r') as f:
        questions = json.load(f)
        
    total = len(questions)
    passed = 0
    
    for q in questions:
        print(f"\nEvaluating: '{q['question']}'")
        
        # 1. Retrieval
        try:
            chunks = retrieve_documents(q["question"], top_k=3)
        except Exception as e:
            print(f"  [X] Retrieval failed: {e}")
            continue
            
        retrieved_docs = [c["document_name"] for c in chunks]
        
        if q["is_in_scope"]:
            if q["expected_source_document"] not in retrieved_docs:
                print(f"  [X] Failed: Expected source '{q['expected_source_document']}' not in retrieved docs {retrieved_docs}")
                continue
            else:
                print(f"  [+] Retrieval success: Found '{q['expected_source_document']}'")
        
        # 2. Generation
        try:
            answer, _ = generate_answer(q["question"], chunks)
        except Exception as e:
            print(f"  [X] Generation failed: {e}")
            continue
            
        # 3. Validation
        if q["expected_answer_snippet"].lower() in answer.lower():
            print(f"  [+] Generation success: Output contains expected snippet '{q['expected_answer_snippet']}'")
            passed += 1
        else:
            print(f"  [X] Failed: Answer did not contain '{q['expected_answer_snippet']}'.")
            print(f"      Actual Answer: {answer}")

    print(f"\n==========================================")
    print(f"Evaluation Complete: {passed}/{total} Passed")
    print(f"==========================================")
    
    if passed < total:
        sys.exit(1)

if __name__ == "__main__":
    q_file = os.path.join(os.path.dirname(__file__), "questions.json")
    run_evaluation(q_file)
