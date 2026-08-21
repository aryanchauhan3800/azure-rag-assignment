import argparse
import sys
from src.retrieval.search import retrieve_documents
from src.generation.chat import generate_answer
from src.generation.router import is_conversational, get_conversational_response

def main():
    parser = argparse.ArgumentParser(description="End-to-End Azure RAG Application")
    parser.add_argument("query", type=str, help="The question to ask")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve for context")
    args = parser.parse_args()
    
    if is_conversational(args.query):
        print(f"\n[1] Query routed as conversational: '{args.query}'")
        print("\n" + "="*50)
        print("ANSWER:")
        print(get_conversational_response())
        print("="*50 + "\n")
        return
        
    print(f"\n[1] Retrieving relevant documents for query: '{args.query}'...")
    try:
        chunks = retrieve_documents(args.query, top_k=args.top_k)
    except Exception as e:
        print(f"Retrieval Error: {e}")
        sys.exit(1)
        
    if not chunks:
        print("No documents were retrieved.")
        sys.exit(0)
        
    print(f"    Retrieved {len(chunks)} chunks.")
    
    print(f"\n[2] Generating grounded answer...")
    try:
        answer, sources = generate_answer(args.query, chunks)
        print("\n" + "="*50)
        print("ANSWER:")
        print(answer)
        if sources:
            print("\nSOURCES:")
            for source in sources:
                chunk_count = source.get('chunk_count', 1)
                chunk_text = f"({chunk_count} relevant chunk{'s' if chunk_count > 1 else ''})"
                print(f"- Doc: {source['document_name']} {chunk_text} | Meta: {source['metadata']} | Best Score: {source['score']:.4f}")
        print("="*50 + "\n")
    except Exception as e:
        print(f"Generation Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
