import os
import argparse
from rag_pipeline import VectorStore, QwenGenerator, load_and_chunk_documents

def main():
    print("Initializing Qwen3-VL RAG Pipeline...")
    
    # 1. Setup Vector Store
    vector_store = VectorStore()
    
    # 2. Load and Index Documents
    docs_path = os.path.join(os.path.dirname(__file__), "documents")
    print(f"Loading documents from {docs_path}...")
    chunks = load_and_chunk_documents(docs_path)
    
    if not chunks:
        print("No documents found! creating a sample one just in case...")
        # This should have been handled by previous steps, but safety net
        with open(os.path.join(docs_path, "sample.txt"), "w") as f:
            f.write("Qwen-VL is a multimodal model. RAG is retrieval augmented generation.")
        chunks = load_and_chunk_documents(docs_path)
        
    vector_store.add_documents(chunks)
    
    # 3. Initialize Generator (loads the heavy model)
    try:
        generator = QwenGenerator()
    except Exception as e:
        print(f"Failed to load model. Please ensure you have internet access or the model is cached. Error: {e}")
        return

    # 4. Interactive Loop or Single Query
    questions = [
        "What is Qwen2-VL?",
        "How is it different from previous versions?",
        "What is RAG?"
    ]
    
    print("\n--- Running Sample Queries ---")
    for q in questions:
        print(f"\nQuestion: {q}")
        
        # Retrieve
        retrieved_docs = vector_store.search(q, k=2)
        print("Retrieved Context:")
        for i, doc in enumerate(retrieved_docs):
            print(f"[{i+1}] {doc[:100]}...")
            
        # Context Construction
        context_str = "\n\n".join(retrieved_docs)
        prompt = f"Using the following context, answer the question.\n\nContext:\n{context_str}\n\nQuestion: {q}"
        
        # Generate
        print("Generating answer...")
        answer = generator.generate(prompt)
        print(f"Answer: {answer}")
        
    print("\n--- Done ---")

if __name__ == "__main__":
    main()
