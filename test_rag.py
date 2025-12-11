import os
from rag_pipeline import VectorStore, QwenGenerator, load_documents

def main():
    # 1. Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(base_dir, "documents")
    
    # 2. Load Documents
    print("Loading documents...")
    if not os.path.exists(docs_dir):
        print(f"Error: {docs_dir} does not exist.")
        return

    docs = load_documents(docs_dir)
    if not docs:
        print("No documents found.")
        return
        
    print(f"Loaded {len(docs)} chunks.")
    
    # 3. Initialize Vector Store
    vs = VectorStore()
    vs.add_documents(docs)
    
    # 4. Initialize Generator
    print("Initializing Qwen Model...")
    try:
        gen = QwenGenerator()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 5. Run Test Query
    query = "What was the revenue in Q4 according to the chart?"
    print(f"\nQuery: {query}")
    
    # Retrieve
    retrieved = vs.search(query, k=3)
    
    context_str = ""
    image_paths = []
    seen = set()
    
    print("\nRetrieved Context:")
    for i, doc in enumerate(retrieved):
        print(f"[{i+1}] {doc['source']}")
        context_str += f"Source: {doc['source']}\nText: {doc['text']}\n\n"
        
        if doc['image_path'] and doc['image_path'] not in seen:
            image_paths.append(doc['image_path'])
            seen.add(doc['image_path'])
            print(f"   (Includes Image: {os.path.basename(doc['image_path'])})")
            
    # Generate
    rag_prompt = f"Answer the user's question based on the provided context images and text.\n\nContext Text:\n{context_str}\n\nQuestion: {query}"
    
    print("\nGenerating Answer...")
    answer = gen.generate(rag_prompt, image_paths=image_paths)
    
    print("\n" + "="*50)
    print("MODEL ANSWER:")
    print(answer)
    print("="*50)

if __name__ == "__main__":
    main()
