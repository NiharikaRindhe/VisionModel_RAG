import streamlit as st
import os
import shutil
from rag_pipeline import VectorStore, QwenGenerator, load_documents

st.set_page_config(page_title="Qwen3-VL Multimodal RAG", layout="wide")

DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

# Separate caching for Model (Heavy) and VectorStore (Medium/Stateful)
@st.cache_resource
def load_qwen_model():
    st.sidebar.info("Loading Qwen2-VL Model (One-time)...")
    try:
        gen = QwenGenerator() 
        return gen
    except Exception as e:
        st.error(f"Failed to load Qwen model: {e}")
        return None

@st.cache_resource
def get_vector_store():
    # Helper to hold the single instance of VectorStore
    st.sidebar.info("Initializing Embedding Model...")
    vs = VectorStore()
    return vs

def reindex_documents(vs):
    """Reloads all documents from disk and rebuilds the index."""
    with st.spinner("Indexing Documents..."):
        vs.clear()
        docs = load_documents(DOCS_DIR)
        vs.add_documents(docs)
        st.toast(f"Indexed {len(docs)} chunks.", icon="✅")

st.title("Qwen3-VL Multimodal RAG Demo")

# 1. Load Resources
gen_model = load_qwen_model()
vector_store = get_vector_store()

# Initial Indexing check
if vector_store and len(vector_store.documents) == 0:
    reindex_documents(vector_store)

# 2. Sidebar - File Upload
st.sidebar.header("Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or Text", 
    type=['pdf', 'txt'], 
    accept_multiple_files=True
)

if uploaded_files:
    new_files_count = 0
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DOCS_DIR, uploaded_file.name)
        # Check if file exists to avoid unnecessary writes/re-indexing if identical?
        # For simplicity, overwrite and track.
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        new_files_count += 1
    
    if new_files_count > 0:
        st.sidebar.success(f"Saved {new_files_count} files.")
        if st.sidebar.button("Process & Re-index", type="primary"):
            reindex_documents(vector_store)

st.sidebar.markdown("---")
st.sidebar.markdown("### Index Stats")
st.sidebar.text(f"Chunks: {len(vector_store.documents) if vector_store else 0}")
if gen_model:
    st.sidebar.success("Model: Ready")
else:
    st.sidebar.error("Model: Failed")

# 3. Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if gen_model and vector_store:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve
                    retrieved_docs = vector_store.search(prompt, k=3)
                    
                    context_inputs = []
                    image_paths = []
                    seen_images = set()

                    with st.expander("Retrieved Context (Text & Images)"):
                        if not retrieved_docs:
                            st.warning("No relevant documents found.")
                        
                        for i, doc in enumerate(retrieved_docs):
                            st.markdown(f"**Chunk {i+1}** (Source: {doc['source']})")
                            st.text(doc['text'][:300] + "...")
                            
                            if doc['image_path']:
                                # Display image in expander
                                st.image(doc['image_path'], caption=f"Page Image from {doc['source']}", width=300)
                                
                                # Collect for generation
                                if doc['image_path'] not in seen_images:
                                    image_paths.append(doc['image_path'])
                                    seen_images.add(doc['image_path'])
                            
                            context_inputs.append(f"Source: {doc['source']}\nText: {doc['text']}")

                    context_str = "\n\n".join(context_inputs)
                    
                    # Generate with Images
                    rag_prompt = f"Answer the user's question based on the provided context images and text.\n\nContext Text:\n{context_str}\n\nQuestion: {prompt}"
                    
                    response = gen_model.generate(rag_prompt, image_paths=image_paths)
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Generation Error: {e}")
                    st.exception(e)
    else:
        st.error("System is not ready. Please check sidebar status.")
