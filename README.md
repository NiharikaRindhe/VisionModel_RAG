# Qwen3-VL Multimodal RAG Demo  
A Streamlit Application for Text + Image Retrieval-Augmented Generation

This project provides a **fully local multimodal RAG (Retrieval-Augmented Generation)** workflow powered by:

- **Qwen2-VL / Qwen3-VL** for multimodal (text + image) reasoning  
- A custom **VectorStore** for embedding + retrieval  
- Automatic extraction of **text chunks** and **page images** from PDFs  
- Streamlit for an interactive chat interface  
- Persistent document storage + automatic re-indexing  

You can upload PDFs or text files, index them, and chat with the system. If PDFs contain images, the app will send both **text chunks** *and* **retrieved page images** to Qwen3-VL for richer multimodal understanding.

---

## 🚀 Features

### 🔍 Multimodal RAG
- Retrieves **text chunks**
- Retrieves **associated page images**  
- Sends both to Qwen3-VL for improved context reasoning

### 📄 Document Support
- PDF (converted into text chunks + page images)
- TXT

### ⚡ Automatic Indexing
- Rebuild vector store at any time
- New files saved to a persistent folder (`./documents`)

### 💬 Chat Interface
- Memory of conversation (Streamlit session)
- Qwen generator handles multimodal input

### 📦 Cached Model Loading
Uses `st.cache_resource` to:
- Load Qwen model only once  
- Keep a persistent VectorStore instance  

---

## 📁 Project Structure

To run the code - 
streamlit run app.py

