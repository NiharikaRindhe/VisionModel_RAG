import os
import glob
import torch
import numpy as np
import faiss
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

class VectorStore:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        # Store dicts: {'text': str, 'image_path': str or None, 'source': str}
        self.documents = [] 
    
    def add_documents(self, docs: List[Dict[str, Any]]):
        if not docs:
            return
        texts = [d['text'] for d in docs]
        print(f"Encoding {len(texts)} documents...")
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        self.index.add(embeddings)
        self.documents.extend(docs)
        print(f"Added {len(docs)} documents to index.")

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        try:
            print(f"Searching for: {repr(query)}")
        except Exception:
            print("Searching for query (content suppressed due to encoding error)")
        query_vector = self.encoder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.documents):
                results.append(self.documents[idx])
        return results

    def clear(self):
        print("Clearing vector store...")
        self.index.reset()
        self.documents = []

class QwenGenerator:
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct"):
        print(f"Loading generation model: {model_id}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            # Use float16 for CUDA, float32 for CPU
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id, 
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=dtype
            )
            if self.device == "cpu":
                self.model.to("cpu")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def generate(self, prompt_text: str, image_paths: List[str] = [], max_new_tokens: int = 256) -> str:
        # Construct messages
        content = []
        
        # Add images first
        for img_path in image_paths:
            if img_path and os.path.exists(img_path):
                content.append({"type": "image", "image": img_path})
        
        # Add text
        content.append({"type": "text", "text": prompt_text})
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]

def process_pdf(pdf_path: str, output_image_dir: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    chunks = []
    
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
        
    for i, page in enumerate(doc):
        # Extract text
        text = page.get_text()
        
        # Render image of the page
        pix = page.get_pixmap()
        image_filename = f"{os.path.basename(pdf_path)}_page_{i}.png"
        image_path = os.path.join(output_image_dir, image_filename)
        pix.save(image_path)
        
        # Create chunk
        # If text is empty, use a placeholder so we can still retrieve the image via caption or assumption
        chunk_text = text.strip() if text.strip() else f"Image-only page {i} from {os.path.basename(pdf_path)}"
        
        chunks.append({
            'text': chunk_text,
            'image_path': image_path,
            'source': f"{os.path.basename(pdf_path)} - Page {i}"
        })
        
    return chunks

def load_documents(folder_path: str) -> List[Dict[str, Any]]:
    all_docs = []
    
    # Text files
    for filepath in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            # Simple chunk for txt files
            all_docs.append({
                'text': text, 
                'image_path': None, 
                'source': os.path.basename(filepath)
            })
            
    # PDF files
    images_dir = os.path.join(folder_path, "images")
    for filepath in glob.glob(os.path.join(folder_path, "*.pdf")):
        print(f"Processing PDF: {filepath}")
        pdf_chunks = process_pdf(filepath, images_dir)
        all_docs.extend(pdf_chunks)
        
    return all_docs
