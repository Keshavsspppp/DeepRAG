import os
import io
import pickle
import numpy as np
from pypdf import PdfReader
from groq import Groq

# Global cache for sentence transformer to avoid re-loading
_embedding_model = None

def get_embedding_model():
    """
    Lazy loads and caches the SentenceTransformer model to save memory and start-up time.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2 is a lightweight model (~90MB) that performs well for semantic search
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


def chunk_text(text, chunk_size=300, overlap=50):
    """
    Splits text into chunks of `chunk_size` words with `overlap` words of overlap.
    """
    if not text or chunk_size <= 0:
        return []
        
    words = text.strip().split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
        
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        # Slide window forward (taking overlap into account)
        start += (chunk_size - overlap)
        
    return chunks


def process_document(file_name, file_bytes, file_type, chunk_size=300, overlap=50):
    """
    Parses PDF or TXT bytes and chunks the text page-by-page (for PDF) or fully (for TXT).
    Returns a list of chunk dicts:
        {
            "id": str,
            "text": str,
            "metadata": {
                "source": str,
                "page": int or None,
                "type": str
            }
        }
    """
    chunks_list = []
    
    if file_type.lower() == "pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if not page_text:
                    continue
                
                # Chunk page text
                page_chunks = chunk_text(page_text, chunk_size, overlap)
                for idx, c_text in enumerate(page_chunks):
                    chunks_list.append({
                        "id": f"{file_name}_p{page_num}_c{idx}",
                        "text": c_text,
                        "metadata": {
                            "source": file_name,
                            "page": page_num,
                            "type": "pdf"
                        }
                    })
        except Exception as e:
            raise ValueError(f"Failed to parse PDF '{file_name}': {str(e)}")
            
    else:  # Default to plain text
        try:
            text_content = file_bytes.decode("utf-8", errors="ignore").strip()
            if text_content:
                text_chunks = chunk_text(text_content, chunk_size, overlap)
                for idx, c_text in enumerate(text_chunks):
                    chunks_list.append({
                        "id": f"{file_name}_c{idx}",
                        "text": c_text,
                        "metadata": {
                            "source": file_name,
                            "page": None,
                            "type": "txt"
                        }
                    })
        except Exception as e:
            raise ValueError(f"Failed to parse TXT file '{file_name}': {str(e)}")
            
    return chunks_list


def embed_chunks(chunks):
    """
    Generates embeddings for a list of chunk dictionaries using sentence-transformers.
    Returns: numpy array of shape (num_chunks, embedding_dim)
    """
    if not chunks:
        return np.empty((0, 384), dtype=np.float32)
        
    model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings


class CustomVectorStore:
    """
    A lightweight, pure-Python vector database implementing cosine similarity search.
    """
    def __init__(self):
        self.chunks = []       # List of dicts (chunks)
        self.embeddings = None # numpy array (num_chunks, embedding_dim)

    def add_chunks(self, chunks, embeddings):
        """
        Adds text chunks and their precomputed embeddings to the vector store.
        """
        if not chunks:
            return
            
        self.chunks.extend(chunks)
        new_embeddings = np.array(embeddings, dtype=np.float32)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def similarity_search(self, query_embedding, k=3, filter_docs=None):
        """
        Computes cosine similarity between a query embedding and all stored document embeddings.
        Returns a list of matching chunks with similarity scores.
        
        Args:
            query_embedding: np.ndarray of shape (embedding_dim,)
            k: int, max number of results to return
            filter_docs: list of str, source filenames to restrict search to
        """
        if not self.chunks or self.embeddings is None:
            return []
            
        # Flat 1D query array
        q = np.array(query_embedding, dtype=np.float32).flatten()
        
        # Cosine similarity formula: A . B / (||A|| * ||B||)
        dot_products = np.dot(self.embeddings, q)
        norms = np.linalg.norm(self.embeddings, axis=1)
        q_norm = np.linalg.norm(q)
        
        # Avoid divide-by-zero errors
        norms[norms == 0] = 1e-10
        if q_norm == 0:
            q_norm = 1e-10
            
        similarities = dot_products / (norms * q_norm)
        
        # Sort scores in descending order
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in sorted_indices:
            chunk = self.chunks[idx]
            score = similarities[idx]
            
            # Apply file filter if specified
            if filter_docs and chunk["metadata"].get("source") not in filter_docs:
                continue
                
            results.append({
                "chunk": chunk,
                "score": float(score)
            })
            
            if len(results) >= k:
                break
                
        return results

    def save(self, filepath):
        """
        Persists the vector database to a pickle file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                "chunks": self.chunks,
                "embeddings": self.embeddings
            }, f)

    def load(self, filepath):
        """
        Loads the vector database from a pickle file.
        """
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data.get("chunks", [])
                self.embeddings = data.get("embeddings", None)
            return True
        except Exception:
            return False

    def clear(self):
        """
        Resets the vector database.
        """
        self.chunks = []
        self.embeddings = None


def query_groq_llm(query, retrieved_chunks, api_key, model_name="llama3-8b-8192"):
    """
    Constructs a RAG prompt and queries the Groq API for a grounded response.
    """
    try:
        client = Groq(api_key=api_key)
        
        # Build strict context representation for the prompt
        context_str = ""
        for idx, res in enumerate(retrieved_chunks):
            chunk = res["chunk"]
            score = res["score"]
            source = chunk["metadata"]["source"]
            page_info = f", Page {chunk['metadata']['page']}" if chunk["metadata"].get("page") else ""
            
            context_str += f"--- Source Document: {source}{page_info} (Semantic Similarity Score: {score:.4f}) ---\n"
            context_str += f"{chunk['text']}\n\n"
            
        system_prompt = (
            "You are an expert enterprise RAG assistant. Your task is to answer the user's question "
            "based strictly on the provided Context Blocks. Do not extrapolate, assume, or use external knowledge.\n\n"
            "Strict Guidelines:\n"
            "1. Grounding: Every claim you make MUST be directly supported by the context below. If there are multiple documents, synthesize them together.\n"
            "2. Citations: At the end of the sentences or sections where you reference facts from a document, you MUST insert inline citations "
            "using the exact format: [Source: <filename>, Page: <page_number>] (or simply [Source: <filename>] if page is not available). Example: 'The registration deadline is March 5th [Source: academic_handbook.pdf, Page 12].'\n"
            "3. Hallucination Prevention: If the provided Context Blocks do NOT contain enough information to answer the question, state: "
            "'I'm sorry, but I couldn't find the answer to this question in the uploaded documents.' and do not attempt to supply general answers or guess.\n\n"
            f"Retrieved Document Context:\n{context_str}"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,  # Zero temperature to maximize factual accuracy
            max_tokens=1024
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ **Groq API Error**: {str(e)}\n\nPlease check your Groq API Key and internet connection."
