"""
RAG Processor - Lite & Complete Modes
=====================================
Lite: Only user documents (fast, low RAM)
Complete: UMLS + documents (heavy, requires ChromaDB UMLS)
"""

import os
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# ChromaDB
import chromadb
from sentence_transformers import SentenceTransformer

# Transformers (direct use, not langchain to avoid compatibility issues)
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


class RAGMode(Enum):
    """RAG operation modes."""
    LITE = "lite"         # Only user documents (fast)
    COMPLETE = "complete" # UMLS + documents (heavy)


class LLMType(Enum):
    """LLM backend - Solo Llama2 via Ollama."""
    OLLAMA_LLAMA2 = "ollama_llama2"


@dataclass
class RAGResponse:
    """Response from RAG system."""
    answer: str
    sources: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    llm_used: str
    mode: str


class RAGProcessor:
    """
    RAG Processor - Sistema de Recuperación Aumentada.
    
    Modos:
    - LITE: Solo busca en documentos subidos (rápido)
    - COMPLETE: Busca en UMLS + documentos (requiere ChromaDB UMLS)
    
    LLMs soportados:
    - Ollama (Phi, Llama2) - Requiere: ollama serve
    """
    
    # Configuración de LLM (solo Llama2 via Ollama)
    LLM_CONFIGS = {
        LLMType.OLLAMA_LLAMA2: {
            "type": "ollama",
            "model": "llama2",
            "url": "http://localhost:11434/api/generate",
            "description": "Llama2 via Ollama (~4GB)"
        }
    }
    
    # HuggingFace API token (set via UI or environment)
    hf_token: str = None
    
    def __init__(
        self,
        mode: RAGMode = RAGMode.LITE,
        chromadb_path: str = "Datasets/chromadb_umls",
        docs_chromadb_path: str = "Datasets/rag_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        default_llm: LLMType = LLMType.OLLAMA_LLAMA2
    ):
        """
        Initialize RAG Processor.
        
        Args:
            mode: LITE (docs only) or COMPLETE (UMLS + docs)
            chromadb_path: Path to UMLS ChromaDB (only for COMPLETE mode)
            docs_chromadb_path: Path to documents ChromaDB (always loaded)
            embedding_model: Model for embeddings
            default_llm: Default LLM to use
        """
        self.mode = mode
        self.chromadb_path = chromadb_path
        self.docs_chromadb_path = docs_chromadb_path
        self.default_llm = default_llm
        
        # Initialize embeddings first
        print(f"⏳ Cargando embeddings...")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB collections based on mode
        self._init_collections()
        
        # LLM cache (lazy loading)
        self._llm_cache = {}
        self._tokenizer = None
        
        print(f"✅ RAGProcessor iniciado")
        print(f"   Modo: {mode.value.upper()}")
        print(f"   LLM: {default_llm.value}")
    
    def _init_collections(self):
        """Initialize ChromaDB collections based on mode."""
        
        # Always load documents collection (user uploads)
        os.makedirs(self.docs_chromadb_path, exist_ok=True)
        self.docs_client = chromadb.PersistentClient(path=self.docs_chromadb_path)
        self.docs_collection = self.docs_client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"📄 Documentos: {self.docs_collection.count()} chunks")
        
        # Only load UMLS in COMPLETE mode
        if self.mode == RAGMode.COMPLETE:
            try:
                self.umls_client = chromadb.PersistentClient(path=self.chromadb_path)
                self.umls_collection = self.umls_client.get_collection("umls_concepts")
                print(f"📚 UMLS: {self.umls_collection.count():,} conceptos")
            except Exception as e:
                print(f"⚠️ UMLS no disponible: {e}")
                self.umls_collection = None
        else:
            self.umls_collection = None
            print("ℹ️ Modo LITE: UMLS no cargado")
    
    def add_document(self, text: str, metadata: dict = None) -> str:
        """
        Add a document chunk to the RAG collection.
        
        Args:
            text: Document text
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        doc_id = f"doc_{self.docs_collection.count() + 1}"
        embedding = self.embedder.encode([text])[0]
        
        self.docs_collection.add(
            documents=[text],
            embeddings=[embedding.tolist()],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
        
        return doc_id
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search in user documents collection."""
        
        if self.docs_collection.count() == 0:
            return []
        
        embedding = self.embedder.encode([query])[0]
        
        results = self.docs_collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted = []
        for i in range(len(results['documents'][0])):
            formatted.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "score": 1 - results['distances'][0][i],
                "source": "Documentos"
            })
        
        return formatted
    
    def search_umls(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search in UMLS collection (only in COMPLETE mode)."""
        
        if not self.umls_collection:
            return []
        
        embedding = self.embedder.encode([query])[0]
        
        results = self.umls_collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted = []
        for i in range(len(results['documents'][0])):
            formatted.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "score": 1 - results['distances'][0][i],
                "source": "UMLS"
            })
        
        return formatted
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search based on current mode.
        
        LITE: Only documents
        COMPLETE: UMLS + documents
        """
        doc_results = self.search_documents(query, n_results)
        
        if self.mode == RAGMode.COMPLETE:
            umls_results = self.search_umls(query, n_results)
            all_results = umls_results + doc_results
            all_results.sort(key=lambda x: x["score"], reverse=True)
            return all_results[:n_results * 2]
        
        return doc_results
    
    def _load_llm(self, llm_type: LLMType):
        """Load LLM with caching. Optimized for Apple M1."""
        
        if llm_type in self._llm_cache:
            return self._llm_cache[llm_type]
        
        config = self.LLM_CONFIGS[llm_type]
        
        # Detect best device: MPS (Mac), CUDA (NVIDIA), or CPU
        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float32  # MPS works better with float32
        elif torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
        
        print(f"🔧 Usando dispositivo: {device}")
        
        if "adapter_path" in config:
            # Fine-tuned model with LoRA
            try:
                from peft import PeftModel
                
                base = AutoModelForCausalLM.from_pretrained(
                    config["base_model"],
                    torch_dtype=dtype,
                    device_map=device if device != "mps" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                if device == "mps":
                    base = base.to(device)
                
                llm = PeftModel.from_pretrained(base, config["adapter_path"])
                self._tokenizer = AutoTokenizer.from_pretrained(
                    config["adapter_path"],
                    trust_remote_code=True
                )
                
            except Exception as e:
                print(f"⚠️ Error cargando adapter: {e}")
                return self._load_llm(LLMType.SIMPLE)
        else:
            # Standard model via pipeline - optimized for M1
            llm = pipeline(
                "text-generation",
                model=config["model"],
                torch_dtype=dtype,
                device=device if device != "cpu" else -1,
                trust_remote_code=True,
                model_kwargs={"low_cpu_mem_usage": True}
            )
            self._tokenizer = llm.tokenizer
        
        self._llm_cache[llm_type] = llm
        print(f"✅ LLM cargado: {llm_type.value}")
        
        return llm
    
    def generate(self, query: str, context: List[Dict], llm_type: LLMType = None) -> str:
        """Generate response using LLM. Supports Ollama (fast) and HuggingFace."""
        
        llm_type = llm_type or self.default_llm
        config = self.LLM_CONFIGS[llm_type]
        
        # Build context from ChromaDB results
        context_text = "\n\n".join([
            f"[{c['source']}] {c['content'][:400]}"
            for c in context[:3]
        ])
        
        # Prompt
        prompt = f"""Eres un asistente médico. Responde basándote SOLO en el contexto dado.

CONTEXTO:
{context_text}

PREGUNTA: {query}

RESPUESTA:"""
        
        try:
            # HuggingFace Inference API (cloud)
            if config.get("type") == "hf_api":
                return self._generate_hf_api(prompt, config)
            
            # Ollama (HTTP request - fast!)
            if config.get("type") == "ollama":
                return self._generate_ollama(prompt, config)
            
            # HuggingFace local models
            llm = self._load_llm(llm_type)
            
            if config.get("type") == "adapter":
                # LoRA model
                inputs = self._tokenizer(prompt, return_tensors="pt")
                if hasattr(llm, 'device'):
                    inputs = {k: v.to(llm.device) for k, v in inputs.items()}
                
                outputs = llm.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id
                )
                response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.split("RESPUESTA:")[-1].strip()
            else:
                # Pipeline model
                result = llm(
                    prompt,
                    max_new_tokens=300,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=llm.tokenizer.eos_token_id
                )
                response = result[0]["generated_text"].split("RESPUESTA:")[-1].strip()
            
            return response
            
        except Exception as e:
            return f"Error generando respuesta: {str(e)}"
    
    def _generate_hf_api(self, prompt: str, config: dict) -> str:
        """Generate response using HuggingFace Inference API (cloud)."""
        
        if not self.hf_token:
            return "❌ Token de HuggingFace requerido. Ingresa tu token en el sidebar."
        
        try:
            response = requests.post(
                config["url"],
                headers={"Authorization": f"Bearer {self.hf_token}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 300,
                        "temperature": 0.3,
                        "return_full_text": False
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "Sin respuesta")
                return str(result)
            elif response.status_code == 401:
                return "❌ Token inválido. Verifica tu token de HuggingFace."
            elif response.status_code == 503:
                return "⏳ Modelo cargando en HuggingFace. Intenta de nuevo en 30 segundos."
            else:
                return f"Error HF API: {response.status_code} - {response.text[:100]}"
                
        except requests.exceptions.Timeout:
            return "❌ Timeout: El servidor de HuggingFace tardó demasiado."
        except Exception as e:
            return f"Error HF API: {str(e)}"
    
    def _generate_ollama(self, prompt: str, config: dict) -> str:
        """Generate response using Ollama (HTTP request to localhost)."""
        
        try:
            response = requests.post(
                config["url"],
                json={
                    "model": config["model"],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 300
                    }
                },
                timeout=180  # 3 minutos para primera carga
            )
            
            if response.status_code == 200:
                return response.json().get("response", "Sin respuesta")
            else:
                return f"Error Ollama: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "❌ Ollama no está corriendo. Ejecuta: ollama serve"
        except requests.exceptions.ReadTimeout:
            return "⏳ Timeout: Llama2 tardó demasiado. Intenta con una pregunta más corta."
        except Exception as e:
            return f"Error Ollama: {str(e)}"
    
    def query(self, question: str, llm_type: LLMType = None) -> RAGResponse:
        """
        Full RAG pipeline.
        
        Args:
            question: User question
            llm_type: LLM to use
            
        Returns:
            RAGResponse with answer and sources
        """
        llm_type = llm_type or self.default_llm
        
        # Search
        sources = self.search(question)
        
        # Generate
        if sources:
            answer = self.generate(question, sources, llm_type)
        else:
            answer = "No encontré información relevante en los documentos cargados."
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            entities=[],
            llm_used=llm_type.value,
            mode=self.mode.value
        )
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            "mode": self.mode.value,
            "documents": self.docs_collection.count(),
            "umls": self.umls_collection.count() if self.umls_collection else 0
        }
    
    def clear_documents(self):
        """Clear user documents collection."""
        self.docs_client.delete_collection("rag_documents")
        self.docs_collection = self.docs_client.create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
        print("🗑️ Documentos eliminados")
