# Legal RAG Chatbot Implementation
# 法律諮詢 RAG Chatbot 實作
"""
This implementation demonstrates a comprehensive RAG (Retrieval-Augmented Generation) system
for legal consultation chatbot as required in the KKCompany Data Scientist interview.

Key Features:
1. Modular design with clear separation of concerns
2. Type hints for better code clarity
3. Comprehensive error handling
4. Logging for debugging and monitoring
5. Configuration management
6. Proper documentation

Core Components:
- DocumentLoader: Load and preprocess legal documents
- TextSplitter: Split documents into manageable chunks
- EmbeddingModel: Convert text to vector representations
- VectorStore: Store and retrieve document embeddings
- Retriever: Find relevant documents based on query
- LLMRanker: Rank retrieved documents using LLM
- ResponseGenerator: Generate final responses
"""

import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration dataclass
@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name: str = "gpt-3.5-turbo"  # Can be replaced with open-source models
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_retrieval: int = 5
    top_k_ranking: int = 3
    temperature: float = 0.0
    max_tokens: int = 512

# Abstract base classes for extensibility
class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        pass

class BaseLLM(ABC):
    """Abstract base class for language models"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response"""
        pass

# Document processing components
class DocumentLoader:
    """Load and preprocess legal documents"""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.pdf', '.md']
        logger.info("DocumentLoader initialized")
    
    def load_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Load documents from file paths
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of document dictionaries with content and metadata
        """
        documents = []
        
        for file_path in file_paths:
            try:
                path = Path(file_path)
                if not path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                if path.suffix not in self.supported_formats:
                    logger.warning(f"Unsupported format: {path.suffix}")
                    continue
                
                # For demonstration, we'll handle text files
                if path.suffix == '.txt':
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    documents.append({
                        'content': content,
                        'metadata': {
                            'source': str(path),
                            'filename': path.name,
                            'file_type': path.suffix[1:]
                        }
                    })
                    
                logger.info(f"Loaded document: {path.name}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

class TextSplitter:
    """Split documents into chunks for better retrieval"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"TextSplitter initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into chunks
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        for doc in documents:
            content = doc['content']
            doc_chunks = self._split_text(content)
            
            for i, chunk_text in enumerate(doc_chunks):
                chunks.append({
                    'content': chunk_text,
                    'metadata': {
                        **doc['metadata'],
                        'chunk_id': i,
                        'total_chunks': len(doc_chunks)
                    }
                })
        
        logger.info(f"Total chunks created: {len(chunks)}")
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + self.chunk_size // 2, end - 100), -1):
                    if text[i] in '.!?。！？':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            
            if end >= len(text):
                break
        
        return chunks

# Simple embedding model implementation (for demonstration)
class SimpleEmbeddingModel(BaseEmbeddingModel):
    """Simple embedding model for demonstration purposes"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_dim = 384  # Typical dimension for MiniLM
        logger.info(f"EmbeddingModel initialized: {model_name}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings
        Note: This is a simplified implementation for demonstration.
        In practice, you would use actual sentence-transformers library.
        """
        # Simulate embeddings (in real implementation, use actual model)
        embeddings = []
        for text in texts:
            # Create a simple hash-based embedding for demonstration
            embedding = self._create_dummy_embedding(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _create_dummy_embedding(self, text: str) -> np.ndarray:
        """Create dummy embedding based on text characteristics"""
        # This is just for demonstration - replace with actual model
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self.embedding_dim)

# Vector store for similarity search
class VectorStore:
    """Store and retrieve document embeddings"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.embeddings = []
        self.documents = []
        logger.info("VectorStore initialized")
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents and their embeddings to the store"""
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")
        
        self.embeddings.extend(embeddings.tolist())
        self.documents.extend(documents)
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform similarity search
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if not self.embeddings:
            return []
        
        # Calculate cosine similarity
        embeddings_matrix = np.array(self.embeddings)
        similarities = np.dot(embeddings_matrix, query_embedding) / (
            np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        logger.info(f"Retrieved {len(results)} documents from vector store")
        return results

# Simple LLM implementation (for demonstration)
class SimpleLLM(BaseLLM):
    """Simple LLM implementation for demonstration"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        logger.info(f"LLM initialized: {model_name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text response
        Note: This is a placeholder. In practice, integrate with actual LLM APIs.
        """
        # Simulate LLM response for demonstration
        if "排序" in prompt or "ranking" in prompt.lower():
            return "根據相關性，文件排序為: 1, 2, 3"
        else:
            return "根據提供的法律條文，我的建議是..."

# RAG pipeline components
class Retriever:
    """Retrieve relevant documents based on query"""
    
    def __init__(self, vector_store: VectorStore, embedding_model: BaseEmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        logger.info("Retriever initialized")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for query
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search similar documents
            results = self.vector_store.similarity_search(query_embedding, top_k)
            
            # Extract documents
            documents = [doc for doc, score in results]
            
            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return []

class LLMRanker:
    """Rank retrieved documents using LLM"""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        logger.info("LLMRanker initialized")
    
    def rank_documents(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Rank documents using LLM
        
        Args:
            query: User query
            documents: List of retrieved documents
            top_k: Number of top documents to return
            
        Returns:
            List of ranked documents
        """
        if not documents:
            return []
        
        try:
            # Create ranking prompt
            ranking_prompt = self._create_ranking_prompt(query, documents)
            
            # Get LLM ranking
            ranking_response = self.llm.generate(ranking_prompt)
            
            # For demonstration, return top_k documents
            # In practice, parse LLM response to get actual rankings
            ranked_docs = documents[:top_k]
            
            logger.info(f"Ranked {len(ranked_docs)} documents")
            return ranked_docs
            
        except Exception as e:
            logger.error(f"Error in ranking: {str(e)}")
            return documents[:top_k]
    
    def _create_ranking_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Create prompt for document ranking"""
        prompt = f"""請根據查詢內容對以下文件進行相關性排序：

查詢: {query}

文件:
"""
        for i, doc in enumerate(documents):
            content = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
            prompt += f"{i+1}. {content}\n"
        
        prompt += "\n請按相關性從高到低排序，並說明原因。"
        return prompt

class ResponseGenerator:
    """Generate final responses based on context"""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        logger.info("ResponseGenerator initialized")
    
    def generate_response(self, query: str, context_documents: List[Dict[str, Any]]) -> str:
        """
        Generate response based on query and context
        
        Args:
            query: User query
            context_documents: List of relevant documents
            
        Returns:
            Generated response
        """
        try:
            # Create generation prompt
            generation_prompt = self._create_generation_prompt(query, context_documents)
            
            # Generate response
            response = self.llm.generate(generation_prompt)
            
            logger.info("Response generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            return "抱歉，我無法為您的問題提供回答。請重新表述您的問題。"
    
    def _create_generation_prompt(self, query: str, context_documents: List[Dict[str, Any]]) -> str:
        """Create prompt for response generation"""
        prompt = f"""你是一個專業的法律諮詢助手。請根據提供的法律條文回答用戶問題。

背景資料:
"""
        for doc in context_documents:
            content = doc['content']
            source = doc['metadata'].get('filename', '未知來源')
            prompt += f"來源: {source}\n內容: {content}\n\n"
        
        prompt += f"""
用戶問題: {query}

請基於上述法律條文提供準確、專業的回答。如果條文中沒有相關資訊，請誠實說明並建議諮詢專業律師。
"""
        return prompt

# Main RAG Pipeline
class LegalRAGChatbot:
    """Main RAG pipeline for legal chatbot"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(config.chunk_size, config.chunk_overlap)
        self.embedding_model = SimpleEmbeddingModel(config.embedding_model_name)
        self.vector_store = VectorStore()
        self.llm = SimpleLLM(config.llm_model_name)
        self.retriever = Retriever(self.vector_store, self.embedding_model)
        self.ranker = LLMRanker(self.llm)
        self.generator = ResponseGenerator(self.llm)
        
        logger.info("LegalRAGChatbot initialized")
    
    def build_knowledge_base(self, document_paths: List[str]):
        """
        Build knowledge base from documents
        
        Args:
            document_paths: List of document file paths
        """
        logger.info("Building knowledge base...")
        
        # Load documents
        documents = self.document_loader.load_documents(document_paths)
        
        if not documents:
            logger.warning("No documents loaded")
            return
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Generate embeddings
        chunk_texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts)
        
        # Add to vector store
        self.vector_store.add_documents(chunks, embeddings)
        
        logger.info("Knowledge base built successfully")
    
    def query(self, user_query: str) -> str:
        """
        Process user query and return response
        
        Args:
            user_query: User's legal question
            
        Returns:
            Generated response
        """
        logger.info(f"Processing query: {user_query}")
        
        try:
            # Step 1: Retrieval
            retrieved_docs = self.retriever.retrieve(
                user_query, 
                self.config.top_k_retrieval
            )
            
            if not retrieved_docs:
                return "抱歉，我找不到相關的法律資訊來回答您的問題。"
            
            # Step 2: Ranking
            ranked_docs = self.ranker.rank_documents(
                user_query, 
                retrieved_docs, 
                self.config.top_k_ranking
            )
            
            # Step 3: Generation
            response = self.generator.generate_response(user_query, ranked_docs)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return "系統發生錯誤，請稍後再試。"

# Example usage and testing
def main():
    """Main function demonstrating the RAG system"""
    # Configuration
    config = RAGConfig(
        chunk_size=500,
        chunk_overlap=50,
        top_k_retrieval=5,
        top_k_ranking=3
    )
    
    # Initialize chatbot
    chatbot = LegalRAGChatbot(config)
    
    # Example documents (in practice, load from actual legal documents)
    sample_documents = [
        "sample_legal_doc1.txt",
        "sample_legal_doc2.txt"
    ]
    
    # Build knowledge base
    # chatbot.build_knowledge_base(sample_documents)
    
    # Example queries
    test_queries = [
        "合約違約的法律後果是什麼？",
        "如何處理租賃糾紛？",
        "公司法中關於股東權益的規定是什麼？"
    ]
    
    # Process queries
    for query in test_queries:
        print(f"\n用戶問題: {query}")
        response = chatbot.query(query)
        print(f"回答: {response}")

if __name__ == "__main__":
    main()