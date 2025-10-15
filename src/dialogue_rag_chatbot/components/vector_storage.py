from dialogue_rag_chatbot.logging import logger
from typing import List, Dict, Any, Tuple
import numpy as np
from dialogue_rag_chatbot.entity import VectorStorageConfig

class VectorStorage:
    """Store and retrieve document embeddings"""
    
    def __init__(self, config: VectorStorageConfig):
        self.config = config
        self.embedding_dim = self.config.embedding_dim
        self.embeddings = []
        self.documents = []
        logger.info(f"VectorStore initialized with embedding_dim={self.embedding_dim}")
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents and their embeddings to the store"""
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")
        
        self.embeddings.extend(embeddings.tolist())
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Perform similarity search"""
        if not self.embeddings:
            return []
        
        query_embedding = query_embedding.flatten()

        embeddings_matrix = np.array(self.embeddings)
        similarities = np.dot(embeddings_matrix, query_embedding.T) / (
            np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(query_embedding) + 1e-10
        )
        
        # top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        top_indices = np.argpartition(-similarities, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        results = [(self.documents[idx], float(similarities[idx])) for idx in top_indices]
        logger.info(f"Retrieved {len(results)} documents from vector store")
        return results