import os
from legal_rag_chatbot.logging import logger
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from typing import List, Dict, Any, Tuple
from legal_rag_chatbot.entity import DataEmbeddingConfig

class EmbeddingModel:
    """Granite embedding model using SentenceTransformers"""
    
    def __init__(self, config: DataEmbeddingConfig):
        try:
            self.config = config
            self.model = SentenceTransformer(self.config.model_name)
            logger.info(f"Granite embedding model '{self.model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise
    
    def encode(self, texts):
        """Encode texts into embeddings"""
        try:
            # Granite embedding models return 768-dimensional vectors
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.info(f"Encoded {len(texts)} texts into embeddings of shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise


class DataEmbeddingTransformation:
    """
    Transforms text data into embeddings using a specified model.
    """
    def __init__(self, config: DataEmbeddingConfig):
        self.config = config
        self.embedding_model = EmbeddingModel(config=config)

    def _generate_embeddings(self, batch: Dict[str, List]):
        """
        Takes a batch of examples and generates embeddings for the specified text column.
        This method is designed to be used with dataset.map().
        """
        texts = batch[self.config.text_column]
        embeddings = self.embedding_model.encode(texts)
        
        return {"embedding": embeddings.tolist()}

    def convert(self, save_path: str = "samsum_dataset"):
        """
        Loads the dataset, generates embeddings for the text column,
        and saves the processed dataset to disk.
        """
        logger.info(f"Loading dataset from '{self.config.data_path}'...")
        
        dataset = load_dataset(self.config.data_path, split="train") 
        
        logger.info(f"Generating embeddings for column '{self.config.text_column}'...")
       
        dataset_with_embeddings = dataset.map(
            self._generate_embeddings, 
            batched=True, 
            batch_size=32
        )
        
        output_path = os.path.join(self.config.root_dir, save_path)
        logger.info(f"Saving dataset with embeddings to '{output_path}'...")
        dataset_with_embeddings.save_to_disk(output_path)
        
        logger.info("Data embedding transformation complete.")
        return dataset_with_embeddings