import os
from dotenv import load_dotenv
from dialogue_rag_chatbot.constants import *
from dialogue_rag_chatbot.utils.common import read_yaml, create_directories
from dialogue_rag_chatbot.entity import (
    DataIngestionConfig,
    DataEmbeddingConfig,
    VectorStorageConfig,
    GeminiConfig,
    RAGSystemConfig
)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_embedding_config(self) -> DataEmbeddingConfig:
        config = self.config.data_embedding

        create_directories([config.root_dir])
        data_embedding_config = DataEmbeddingConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            model_name = config.model_name,
            text_column= config.text_column
        )

        return data_embedding_config
    
    def get_vector_storage_config(self)-> VectorStorageConfig:

        config = self.config.vector_storage
        vector_storage_config = VectorStorageConfig(
            data_path=config.data_path,
            embedding_dim=config.embedding_dim
        )

        return vector_storage_config
    
    def get_gemini_config(self)-> GeminiConfig:
        config = self.config.gemini
        load_dotenv()
        gemini_config = GeminiConfig(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=config.gemini_model,
            gemini_temperature= config.gemini_temperature,
            gemini_max_output_tokens=config.gemini_max_output_tokens
        )

        return gemini_config

    def get_rag_sys_config(self)-> RAGSystemConfig:
        config = self.config.RAG_system
        rag_sys_config = RAGSystemConfig(
            max_retrieval_docs = config.max_retrieval_docs
        )

        return rag_sys_config