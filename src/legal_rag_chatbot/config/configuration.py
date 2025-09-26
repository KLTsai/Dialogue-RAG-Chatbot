# from dataclasses import dataclass

# @dataclass
# class RAGConfig:
#     chunk_size: int
#     chunk_overlap: int
#     top_k_retrieval: int
#     top_k_ranking: int
#     embedding_model_name: str = "distilbert-base-uncased"
#     llm_model_name: str = "gpt2"
from legal_rag_chatbot.constants import *
from legal_rag_chatbot.utils.common import read_yaml, create_directories
from legal_rag_chatbot.entity import (
    DataIngestionConfig,
    DataEmbeddingConfig
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

    # def get_data_loading_config(self) -> DataLoadingConfig:
    #     config = self.config.data_loading
        
    #     create_directories([config.root_dir])

    #     data_validation_config = DataLoadingConfig(
    #         root_dir=config.root_dir,
    #         ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES
    #     )

    #     return data_validation_config
    
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