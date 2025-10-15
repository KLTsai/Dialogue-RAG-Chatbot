from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import os

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataEmbeddingConfig:
    root_dir: Path
    data_path: Path
    model_name: str
    text_column: str

@dataclass(frozen=True)
class VectorStorageConfig:
    data_path: Path
    embedding_dim: int

@dataclass
class GeminiConfig:
    """RAG 系統配置"""
    # Gemini API 設定
    gemini_api_key: str
    gemini_model: str
    gemini_temperature: float
    gemini_max_output_tokens: int
    
    def __post_init__(self):
        if not self.gemini_api_key:
            raise ValueError("請設定 GEMINI_API_KEY 環境變數或在配置中提供 API key")
        
@dataclass
class RAGSystemConfig:
    max_retrieval_docs: int

class RetrieveDecision(str, Enum):
    """檢索決策枚舉"""
    YES = "yes"
    NO = "no" 
    CONTINUE = "continue"

class IsREL(str, Enum):
    """相關性判斷枚舉"""
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"

class IsSUP(str, Enum):
    """支撐性判斷枚舉"""
    FULLY_SUPPORTED = "fully supported"
    PARTIALLY_SUPPORTED = "partially supported"
    NO_SUPPORT = "no support"

class IsUSE(int, Enum):
    """有用性評分枚舉"""
    VERY_USEFUL = 5
    USEFUL = 4
    MODERATELY_USEFUL = 3
    LESS_USEFUL = 2
    NOT_USEFUL = 1