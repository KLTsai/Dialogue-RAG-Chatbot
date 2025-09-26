from dataclasses import dataclass
from pathlib import Path

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
