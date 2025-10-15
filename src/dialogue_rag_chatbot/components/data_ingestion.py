import os
import urllib.request as request
import py7zr
from dialogue_rag_chatbot.logging import logger
from dialogue_rag_chatbot.utils.common import get_size
from dialogue_rag_chatbot.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with py7zr.SevenZipFile(self.config.local_data_file, mode="r") as z:
            z.extractall(unzip_path)