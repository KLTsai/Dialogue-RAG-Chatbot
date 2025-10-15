import os
from box.exceptions import BoxValueError
import yaml
from src.dialogue_rag_chatbot.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import List, Dict, Any, Tuple
from src.dialogue_rag_chatbot.entity import (
    IsREL,
    IsSUP
)


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def calculate_score(candidate: List[Dict[str, Any]]) -> int:
    """ calculate score for IsUse

    Args :
        candidate (List[Dict[str, Any]]): candidate answer list

    Returns:
        int: socre after weighted sum

    """

    score = 0

    # IsREL 權重
    if candidate['is_relevant'] == IsREL.RELEVANT:
        score += 10

    # IsSUP 權重
    support_scores = {
        IsSUP.FULLY_SUPPORTED: 10,
        IsSUP.PARTIALLY_SUPPORTED: 5,
        IsSUP.NO_SUPPORT: 0
    }
    score += support_scores.get(candidate['support_level'], 0)

    # IsUSE 權重
    score += candidate['usefulness_score'].value * 2

    return score