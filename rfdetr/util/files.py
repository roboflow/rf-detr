import requests
import yaml
import json
import os
from tqdm import tqdm
from pathlib import Path

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers['content-length'])
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def load_yaml(yaml_path: str | Path) -> dict:
    """
    Loads a yaml file and returns its content as a dictionary.
    
    Args:
        yaml_path: Path to the YAML file
        
    Returns:
        dict: Parsed YAML content
        
    Raises:
        TypeError: If path is not a string or Path type object
        ValueError: If path is empty
        FileNotFoundError: If file doesn't exist
        PermissionError: If file is not readable
        yaml.YAMLError: If YAML parsing fails
    """
    if not isinstance(yaml_path, str | Path):
        raise TypeError("Path must be a string or Path type object")
    if not yaml_path.strip():
        raise ValueError("Path cannot be empty")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Yaml file not found: {yaml_path}")
    if not os.access(yaml_path, os.R_OK):
        raise PermissionError(f"No read permissions for file: {yaml_path}")

    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML file: {yaml_path}. Error: {str(e)}")


def load_json(json_path: str | Path) -> dict:
    """
    Loads a json file and returns its content as a dictionary.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        dict: Parsed JSON content
        
    Raises:
        TypeError: If path is not a string or Path type object
        ValueError: If path is empty
        FileNotFoundError: If file doesn't exist
        PermissionError: If file is not readable
        json.JSONDecodeError: If JSON parsing fails
    """
    if not isinstance(json_path, str | Path):
        raise TypeError("Path must be a string or Path type object")
    if not json_path.strip():
        raise ValueError("Path cannot be empty")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Json file not found: {json_path}")
    if not os.access(json_path, os.R_OK):
        raise PermissionError(f"No read permissions for file: {json_path}")

    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Failed to parse JSON file: {json_path}. Error: {str(e)}", e.doc, e.pos)

def load_txt(txt_path: str | Path) -> str:
    """
    Loads a txt file and returns its content as a string.
    """
    if not isinstance(txt_path, str | Path):
        raise TypeError("Path must be a string or Path type object")
    if not txt_path.strip():
        raise ValueError("Path cannot be empty")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Json file not found: {txt_path}")
    if not os.access(txt_path, os.R_OK):
        raise PermissionError(f"No read permissions for file: {txt_path}")
    
    with open(txt_path, 'r') as f:
        return f.read()


def load_txt_lines(txt_path: str | Path) -> list[str]:
    """
    Loads a txt file and returns its content as a list of strings.
    """
    return load_txt(txt_path).splitlines()
