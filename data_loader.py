import os
import json
import glob
from typing import List, Dict

def load_local_images(folder_path: str, extensions: tuple = ('*.jpg', '*.png', '*.jpeg')) -> List[Dict]:
    """
    Load test images from a local folder.
    Returns a list of dicts: [{'image_path': path, 'prompt': ''}, ...]
    """
    image_paths = []
    for ext in extensions:
        # Support recursive globbing if needed, here just flat directory
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    data = []
    for path in image_paths:
        data.append({
            'image_path': path,
            'prompt': ''  # Local images without associated prompts
        })
    return data

def load_hf_jsonl(jsonl_path: str) -> List[Dict]:
    """
    Load from a JSONL metadata file (e.g. from HuggingFace dataset).
    Expected format per line: {"image_path": "...", "prompt": "..."}
    """
    data = []
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        
    return data

def get_dataloader(source_type: str, path: str) -> List[Dict]:
    """
    Unified entry to load data. 
    source_type: 'local' or 'jsonl'.
    """
    if source_type == 'local':
        return load_local_images(path)
    elif source_type == 'jsonl':
        return load_hf_jsonl(path)
    else:
        raise ValueError("Unsupported source_type. Use 'local' or 'jsonl'.")

if __name__ == "__main__":
    # Example usage:
    # print(get_dataloader('local', './test_images'))
    pass
