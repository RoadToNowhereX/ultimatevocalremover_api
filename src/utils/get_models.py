import os
import glob 
import urllib.request 
from typing import List, Optional
from pathlib import Path
import json 

uvr_path = Path(__file__).parent.parent

def download_model(model_name:str, model_arch:str, model_path:List[str]=None, save_path:str=None, logger=None)->str:
    """Download model from Hugging Face model hub
    
    Args:
        model_name (str): model name. 
        model_path (list[str]): model pathS to download the model from. Defaults to None (loads paths from uvr/models_dir/models.json file)
        model_arch (str): model architecture. A path in ../models_dir/{model_arch}/weights/{model_name}
                            If path is not found it will be created. And if the model is already downloaded it will not be downloaded again.
        logger (logging.Logger, optional): logger. Defaults to None.
    
    Returns:
        str: path to the downloaded model
    """
    if model_path is None:
        if logger:
            logger.error(f"Model path is not provided for {model_name} auto loading from models.json file")
        models_json_path = os.path.join(uvr_path, "models_dir", "models.json")
        models = json.load(open(models_json_path, "r"))
        model_path = models[model_arch][model_name]["model_path"]

    if save_path is None:
        save_path = os.path.join(uvr_path, "models_dir", model_arch, "weights", model_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    files = [path.split("/")[-1] for path in model_path]
    if model_exists_in_package(model_name=model_name, model_arch=model_arch, files=files):
        if logger:
            logger.info(f"Model {model_name} is already exists in {save_path}")
        return save_path
    
    try:
        # os.system(f"wget {model_path} -P {local_model_path}")
        for file_name, path in zip(files, model_path):
            local_file_path = os.path.join(save_path, file_name)
            urllib.request.urlretrieve(path, local_file_path)
            if logger:
                logger.info(f"Downloaded {model_name} from {model_path}")
            
        
        return save_path
    
    except Exception as e:
        if logger:
            logger.error(f"Failed to download {model_name} from {model_path} with error {e}")
    
    return None


def model_exists_in_package(model_name:str, model_arch:str, save_path:str=None, files:List=None)->bool:
    """Check if the model exists in ../models_dir/{model_arch}/weights/{model_name}
    
    Args:
        model_name (str): model name.
        model_arch (str): model architecture.
        files (list[str], optional): list of files to check if they exist. Defaults to None. If not provided it will check if the model directory exists.
    
    Returns:
        bool: True if the model exists, False otherwise
    """
    # remove extension from the model name
    if len(model_name.split('.')) > 1:
        model_name = model_name.split('.')[0]

    if save_path is None:
        save_path = os.path.join(uvr_path, "models_dir", model_arch, "weights", model_name)

    if files is not None:
        for file in files:
            local_model_path = os.path.join(save_path, file)
            if not os.path.isfile(local_model_path):
                return False

    if os.path.exists(save_path):
        return True
    return False


def get_model_path(model_name: str, model_dir: str) -> Optional[str]:
    """
    Get model path. Handles cases where the model is inside a subdirectory
    with the same name.
    """
    if not os.path.exists(model_dir):
        return None

    # 检查 model_name 是否已经包含扩展名
    base_name, extension = os.path.splitext(model_name)
    if extension:
        path = os.path.join(model_dir, model_name)
        if os.path.exists(path) and os.path.isfile(path): # 确保是文件
            return path
        return None

    # 如果 model_name 没有扩展名，则开始搜索
    
    # 优先检查是否存在一个与 model_name 同名的子目录
    potential_model_subdir = os.path.join(model_dir, model_name)
    if os.path.isdir(potential_model_subdir):
        # 如果是目录，则在该目录内寻找模型文件（例如 .onnx, .pth 等）
        for file in os.listdir(potential_model_subdir):
            # 你可以根据需要寻找特定的扩展名，这里以 .onnx 为例
            if file.endswith(('.onnx', '.pth', '.ckpt')):
                return os.path.join(potential_model_subdir, file)
        # 如果目录内没有找到模型文件，可以返回目录路径或None，取决于后续逻辑
        # 在这个报错场景下，我们应该继续寻找，而不是返回目录
        
    # 如果没有同名子目录，或者子目录里没找到，就在当前目录继续寻找文件
    for file in os.listdir(model_dir):
        if os.path.isfile(os.path.join(model_dir, file)): # 只检查文件，忽略子目录
            file_basename, _ = os.path.splitext(file)
            if file_basename == model_name:
                return os.path.join(model_dir, file)

    return None

"""
Example of the model json file:
models_json = {

"demucs":{
    "name1":{
        "model_path":"https://abc/bcd/model.pt",
        "other_metadata":1,
    },
    }
}
"""

def download_all_models(models_json:dict=None, save_path:str=None, logger=None)->dict:
    """Download all models from the models_json
    
    Args:
        models_json (dict): dictionary of models to download. Defaults to None (loads paths from uvr/models_dir/models.json file)
        logger (logging.Logger, optional): logger. Defaults to None.

    Returns:
        dict: dictionary of downloaded models. with the same structure as the input models_json.
                architectures -> model_name -> model_path. Also the model_path will be the local path to the downloaded model.
                If the model is already downloaded it will not be downloaded again. And if the model failed to download it will be None.
    """
    paths = {}
    if models_json is None:
        if logger:
            logger.error(f"Model path is not provided for {model_name} auto loading from models.json file")
        models_json_path = os.path.join(uvr_path, "models_dir", "models.json")
        models_json = json.load(open(models_json_path, "r"))
        
    for model_arch, models in models_json.items():
        paths[model_arch] = {}
        for model_name, model_data in models.items():
            model_path = model_data["model_path"]
            model_path = download_model(model_name=model_name, model_path=model_path, model_arch=model_arch, logger=logger, save_path=save_path)
            paths[model_arch][model_name] = model_path

    return paths

