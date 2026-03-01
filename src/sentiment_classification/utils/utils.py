import os
import json
import psutil
import torch

def get_hardware_info():
    """
    Retrieves system hardware information including CPU, RAM, and GPU.
    """
    info = {
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "max_frequency_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "current_frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "cpu_usage_percent": psutil.cpu_percent(interval=1)
        },
        "ram": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
            "usage_percent": psutil.virtual_memory().percent
        },
        "gpu": []
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
            }
            info["gpu"].append(gpu_info)
    else:
        info["gpu"] = "No GPU Available"

    return info

def save_json(data: dict, filepath: str):
    """
    Saves a dictionary as a JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
