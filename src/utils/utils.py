import os
import random
import json

import torch
import numpy as np

def make_directories(exp_name: str):

    subfolders = ['results/', 'saliency_maps/original/', 'saliency_maps/saliency/']

    for subfolder in subfolders:
        os.makedirs(os.path.join(f"experiments/{exp_name}/", subfolder), exist_ok=True)

def set_random_seed(seed: int):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

def append_json(new_data, path):
    try:
        with open(path, "r") as file:
            file_data = json.load(file)
    except:
        file_data = []
        print("Generating new experiment file")

    with open(path, "w+") as file:
        file_data.append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent=4)