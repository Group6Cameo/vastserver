from shutil import copyfile
import shutil
import subprocess
import sys
from pathlib import Path
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from model.utils.camouflage_utils import extract_16_9_region
from saicinpainting.training.trainers import load_checkpoint
from omegaconf import OmegaConf
import yaml
import torch
from model.lama.bin.predict import process_predict


def generate_camouflage(background_image, mask_path):
    """Generate camouflage using the LaMa model"""
    os.makedirs('./surroundings_data', exist_ok=True)
    os.makedirs('./output', exist_ok=True)

    # Get the preloaded model and device
    model, device = lama_model.get_model()

    # Create a complete predict config
    predict_config = OmegaConf.create({
        'indir': '/app/surroundings_data',
        'outdir': '/app/output',
        'model': {
            'path': '/app/model/big-lama',
            'checkpoint': 'best.ckpt'
        },
        'dataset': {
            'kind': 'default',
            'img_suffix': os.path.splitext(background_image)[1].lower(),
            'pad_out_to_modulo': 8
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'out_key': 'inpainted',
        'refine': False,
        'out_ext': '.png'
    })

    # Run prediction using the preloaded model
    process_predict(predict_config, preloaded_model=model,
                    preloaded_device=device)

    # Process results
    output_filename = f"output/{os.path.splitext(os.path.basename(background_image))[0]}_mask.png"
    result = cv2.imread(output_filename)
    print("results ready")

    # Use cv2.imread instead of plt.imread for mask, and convert to grayscale if needed
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask from {mask_path}")
    mask = mask / 255.0  # Normalize to [0,1]
    print("mask")

    extracted_inpaint = np.zeros_like(result)
    print("extracted")
    # Convert mask to boolean array
    mask = mask > 0.5  # Convert float values to boolean
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    print("3d")
    extracted_inpaint[mask_3d] = result[mask_3d]
    print(np.unique(extracted_inpaint))
    print("returns")

    mask_uint8 = (mask * 255).astype(np.uint8)
    final_result = extract_16_9_region(extracted_inpaint, mask_uint8)

    # Resize to 2560x1440 (16:9)
    target_width = 2560
    target_height = 1440  # 16:9 ratio

    upscaled_result = cv2.resize(
        final_result, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite('/app/output/testsave.png', upscaled_result)

    return upscaled_result


class LamaModel:
    def __init__(self):
        self.model = None
        self.device = None

    def load(self):
        if self.model is not None:
            return

        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)

        # Load configuration
        train_config_path = '/app/model/big-lama/config.yaml'
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        # Load model
        checkpoint_path = '/app/model/big-lama/models/best.ckpt'
        self.model = load_checkpoint(
            train_config, checkpoint_path, strict=False, map_location=device_name)
        self.model.freeze()
        self.model.to(self.device)

    def get_model(self):
        if self.model is None:
            self.load()
        return self.model, self.device


# Create a global instance
lama_model = LamaModel()
