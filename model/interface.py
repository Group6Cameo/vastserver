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


def generate_camouflage(background_image, mask_path):
    """Generate camouflage using the LaMa model"""
    os.makedirs('./surroundings_data', exist_ok=True)
    os.makedirs('./output', exist_ok=True)

    # Prepare background image
    img_suffix = os.path.splitext(background_image)[1].lower()
    if img_suffix not in ['.png', '.jpg', '.jpeg']:
        raise ValueError(
            f'Unsupported image format: {img_suffix}. Use [.png, .jpeg, .jpg]')

    # Run LaMa prediction directly
    cmd = [
        'python3',
        '/app/model/lama/bin/predict.py',
        # 'refine=True',
        'model.path=/app/model/big-lama',
        f'indir=/app/surroundings_data',
        'outdir=/app/output',
        f'dataset.img_suffix={img_suffix}'
    ]

    subprocess.run(cmd, check=True)

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
