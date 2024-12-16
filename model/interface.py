from shutil import copyfile
import shutil
import subprocess
import sys
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import base64


def generate_camouflage(background_image, mask_path):
    """Generate camouflage using the LaMa model"""
    os.makedirs('./surroundings_data', exist_ok=True)
    os.makedirs('./output', exist_ok=True)

    # Prepare background image
    img_suffix = os.path.splitext(background_image)[1].lower()
    if img_suffix not in ['.png', '.jpg', '.jpeg']:
        raise ValueError(
            f'Unsupported image format: {img_suffix}. Use [.png, .jpeg, .jpg]')

    # Run LaMa prediction
    cmd = [
        'docker', 'run',
        '--gpus', 'all',
        '--rm',
        '-v', f'{os.getcwd()}:/data',
        'lama-model',
        'python3', '/app/bin/predict.py',
        'model.path=/app/big-lama',
        f'indir=/data/surroundings_data',
        'outdir=/data/output',
        f'dataset.img_suffix={img_suffix}'
    ]

    subprocess.run(cmd, check=True)

    # Process results
    output_filename = f"output/{os.path.splitext(os.path.basename(background_image))[0]}_inpainted.png"
    result = plt.imread(output_filename)

    # Apply mask to get final result
    mask = plt.imread(mask_path)
    extracted_inpaint = np.zeros_like(result)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    extracted_inpaint[mask_3d] = result[mask_3d]

    return extracted_inpaint
