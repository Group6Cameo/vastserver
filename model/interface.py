from shutil import copyfile
import shutil
import subprocess
import sys
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import base64


def generate_camouflage(image_name, mask_name):
    """Generate camouflage using the LaMa model"""
    # Create surroundings_data directory if it doesn't exist
    os.makedirs('./surroundings_data', exist_ok=True)

    # Copy and prepare input image
    copyfile(image_name, f'./surroundings_data/{image_name}')
    os.remove(image_name)
    image_name = f'./surroundings_data/{image_name}'

    # Get image suffix
    img_suffix = os.path.splitext(image_name)[1].lower()
    if img_suffix not in ['.png', '.jpg', '.jpeg']:
        raise ValueError(
            f'Unsupported image format: {img_suffix}. Use [.png, .jpeg, .jpg]')

    # Run prediction using Docker
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

    # Process and display results
    output_filename = f"/content/output/{os.path.splitext(os.path.basename(image_name))[0]}_mask.png"
    endresult = plt.imread(output_filename)

    plt.rcParams['figure.dpi'] = 200
    plt.imshow(endresult)
    plt.axis('off')
    plt.title('endresult')
    plt.show()

    # Process mask and show inpainting result
    mask = plt.imread(mask_name)
    extracted_inpaint = np.zeros_like(endresult)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    extracted_inpaint[mask_3d] = endresult[mask_3d]

    plt.imshow(extracted_inpaint)
    plt.axis('off')
    plt.title('inpainting result')
    plt.show()

    return extracted_inpaint
