"""
Interface module for AI camouflage generation using LaMa inpainting.

This module provides the core functionality for generating camouflage patterns
using the LaMa (Large Mask) inpainting model. It handles model initialization,
inference, and result processing.

The module consists of three main components:
1. SuperResolutionWrapper - Handles image upscaling using RealESRGAN
2. LamaModel - Singleton class managing the LaMa inpainting model
3. generate_camouflage - Main function orchestrating the camouflage generation process

Technical Details:
- Supports both CPU and CUDA execution with automatic device selection
- Uses RealESRGAN for high-quality 4x upscaling
- Includes fallback upscaling methods for error handling
- Processes images to 2560x1440 resolution
- Implements 16:9 aspect ratio extraction

Dependencies:
- torch: Deep learning framework
- cv2: Image processing
- RealESRGAN: Super-resolution
- LaMa: Inpainting model
"""

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
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


class SuperResolutionWrapper:
    """
    Wrapper class for RealESRGAN super-resolution model implementation.

    This class manages the loading and execution of the RealESRGAN model for
    high-quality image upscaling. It implements lazy loading to optimize
    memory usage and provides a simple interface for image enhancement.

    Attributes:
        upsampler (RealESRGANer): Instance of RealESRGAN upsampling model
        device (torch.device): Device for model execution (CPU/CUDA)

    Methods:
        load_model(): Initializes the RealESRGAN model with optimal settings
        enhance(img): Performs 4x upscaling on input image
    """

    def __init__(self):
        self.upsampler = None
        self.device = None

    def load_model(self):
        if self.upsampler is not None:
            return

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        model_path = '/app/models/RealESRGAN_x4plus.pth'

        # Initialize the upsampler
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=400,       # Tile size for GPU memory optimization
            tile_pad=10,    # Padding between tiles
            pre_pad=0,      # Pre-padding size
            half=False,     # Use FP32 precision
            device=self.device
        )

    def enhance(self, img):
        if self.upsampler is None:
            self.load_model()

        # Convert BGR to RGB and normalize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform super-resolution
        output, _ = self.upsampler.enhance(img_rgb, outscale=4)

        # Convert back to BGR and return
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


sr_wrapper = SuperResolutionWrapper()


def smart_upscale(image, target_size):
    """
    Fallback upscaling method when RealESRGAN fails.

    Implements a combination of detail enhancement, Lanczos interpolation,
    and sharpening to provide decent quality upscaling results.

    Args:
        image (np.ndarray): Input image in BGR format
        target_size (tuple): Desired output resolution (width, height)

    Returns:
        np.ndarray: Upscaled image in BGR format
    """
    enhanced = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    result = cv2.resize(enhanced, target_size,
                        interpolation=cv2.INTER_LANCZOS4)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(result, -1, kernel)


def generate_camouflage(background_image, mask_path):
    """
    Generate camouflage patterns using the LaMa inpainting model.

    This function orchestrates the entire camouflage generation process:
    1. Sets up necessary directories
    2. Loads and configures the LaMa model
    3. Processes the input image and mask
    4. Performs inpainting
    5. Extracts the relevant region
    6. Upscales the result to 2560x1440

    Args:
        background_image (str): Path to the background image
        mask_path (str): Path to the mask image defining camouflage regions

    Returns:
        np.ndarray: Generated camouflage image in BGR format at 2560x1440 resolution

    Raises:
        ValueError: If mask file cannot be read
        Exception: If super-resolution fails (falls back to basic upscaling)
    """
    os.makedirs('./surroundings_data', exist_ok=True)
    os.makedirs('./output', exist_ok=True)

    model, device = lama_model.get_model()

    predict_config = OmegaConf.create({
        'indir': '/app/surroundings_data',
        'outdir': '/app/output',
        'model': {'path': '/app/model/big-lama', 'checkpoint': 'best.ckpt'},
        'dataset': {
            'kind': 'default',
            'img_suffix': os.path.splitext(background_image)[1].lower(),
            'pad_out_to_modulo': 8
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'out_key': 'inpainted',
        'refine': True,
        'refiner': {
            'gpu_ids': '0, 1',
            'modulo': 8,
            'n_iters': 15,
            'lr': 0.002,
            'min_side': 512,
            'max_scales': 3,
            'px_budget': 1800000
        },
        'out_ext': '.png'
    })

    process_predict(predict_config, preloaded_model=model,
                    preloaded_device=device)

    output_filename = f"output/{os.path.splitext(os.path.basename(background_image))[0]}_mask.png"
    result = cv2.imread(output_filename)

    # Use cv2.imread instead of plt.imread for mask, and convert to grayscale if needed
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask from {mask_path}")

    # Enhanced mask processing
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = (mask / 255.0) > 0.5

    extracted_inpaint = np.zeros_like(result)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    extracted_inpaint[mask_3d] = result[mask_3d]

    mask_uint8 = (mask * 255).astype(np.uint8)
    final_result = extract_16_9_region(extracted_inpaint, mask_uint8)

    target_size = (2560, 1440)

    try:
        # Step 1: Apply Real-ESRGAN 4x upscale
        sr_enhanced = sr_wrapper.enhance(final_result)

        # Step 2: Resize to exact target dimensions
        upscaled_result = cv2.resize(
            sr_enhanced,
            target_size,
            interpolation=cv2.INTER_CUBIC
        )

        # Step 3: Post-processing
        upscaled_result = cv2.detailEnhance(
            upscaled_result, sigma_s=5, sigma_r=0.15)
        upscaled_result = cv2.medianBlur(upscaled_result, 3)

    except Exception as e:
        print(f"Super-resolution error: {e}, using fallback upscaling")
        upscaled_result = smart_upscale(final_result, target_size)

    cv2.imwrite('/app/output/testsave.png', upscaled_result)
    return upscaled_result


class LamaModel:
    """
    Singleton class managing the LaMa inpainting model.

    This class handles the initialization, loading, and access to the LaMa model.
    It implements lazy loading to optimize memory usage and provides thread-safe
    access to the model instance.

    Attributes:
        model: The loaded LaMa model instance
        device (torch.device): Device for model execution (CPU/CUDA)

    Methods:
        load(): Initializes the model with appropriate configuration
        get_model(): Returns the loaded model and device (loads if necessary)

    Technical Details:
        - Automatically selects between CUDA and CPU execution
        - Configures model for inference-only mode
        - Disables visualizations for optimal performance
        - Freezes model weights after loading
    """

    def __init__(self):
        self.model = None
        self.device = None

    def load(self):
        """
        Load the LaMa model and initialize it on the appropriate device.

        This method handles:
        - Device selection (CUDA/CPU)
        - Configuration loading from configs/prediction/default.yaml
        - Model checkpoint loading
        - Model preparation (freezing weights, moving to device)

        The model is configured for inference-only mode with visualizations disabled
        for optimal performance.

        Raises:
            RuntimeError: If model checkpoint or config cannot be loaded
        """
        if self.model is not None:
            return

        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)

        train_config_path = '/app/model/big-lama/config.yaml'
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = '/app/model/big-lama/models/best.ckpt'
        self.model = load_checkpoint(
            train_config, checkpoint_path, strict=False, map_location=device_name)
        self.model.freeze()
        self.model.to(self.device)

    def get_model(self):
        if self.model is None:
            self.load()
        return self.model, self.device


lama_model = LamaModel()
