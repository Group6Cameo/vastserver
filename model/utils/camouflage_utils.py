"""
Utility functions for camouflage pattern processing.

This module provides helper functions for processing and formatting
camouflage patterns, including region extraction, aspect ratio
adjustment, and image manipulation utilities.

The utilities focus on maintaining proper aspect ratios and
extracting relevant regions from processed images while preserving
quality and maintaining consistent output formats.
"""

import cv2
import numpy as np


def extract_inpainted_region(mask, inpainted_image):
    """
    Extracts the inpainted region from the full image based on the mask.
    """
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    extracted_inpaint = np.zeros_like(inpainted_image)
    extracted_inpaint[mask_3d] = inpainted_image[mask_3d]
    return extracted_inpaint


def resize_to_16_9(image):
    """
    Resizes the image to a 16:9 aspect ratio, maintaining the original image size.
    """
    # Calculate target dimensions while maintaining 16:9 ratio
    height, width = image.shape[:2]
    target_width = width
    target_height = int(width * 9/16)

    # Create new image with target dimensions
    resized = cv2.resize(image, (target_width, target_height),
                         interpolation=cv2.INTER_AREA)
    return resized


def extract_16_9_region(image, mask):
    """
    Extracts a 16:9 aspect ratio region from within the masked area.
    """
    # Check if mask is already grayscale (single channel)
    if len(mask.shape) == 3:
        grey_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        grey_mask = mask

    # Ensure mask is in uint8 format
    grey_mask = (
        grey_mask * 255).astype(np.uint8) if grey_mask.dtype != np.uint8 else grey_mask

    contours, _ = cv2.findContours(
        grey_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box around the contours
    x, y, w, h = cv2.boundingRect(contours[0])

    # Zoom in by reducing the bounding box size by 30%
    padding_x = int(w * 0.3)
    padding_y = int(h * 0.3)
    x += padding_x // 2
    y += padding_y // 2
    w -= padding_x
    h -= padding_y

    # Calculate dimensions for 16:9 region that fits inside the rectangle
    if w/h > 16/9:  # If wider than 16:9
        new_w = int(h * 16/9)  # Use height to determine width
        new_h = h
    else:  # If taller than 16:9
        new_w = w
        new_h = int(w * 9/16)  # Use width to determine height

    # Calculate starting points to center the 16:9 region
    x_start = x + (w - new_w) // 2
    y_start = y + (h - new_h) // 2

    # Crop the image to the 16:9 area
    cropped_image = image[y_start:y_start + new_h, x_start:x_start + new_w]

    return cropped_image  # No need to resize as it's already 16:9
