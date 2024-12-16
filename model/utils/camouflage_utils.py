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
    Extracts a 16:9 aspect ratio region from the image based on the shape defined by the mask.
    """
    # Find contours from the mask
    grey_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    grey_mask = (grey_mask * 255).astype(np.uint8)  # Convert to 8-bit format

    contours, _ = cv2.findContours(
        grey_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box around the contours
    x, y, w, h = cv2.boundingRect(contours[0])

    # Calculate target dimensions for 16:9 aspect ratio
    if w / h > 16 / 9:
        new_w = w
        new_h = int(w * 9 / 16)
    else:
        new_h = h
        new_w = int(h * 16 / 9)

    # Center the 16:9 box around the original bounding box
    x_center = x + w // 2
    y_center = y + h // 2
    x_start = max(0, x_center - new_w // 2)
    y_start = max(0, y_center - new_h // 2)

    # Ensure the cropping region is within the image boundaries
    x_end = min(x_start + new_w, image.shape[1])
    y_end = min(y_start + new_h, image.shape[0])

    # Crop the image to the 16:9 area
    cropped_image = image[y_start:y_end, x_start:x_end]

    # Resize to maintain 16:9 aspect ratio, if necessary
    return resize_to_16_9(cropped_image)
