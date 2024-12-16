from ultralytics import YOLO
import cv2
import numpy as np
import sys
import argparse


def predict_image(image_path, mask_path):
    try:
        # Load trained model
        model = YOLO('model/YOLO/weights/best.pt')

        # Read original image
        original_img = cv2.imread(image_path)
        height, width = original_img.shape[:2]

        # Create blank mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Make predictions
        results = model.predict(
            source=image_path,
            conf=0.05,
            show=False
        )

        # Fill mask with detections
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Save the mask
        cv2.imwrite(mask_path, mask)
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('mask_path', help='Path to output mask')
    args = parser.parse_args()

    predict_image(args.image_path, args.mask_path)
