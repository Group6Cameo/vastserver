from ultralytics import YOLO
import cv2
import numpy as np
import sys
import argparse
import os


def predict_image(image_path, annotated_path):
    try:
        # Load trained model
        model = YOLO('model/YOLO/weights/best.pt')
        mask_path = os.path.splitext(image_path)[0] + "_mask.png"

        # Read original image
        original_img = cv2.imread(image_path)
        height, width = original_img.shape[:2]

        # Create blank mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Make predictions
        results = model.predict(
            source=image_path,
            conf=0.5,
            show=False
        )

        # Create annotated image (copy of original)
        annotated_img = original_img.copy()

        # Fill mask and draw boxes on annotated image
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Fill mask
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

                # Draw on annotated image
                cv2.rectangle(annotated_img, (x1, y1),
                              (x2, y2), (0, 255, 0), 2)

        # Save both images
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(annotated_path, annotated_img)
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('mask_path', help='Path to output mask')
    parser.add_argument(
        'annotated_path', help='Path to output annotated image')
    args = parser.parse_args()

    predict_image(args.image_path, args.mask_path, args.annotated_path)
