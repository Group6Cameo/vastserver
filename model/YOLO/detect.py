from ultralytics import YOLO
import cv2
import numpy as np
import sys


def predict_image():
    try:
        # Load your trained model
        model = YOLO('runs/detect/cameo_model7/weights/best.pt')

        # Get image path
        image_path = '/Users/montijn/Downloads/ResizedImages/IMG_1312.jpg'

        # Read original image to get dimensions
        original_img = cv2.imread(image_path)
        height, width = original_img.shape[:2]

        # Create blank mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Make predictions
        results = model.predict(
            source=image_path,
            conf=0.05,  # Lower confidence threshold to ensure detection
            show=True   # Show the original detection
        )

        # For each detection, fill the mask
        for result in results:
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Convert to integers
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                # Fill the detected area with white (255)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Show the mask
        cv2.imshow('Mask', mask)

        # Save the mask
        cv2.imwrite('detection_mask.png', mask)

        # Wait until a key is pressed or Ctrl+C
        cv2.waitKey(0)

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cv2.destroyAllWindows()
        sys.exit(0)


if __name__ == '__main__':
    predict_image()
