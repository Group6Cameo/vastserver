from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import logging
import os
import cv2
import numpy as np
from model.interface import generate_camouflage, lama_model
from model.YOLO.detect import predict_image

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
os.makedirs("surroundings_data", exist_ok=True)
os.makedirs("output", exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize the model when the app starts"""
    logger.info("Loading LaMa model...")
    lama_model.load()
    logger.info("LaMa model loaded successfully")


@app.get("/generate-camouflage")
async def generate_camouflage_pattern(
    image: UploadFile = File(...),
):
    try:
        # Save uploaded files
        image_path = os.path.abspath(f"surroundings_data/{image.filename}")

        # Save uploaded file and read content
        content = await image.read()
        with open(image_path, "wb") as buffer:
            buffer.write(content)

        # Check file size and resize if necessary (targeting ~800KB)
        if os.path.getsize(image_path) > 800000:  # 800KB in bytes
            img = cv2.imread(image_path)
            h, w = img.shape[:2]

            # Calculate new size while maintaining aspect ratio
            target_pixels = 1000000  # Target ~1M pixels
            scale = np.sqrt(target_pixels / (h * w))
            new_h = int(h * scale)
            new_w = int(w * scale)

            # Resize image
            resized = cv2.resize(img, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
            cv2.imwrite(image_path, resized)

        base_name = os.path.splitext(image.filename)[0]
        mask_path = os.path.abspath(
            f"surroundings_data/{base_name}_mask.png")
        annotated_path = os.path.abspath(
            f"surroundings_data/{image.filename}_annotated.png")

        # Run YOLO detection directly
        logger.info("Running YOLO detection...")
        result = predict_image(image_path, annotated_path)

        if not result:
            logger.error("YOLO detection failed")
            raise HTTPException(
                status_code=500, detail="YOLO detection failed")

        # Run LaMa inpainting
        logger.info("Running LaMa inpainting...")
        result = generate_camouflage(image_path, mask_path)

        # Clean up
        # for path in [image_path, mask_path, annotated_path]:
        #     if os.path.exists(path):
        #         os.remove(path)

        # Convert numpy array to image bytes
        success, encoded_img = cv2.imencode('.png', result)
        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to encode image")

        # Alternatively, if you want to save the file and return it:
        output_path = "/app/output/result.png"
        cv2.imwrite(output_path, result)
        return FileResponse(output_path, media_type="image/png")

    except Exception as e:
        logger.error(f"Error in camouflage generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def process_input(input_data):
    # Define how to convert the file input to the model's required input format
    pass


def post_process_prediction(prediction):
    # Define how to convert the model's output to the desired response format
    pass
