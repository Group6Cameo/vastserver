from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import logging
import os
import cv2
from model.interface import generate_camouflage
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


@app.get("/generate-camouflage")
async def generate_camouflage_pattern(
    image: UploadFile = File(...),
):
    try:
        # Save uploaded files
        print("hier")
        image_path = os.path.abspath(f"surroundings_data/{image.filename}")
        base_name = os.path.splitext(image.filename)[0]
        mask_path = os.path.abspath(
            f"surroundings_data/{base_name}_mask.png")
        annotated_path = os.path.abspath(
            f"surroundings_data/{image.filename}_annotated.png")

        # Save uploaded file
        with open(image_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)

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
        output_path = "output/result.png"
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
