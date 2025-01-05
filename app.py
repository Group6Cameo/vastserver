from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import logging
import os
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
    # image: UploadFile = File(...),
    # background: UploadFile = File(...)
):
    try:
        # Save uploaded files
        # image_path = os.path.abspath(f"surroundings_data/{image.filename}")
        # background_path = os.path.abspath(
        #     f"surroundings_data/background_{background.filename}")
        # mask_path = os.path.abspath(
        #     f"surroundings_data/{image.filename}_mask.png")
        # annotated_path = os.path.abspath(
        #     f"surroundings_data/{image.filename}_annotated.png")

        # # Save uploaded files
        # for file, path in [(image, image_path), (background, background_path)]:
        #     with open(path, "wb") as buffer:
        #         content = await file.read()
        #         buffer.write(content)
        image_path = "surroundings_data/originaltest.jpg"
        mask_path = image_path.replace(".jpg", "_mask.png")
        annotated_path = "surroundings_data/annotatedtest.jpg"

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
        for path in [image_path, mask_path, annotated_path]:
            if os.path.exists(path):
                os.remove(path)

        return {"result": result.tolist()}

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
