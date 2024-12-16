from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import logging
import os
import subprocess
from model.interface import generate_camouflage
from pathlib import Path

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


@app.post("/generate-camouflage")
async def generate_camouflage_pattern(
    image: UploadFile = File(...),
    background: UploadFile = File(...)
):
    try:
        # Save uploaded files
        image_path = f"surroundings_data/{image.filename}"
        background_path = f"surroundings_data/background_{background.filename}"
        mask_path = f"surroundings_data/{image.filename}_mask.png"

        # Save uploaded files
        for file, path in [(image, image_path), (background, background_path)]:
            with open(path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

        # Run YOLO detection
        logger.info("Running YOLO detection...")
        yolo_cmd = [
            "docker", "run", "--gpus", "all", "--rm",
            "-v", f"{os.getcwd()}/surroundings_data:/app/data",
            "yolo-model",
            "python3", "detect.py",
            f"/app/data/{image.filename}",
            f"/app/data/{image.filename}_mask.png"
        ]
        subprocess.run(yolo_cmd, check=True)

        # Run LaMa inpainting
        logger.info("Running LaMa inpainting...")
        result = generate_camouflage(background_path, mask_path)

        # Clean up
        for path in [image_path, background_path, mask_path]:
            if os.path.exists(path):
                os.remove(path)

        return {"result": result.tolist()}

    except Exception as e:
        logger.error(f"Error in camouflage generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        image_path = f"surroundings_data/{file.filename}"
        mask_path = f"surroundings_data/{file.filename}_mask.png"

        with open(image_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Run YOLO detection to create mask
        cmd = [
            "docker", "run",
            "--gpus", "all",
            "--rm",
            "-v", f"{os.getcwd()}/surroundings_data:/app/data",
            "yolo-model",
            "python3", "detect.py",
            f"/app/data/{file.filename}",
            f"/app/data/{file.filename}_mask.png"
        ]
        subprocess.run(cmd, check=True)

        # Run LaMa inpainting
        result = generate_camouflage(image_path, mask_path)

        return {"result": result.tolist()}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def process_input(input_data):
    # Define how to convert the file input to the model's required input format
    pass


def post_process_prediction(prediction):
    # Define how to convert the model's output to the desired response format
    pass
