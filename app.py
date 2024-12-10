from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import torch
from model.interface import generate_camouflage
import os
import subprocess

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model when the application starts
model = generate_camouflage("path_to_model_weights.pth")
model.eval()  # Ensure the model is in evaluation mode


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
