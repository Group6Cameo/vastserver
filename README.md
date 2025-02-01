# AI Camouflage Generation Server

This server processes images to generate camouflage patterns using a combination of YOLO object detection and LaMa image inpainting.

## Pipeline Overview

1. **Object Detection (YOLO)**
   - Detects objects in the input image
   - Generates masks for detected objects

2. **Image Inpainting (LaMa)**
   - Uses the LaMa (Large Mask) inpainting model
   - Fills masked areas with contextually appropriate patterns
   - Pre-loaded model for efficient processing

3. **Super-Resolution (RealESRGAN)**
   - Upscales the inpainted regions using RealESRGAN 4x model
   - Enhances detail and quality of the generated patterns
   - Falls back to traditional upscaling methods if needed

4. **Post-processing**
   - Extracts the inpainted regions
   - Resizes to 16:9 aspect ratio (2560x1440)
   - Outputs the final camouflage pattern



## Setup

1. On a service like [vast.ai](vast.ai), use the image `montijnb/cameosmall:latest`. This will install everything. To run the application you dont have to edit anything, but if you want to do development, you can access the created machine through SSH, or open a jupyter notebook from vast.ai itself.

## Environment Variables

- `TORCH_HOME`: Set to project root
- `PYTHONPATH`: Set to project root
- `CUDA_VISIBLE_DEVICES`: GPU selection (optional)

## Model Configuration

The LaMa model is configured for:
- Input padding: 8px modulo
- Device: CUDA if available, CPU fallback
- Refinement enabled
- Pre-loaded model instance for faster inference

## Performance Notes

- GPU with 24GB ram, or 2 with 12 each is recommended. By default it uses 2, otherwise edit the config in lama/configs/prediction. To use only one, set `gpu_ids` to 0.
- Model preloading improves response time. (~2s on most GPU's without refinement, ~20s with refinement)
- Processing time varies with image size.

## API Usage

See `app.py`

## Credits

- LaMa model: [Original Repository](https://github.com/advimman/lama)
- YOLO implementation: [YOLOv8](https://github.com/ultralytics/ultralytics)
