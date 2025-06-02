# WBC-Cancer

## Project Description
WBC-Cancer is a deep learning-powered API for classifying White Blood Cells (WBC) and their developmental phases in Acute Lymphoblastic Leukemia (ALL). Built with FastAPI, it provides:
- Automated image-based classification of WBC types and ALL phases
- Voice feedback for predictions
- Inference performance metrics (execution time, FPS)

## Features
- Classifies WBC images into benign and malignant (Pre-B, Pro-B, early Pre-B) ALL phases
- Multiple model architectures supported (e.g., Xception, EfficientNet, Swin, etc.)
- REST API with FastAPI
- Voice output using gTTS
- Reports inference time and frames per second (FPS)

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd WBC-Cancer3.0
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
2. Access the API docs at [http://localhost:8000/docs](http://localhost:8000/docs)
3. To classify an image, use the `/predict/{model_name}` endpoint (POST) with an image file. Example using `curl`:
   ```bash
   curl -X POST "http://localhost:8000/predict/efficientnet_b0" -F "file=@path_to_image.jpg"
   ```
   The response includes the predicted class, confidence, execution time, FPS, and device. The result is also spoken aloud via your system audio.

4. To list available models:
   ```bash
   curl http://localhost:8000/models
   ```

## API Endpoints
- `POST /predict/{model_name}`: Classify a WBC image. Returns prediction, confidence, execution time, FPS, and device. Triggers voice output.
- `GET /models`: List available model names.

## Project Structure
- `main.py` — FastAPI app, model management, prediction, and voice output
- `artifacts/` — Pretrained model files (downloaded via wandb)
- `requirements.txt` — Python dependencies
- `genAI.py` — (Optional) Image generation for B-cell classes
- `CreateFile.py`, `file_locations.csv`, `file_locations.txt` — Utility/data files
- `text2speech.py` — Voice synthesis examples

## Requirements
- fastapi==0.95.0
- uvicorn==0.22.0
- Pillow==9.3.0
- torch==2.0.1
- wandb==0.15.0
- torchvision==0.15.0
- timm==0.9.2
- gTTS==2.2.3

## License
See [LICENSE](LICENSE) for details.