from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import wandb
import uvicorn
import os
import torchvision.transforms as transforms
from gtts import gTTS
import logging
from typing import Dict, Any, Optional
import time
from pathlib import Path
import torchvision.models as models
import timm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_CONFIG = {
    "xception41": "model:v185",
    "inception_v4": "model:v205",
    "efficientnet_b0": "model:v161",
    "convnextv2_tiny": "model:v278",
    "swin_tiny_patch4_window7_224": "model:v105",
    "deit3_base_patch16_224": "model:v229"
}

CLASS_NAMES = {
    0: 'Benign, Likely refers to normal or non-cancerous B-cells.',
    1: '[Malignant] Pre-B, Malignant precursor B-cells, indicating an early form of B-cell ALL.',
    2: '[Malignant] Pro-B, A more primitive stage of B-cell ALL, possibly Pro-B ALL, which is often aggressive.',
    3: '[Malignant] early Pre-B, Likely an intermediate stage between Pro-B and Pre-B ALL.'
}

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224
AUDIO_OUTPUT_PATH = Path("output.mp3")

# Initialize FastAPI app
app = FastAPI(
    title="Blood Cells Cancer Classification API",
    description="API for classifying blood cells using various deep learning models",
    version="2.0.0"
)

class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._model_cache = {}

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess the input image for model inference."""
        try:
            return self.transform(image).unsqueeze(0)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image format")

    def load_model(self, model_name: str) -> torch.nn.Module:
        """Load model from cache or wandb."""
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        try:
            run = wandb.init(project="Blood-Cells-Cancer-ALL", resume="allow", reinit=True)
            model_path = run.use_artifact(MODEL_CONFIG[model_name]).download()
            
            if os.path.isdir(model_path):
                model_files = [f for f in os.listdir(model_path) if f.endswith((".pth", ".pt"))]
                if not model_files:
                    raise FileNotFoundError(f"No model file found in {model_path}")
                model_path = os.path.join(model_path, model_files[0])

            model = self._create_model_architecture(model_name)
            model.load_state_dict(torch.load(model_path), strict=False)
            model.eval()
            model.to(self.device)
            
            self._model_cache[model_name] = model
            return model

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    def _create_model_architecture(self, model_name: str) -> torch.nn.Module:
        """Create and configure model architecture."""
        try:
            model = timm.create_model(model_name, pretrained=False)
            
            if hasattr(model, "classifier"):
                in_features = model.classifier.in_features
                model.classifier = torch.nn.Linear(in_features, NUM_CLASSES)
            elif hasattr(model, "fc"):
                in_features = model.fc.in_features
                model.fc = torch.nn.Linear(in_features, NUM_CLASSES)
            elif hasattr(model, "head"):
                if hasattr(model.head, "fc"):
                    in_features = model.head.fc.in_features
                    model.head.fc = torch.nn.Linear(in_features, NUM_CLASSES)
                elif hasattr(model.head, "classifier"):
                    in_features = model.head.classifier.in_features
                    model.head.classifier = torch.nn.Linear(in_features, NUM_CLASSES)
                elif isinstance(model.head, torch.nn.Linear):
                    in_features = model.head.in_features
                    model.head = torch.nn.Linear(in_features, NUM_CLASSES)
                else:
                    raise AttributeError(f"Could not find a replaceable layer in {model_name}")
            elif hasattr(model, "last_linear"):
                model.last_linear = torch.nn.Linear(in_features=1536, out_features=NUM_CLASSES)
            else:
                raise AttributeError(f"No valid classification layer found in {model_name}")

            return model

        except Exception as e:
            logger.error(f"Error creating model architecture: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error creating model: {str(e)}")

    def predict(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Make prediction and measure performance."""
        try:
            input_tensor = input_tensor.to(self.device)
            
            start_time = time.time()
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1).max().item()
            
            execution_time = time.time() - start_time
            fps = 1 / execution_time

            return {
                "prediction": prediction,
                "class_name": CLASS_NAMES[prediction],
                "confidence": confidence,
                "execution_time": execution_time,
                "fps": fps,
                "device": str(self.device)
            }

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

    def generate_speech(self, prediction_result: Dict[str, Any]) -> None:
        """Generate and play speech output."""
        try:
            speech_text = (
                f"The prediction of types of B-cell development of Acute Lymphoblastic Leukemia is "
                f"{prediction_result['class_name']} with a confidence of {prediction_result['confidence']:.2f}. "
                f"Execution time is {prediction_result['execution_time']:.2f} seconds with "
                f"{prediction_result['fps']:.2f} frames per second."
            )
            
            speech = gTTS(text=speech_text, lang='en')
            speech.save(str(AUDIO_OUTPUT_PATH))
            os.system("mpg321 output.mp3")
            
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            # Don't raise exception as speech is not critical

# Initialize model manager
model_manager = ModelManager()

@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    """Endpoint for making predictions."""
    if model_name not in MODEL_CONFIG:
        raise HTTPException(status_code=400, detail="Invalid model name")

    try:
        # Load and preprocess image
        image = Image.open(file.file).convert("RGB")
        input_tensor = model_manager.preprocess_image(image)
        
        # Load model and make prediction
        model = model_manager.load_model(model_name)
        prediction_result = model_manager.predict(model, input_tensor)
        
        # Generate speech output
        model_manager.generate_speech(prediction_result)
        
        return prediction_result

    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_available_models():
    """Endpoint for getting available models."""
    return {"available_models": list(MODEL_CONFIG.keys())}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
