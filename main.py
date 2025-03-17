from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import wandb
import uvicorn
import os
import torchvision.models as models
import timm
import torchvision.transforms as transforms

app = FastAPI()

# Preprocessing function
def preprocess_image(image: Image.Image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

# Load model from wandb
def load_model(wandb_code: str, model_name: str):
    run = wandb.init(project="Blood-Cells-Cancer-ALL",resume="allow", reinit=True)
    model_path = run.use_artifact(wandb_code).download()
    model_dir = model_path
    if os.path.isdir(model_dir):
        # List files in the directory
        files = os.listdir(model_dir)
        print("Directory contents:", files)  # Debugging
        
        # Search for a model file (pth or pt)
        model_files = [f for f in files if f.endswith(".pth") or f.endswith(".pt")]
        
        if not model_files:
            raise FileNotFoundError(f"No model file found in {model_dir}")

        # Take the first model file found
        model_path = os.path.join(model_dir, model_files[0])
        print(f"Loading model from: {model_path}")
    else:
        model_path = model_dir  # If it's already a file, use it directly

    # Define the architecture based on model_name
    if model_name == 'xception41':
        model = timm.create_model('xception41', pretrained=False)
         # Adjust final layer for 4 classes
        model.head.fc = torch.nn.Linear(model.head.fc.in_features, 4)
    elif model_name == 'inception_v4':
        model = timm.create_model('inception_v4', pretrained=False)
    elif model_name == 'swinT':
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
    elif model_name == 'convnextv2_tiny':
        model = timm.create_model('convnextv2_tiny', pretrained=False)
    elif model_name == 'deit3':
        model = timm.create_model('deit3_tiny_patch16_224', pretrained=False)
    elif model_name == 'efficientNetb0':
        model = timm.create_model('efficientnet_b0', pretrained=False)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
     # Modify the final layer to match the number of classes in your task
    if hasattr(model, 'head'):
        in_features = model.head.fc.in_features  # Get the number of input features to the final layer
        model.head.fc = torch.nn.Linear(in_features, 4)  # Set the output to 4 classes
    else:
        print("Model does not have a 'head' attribute. Modify accordingly based on model architecture.")
    
    # Load model weights
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    return model

# Define available wandb RunIDs for each model
wandb_codes = {
    "xception41": "model:v185",
    "inception_v4": "model:v205",
    "efficientNet_b0": "model:v161",
    "convnextv2_tiny": "model:v278",
    "swinT": "model:v105",
    "deit3": "model:v229"
}

# Define class names for the predictions
class_names = {
    0: 'Benign',
    1: '[Malignant] Pre-B',
    2: '[Malignant] Pro-B',
    3: '[Malignant] early Pre-B'
}

@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    if model_name not in wandb_codes:
        return JSONResponse(status_code=400, content={"message": "Invalid model name"})

    # Load the model
    model = load_model(wandb_codes[model_name], model_name)

    # Read and preprocess the image
    image = Image.open(file.file).convert("RGB")
    input_tensor = preprocess_image(image)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return {"prediction": prediction, "class_name": class_names[prediction]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
