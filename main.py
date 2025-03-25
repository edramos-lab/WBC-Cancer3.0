from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import wandb
import uvicorn
import os
import torchvision.models as models
import timm
import time
import torchvision.transforms as transforms
<<<<<<< HEAD
from gtts import gTTS
=======
>>>>>>> d7002967ced4321d88869bcc5bb6898b75e8fc1f

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
<<<<<<< HEAD
import os
import torch
import timm
import wandb

def load_model(wandb_code: str, model_name: str):
    run = wandb.init(project="Blood-Cells-Cancer-ALL", resume="allow", reinit=True)
    model_path = run.use_artifact(wandb_code).download()
    
    if os.path.isdir(model_path):
        files = os.listdir(model_path)
        print("📂 Directory contents:", files)  # Debugging
        
        model_files = [f for f in files if f.endswith(".pth") or f.endswith(".pt")]
        if not model_files:
            raise FileNotFoundError(f"🚨 No model file found in {model_path}")
        
        model_path = os.path.join(model_path, model_files[0])
        print(f"📌 Loading model from: {model_path}")

    # Load the model architecture
    model = timm.create_model(model_name, pretrained=False)

    num_classes = 4  # Number of classes for classification

    # Modify the classifier layer based on model type
    if hasattr(model, "classifier"):  # EfficientNet, ConvNeXt
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes)
    elif hasattr(model, "fc"):  # ResNet and similar architectures
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif hasattr(model, "head"):  # Inception, Swin, DeiT
        if hasattr(model.head, "fc"):  # InceptionV4 & similar models
            in_features = model.head.fc.in_features
            model.head.fc = torch.nn.Linear(in_features, num_classes)
        elif hasattr(model.head, "classifier"):  # Some Swin and DeiT variants
            in_features = model.head.classifier.in_features
            model.head.classifier = torch.nn.Linear(in_features, num_classes)
        elif isinstance(model.head, torch.nn.Linear):  # Directly a Linear layer
            in_features = model.head.in_features
            model.head = torch.nn.Linear(in_features, num_classes)
        else:
            raise AttributeError(f"⚠️ Could not find a replaceable layer in {model_name}")
    elif hasattr(model, "last_linear"):  # Check if layer exists
        model.last_linear = torch.nn.Linear(in_features=1536, out_features=num_classes)
    else:
        raise AttributeError(f"⚠️ No valid classification layer found in {model_name}")

    print(f"✅ Model {model_name} modified successfully!\n")

    # Load model weights with strict=False to avoid mismatches
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    return model


=======
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

>>>>>>> d7002967ced4321d88869bcc5bb6898b75e8fc1f
# Define available wandb RunIDs for each model
wandb_codes = {
    "xception41": "model:v185",
    "inception_v4": "model:v205",
<<<<<<< HEAD
    "efficientnet_b0": "model:v161",
    "convnextv2_tiny": "model:v278",
    "swin_tiny_patch4_window7_224": "model:v105",
    "deit3_base_patch16_224": "model:v229"
=======
    "efficientNet_b0": "model:v161",
    "convnextv2_tiny": "model:v278",
    "swinT": "model:v105",
    "deit3": "model:v229"
>>>>>>> d7002967ced4321d88869bcc5bb6898b75e8fc1f
}

# Define class names for the predictions
class_names = {
<<<<<<< HEAD
    0: 'Benign, Likely refers to normal or non-cancerous B-cells.',
    1: '[Malignant] Pre-B, Malignant precursor B-cells, indicating an early form of B-cell ALL.',
    2: '[Malignant] Pro-B, A more primitive stage of B-cell ALL, possibly Pro-B ALL, which is often aggressive.',
    3: '[Malignant] early Pre-B, Likely an intermediate stage between Pro-B and Pre-B ALL.'
=======
    0: 'Benign',
    1: '[Malignant] Pre-B',
    2: '[Malignant] Pro-B',
    3: '[Malignant] early Pre-B'
>>>>>>> d7002967ced4321d88869bcc5bb6898b75e8fc1f
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

    # Check if GPU is available and move the model and tensor to the appropriate device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        model.to(device)
        input_tensor = input_tensor.to(device)

        # Measure execution time
        start_time = time.time()

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        end_time = time.time()
        GPU_execution_time = end_time - start_time
        GPU_fps = 1 / GPU_execution_time

        device = "cpu"
        model.to(device)
        input_tensor = input_tensor.to(device)

        # Measure execution time
        start_time = time.time()

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
<<<<<<< HEAD
            confidence = torch.softmax(output, dim=1).max().item()
=======
>>>>>>> d7002967ced4321d88869bcc5bb6898b75e8fc1f

        end_time = time.time()
        CPU_execution_time = end_time - start_time
        CPU_fps = 1 / CPU_execution_time

<<<<<<< HEAD
        speech_text = f"The prediction of types of B-cell development of Acute Lymphoblastic Leukemia is {class_names[prediction]} with a confidence of {confidence:.2f}. GPU execution time is {GPU_execution_time:.2f} seconds with {GPU_fps:.2f} frames per second. CPU execution time is {CPU_execution_time:.2f} seconds with {CPU_fps:.2f} frames per second."
        speech = gTTS(text=speech_text, lang='en')
        speech.save("output.mp3")
        os.system("mpg321 output.mp3")  # Plays the audio

        return {
            "prediction": prediction,
            "class_name": class_names[prediction],
            "confidence": confidence,
=======
        return {
            "prediction": prediction,
            "class_name": class_names[prediction],
>>>>>>> d7002967ced4321d88869bcc5bb6898b75e8fc1f
            "GPU_execution_time": GPU_execution_time,
            "GPU_fps": GPU_fps,
            "CPU_execution_time": CPU_execution_time,
            "CPU_fps": CPU_fps,
            "device": str(device)
        }
    else:
        model.to(device)
        input_tensor = input_tensor.to(device)  # Move the input data to the device
        start_time = time.time()  # Measure execution time
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
<<<<<<< HEAD
            confidence = torch.softmax(output, dim=1).max().item()
=======
>>>>>>> d7002967ced4321d88869bcc5bb6898b75e8fc1f
        end_time = time.time()
        CPU_execution_time = end_time - start_time
        CPU_fps = 1 / CPU_execution_time
        return {
            "prediction": prediction,
            "class_name": class_names[prediction],
<<<<<<< HEAD
            "confidence": confidence,
=======
>>>>>>> d7002967ced4321d88869bcc5bb6898b75e8fc1f
            "execution_time": CPU_execution_time,
            "fps": CPU_fps,
            "device": str(device)
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
<<<<<<< HEAD
    
=======
>>>>>>> d7002967ced4321d88869bcc5bb6898b75e8fc1f
