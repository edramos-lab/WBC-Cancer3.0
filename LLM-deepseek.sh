#!/bin/bash

# Update and install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget python3 python3-pip python3-venv

# Create a virtual environment
python3 -m venv deepseek-env
source deepseek-env/bin/activate

# Install PyTorch (modify based on your GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers and other dependencies
pip install transformers accelerate bitsandbytes

# Clone DeepSeek repository
git clone https://github.com/DeepSeek-AI/DeepSeek-LLM.git
cd DeepSeek-LLM

# Install DeepSeek dependencies
pip install -r requirements.txt

# Download DeepSeek model (modify for specific model size)
wget https://huggingface.co/deepseek-ai/deepseek-llm-7b/resolve/main/pytorch_model.bin
wget https://huggingface.co/deepseek-ai/deepseek-llm-7b/resolve/main/config.json

# Run DeepSeek model using Python
cat << EOF > run_deepseek.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b", torch_dtype=torch.float16, device_map="auto")

prompt = "Hello, how can I help you today?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_length=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
EOF

# Inform user to run Python script
echo "DeepSeek installation complete! To run the model, execute:"
echo "source deepseek-env/bin/activate && python run_deepseek.py"

