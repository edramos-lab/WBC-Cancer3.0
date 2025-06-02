import os
import argparse
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from huggingface_hub import login

# Get token from environment variable
hf_token = os.getenv('HUGGINGFACE_TOKEN')
if not hf_token:
    raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable")
login(hf_token)

def generate_image(prompt, width, height, model_id="stabilityai/stable-diffusion-3-medium"):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None,torch_dtype=torch.float16)
    pipe.to("cuda")
    image = pipe(prompt,width=width, height=height).images[0]
    return image

# Define class descriptions
class_prompts = {
    0: "A healthy benign B-cell under a microscope",
    1: "A malignant Pre-B cell showing early leukemia",
    2: "A malignant Pro-B cell in advanced leukemia",
    3: "An early Pre-B cell in intermediate leukemia stage"
}

def main(num_images_per_class, output_path, width, height):
    # Generate and save images in respective folders
    os.makedirs(output_path, exist_ok=True)

    for class_id, prompt in class_prompts.items():
        class_folder = os.path.join(output_path, f"class_{class_id}")
        os.makedirs(class_folder, exist_ok=True)
        
        for i in range(num_images_per_class):
            img = generate_image(prompt, width, height)
            img.save(os.path.join(class_folder, f"bcell_class_{class_id}_{i + 1}.png"))
    
    print("Images generated and saved in respective folders successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images for B-cell classes using Stable Diffusion.")
    parser.add_argument("num_images_per_class", type=int, help="Number of images to generate per class.")
    parser.add_argument("output_path", type=str, help="Path where the image folder will be created.")
    parser.add_argument("width", type=int, help="Width of the generated images.")
    parser.add_argument("height", type=int, help="Height of the generated images.")
    args = parser.parse_args()

    if args.num_images_per_class <= 0:
        print("Number of images must be a positive integer.")
        sys.exit(1)
    
    if args.width <= 0 or args.height <= 0:
        print("Width and height must be positive integers.")
        sys.exit(1)
    
    main(args.num_images_per_class, args.output_path, args.width, args.height) 