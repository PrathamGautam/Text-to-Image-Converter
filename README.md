Stable Diffusion Image Generation
This repository provides a pipeline for generating images from text prompts using the Stable Diffusion model. The code leverages the diffusers library from Hugging Face and uses Gradio for creating an interactive web interface.

Installation
To get started, you need to install the necessary libraries. You can do this by running:

sh
Copy code
pip install diffusers
pip install --upgrade diffusers transformers scipy
pip install gradio
Additionally, make sure you have the NVIDIA driver installed:

sh
Copy code
!apt-get update
!apt-get install -y nvidia-driver
!nvidia-smi
Usage
Initial Setup
First, import the necessary libraries and load the pre-trained model:

python
Copy code
from diffusers import DiffusionPipeline
import torch

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)
Generating Images
You can generate images by specifying a prompt:

python
Copy code
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

image.save("astronaut_rides_horse.png")
Advanced Image Generation
You can use additional parameters to control the image generation process:

python
Copy code
from torch import autocast

prompt = "Man with his tea cup."
negative_prompt = "not give black and white image"
num_samples = 1
guidance_scale = 6.2
num_inference_steps = 20
height = 512
width = 512

with autocast("cuda"), torch.inference_mode():
    images = pipe(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images

for img in images:
    display(img)
Gradio Interface
To make the image generation interactive, we use Gradio to create a web interface:

python
Copy code
import gradio as gr

def generate_image(prompt, negative_prompt, num_samples, guidance_scale, num_inference_steps, height, width):
    with torch.autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images
    return images

iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here"),
        gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here"),
        gr.Slider(1, 10, value=1, step=1, label="Number of Samples"),
        gr.Slider(0.1, 10, value=6.2, step=0.1, label="Guidance Scale"),
        gr.Slider(1, 50, value=20, step=1, label="Number of Inference Steps"),
        gr.Slider(64, 1024, value=512, step=64, label="Height"),
        gr.Slider(64, 1024, value=512, step=64, label="Width")
    ],
    outputs=gr.Gallery(label="Generated Images"),
    title="Stable Diffusion Image Generation",
    description="Generate images from text prompts using the Stable Diffusion model.",
)

iface.launch()
Example Outputs
Here are some example images generated using the prompts provided:

Astronaut Riding a Horse on Mars
Astronaut Riding a Horse on Mars

Man with his Tea Cup

You can customize the prompts and parameters to generate a wide variety of images.

Contributing
Feel free to open issues or submit pull requests if you have suggestions for improvements or find any bugs.

License
This project is licensed under the MIT License.
