# Stable Diffusion Image Generation
This repository provides a pipeline for generating images from text prompts using the Stable Diffusion model. The code leverages the diffusers library from Hugging Face and uses Gradio for creating an interactive web interface.
## Gradio Interface
To make the image generation interactive, we use Gradio to create a web interface:
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

## Output of Code
![image](https://github.com/PrathamGautam/Text-to-Image-Converter/assets/142311958/b977e47c-1d77-4d2b-928d-ce838359a229)

![image](https://github.com/PrathamGautam/Text-to-Image-Converter/assets/142311958/8ad438a2-6fe6-4762-a530-4008d25c5d35)

## Contributing
Feel free to open issues or submit pull requests if you have suggestions for improvements or find any bugs.

## License
This project is licensed under the MIT License.
