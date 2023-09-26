import torch, logging
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline 

pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5').to('cuda')

# Initialize a prompt
prompt = "a cat wearing glasses"

# Pass the prompt in the pipeline
pipe(prompt).images[0]
