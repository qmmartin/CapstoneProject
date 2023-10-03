import torch, logging
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import huggingface_hub

# print(torch.cuda.is_available())

# ## disable warnings
# logging.disable(logging.WARNING)  

# ## Import the CLIP artifacts 
# from transformers import CLIPTextModel, CLIPTokenizer

# ## Initiating tokenizer and encoder.
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
# text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda")

repo_id = "./v1-5-pruned-emaonly/"
pipe = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)

# Initialize a prompt
prompt = "a cat wearing glasses"

# Pass the prompt in the pipeline
pipe(prompt).images[0]
