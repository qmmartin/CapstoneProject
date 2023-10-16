import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline 

# print(torch.cuda.is_available()) 

# Initiating tokenizer and encoder.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda")

repo_id = "runwayml/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_auth_token={"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"} ## GPU ONLY
)

pipe = pipe.to("cuda")

prompt = "a cat wearing glasses"

pipe(prompt).images[0]

prompt = ["a cat wearing glasses"]
tok =tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt") 
print(tok.input_ids.shape)
tok

for token in list(tok.input_ids[0,:7]): 
    print(f"{token}:{tokenizer.convert_ids_to_tokens(int(token))}")

emb = text_encoder(tok.input_ids.to("cuda"))[0].half()
print(f"Shape of embedding : {emb.shape}")
emb