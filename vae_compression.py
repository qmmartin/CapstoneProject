import cv2
import numpy as np 
import torch
import matplotlib.pyplot as plt 
from PIL import Image 
from diffusers import AutoencoderKL
from fastdownload import FastDownload  
from torchvision import transforms as tfms

vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float32).to("cuda")

def load_image(p):
   return Image.open(p).convert('RGB').resize((512,512))

def pil_to_latents(image):   
    init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0   
    init_image = init_image.to(device="cuda", dtype=torch.float32)
    init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215     
    return init_latent_dist  

def latents_to_pil(latents):     
    latents = (1 / 0.18215) * latents     
    with torch.no_grad():         
        image = vae.decode(latents).sample     
    
    image = (image / 2 + 0.5).clamp(0, 1)     
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()      
    images = (image * 255).round().astype("uint8")     
    pil_images = [Image.fromarray(image) for image in images]        
    return pil_images

p = FastDownload().download('https://upload.wikimedia.org/wikipedia/commons/1/1c/Squirrel_posing.jpg')

img = load_image(p)

# Image Load Test
# img_np = np.array(img)
# img_np = img_np[:,:,::-1]
# show(img_np)

print(f"Dimension of this image: {np.array(img).shape}")

latent_img = pil_to_latents(img)

print(f"Dimension of this latent representation: {latent_img.shape}")

# Channel Values Test
# for c in range(4):
#     img = latent_img[0, c, :, :].detach().cpu()
#     print(f"Channel {c}: Min={img.min()}, Max={img.max()}")

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for c in range(4):
    img = latent_img[0, c, :, :].detach().cpu()
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
    axs[c].imshow(img, cmap='Greys')
    axs[c].axis('off')
plt.show()

# decoded_img = latents_to_pil(latent_img)
# decoded_img[0]


