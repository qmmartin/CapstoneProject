import cv2
import numpy as np 
import torch
import matplotlib.pyplot as plt 
from PIL import Image 
from diffusers import AutoencoderKL
from fastdownload import FastDownload  
from torchvision import transforms as tfms


vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16).to("cuda")

def show(img):
    cv2.imshow("img",img)
    cv2.waitKey(0)

def load_image(p):
   '''     
   Function to load images from a defined path     
   '''    
   return Image.open(p).convert('RGB').resize((512,512))

def pil_to_latents(image):
    '''     
    Function to convert image to latents     
    '''     
    init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0   
    init_image = init_image.to(device="cuda", dtype=torch.float16)
    init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215     
    return init_latent_dist  

def latents_to_pil(latents):     
    '''     
    Function to convert latents to images     
    '''     
    latents = (1 / 0.18215) * latents     
    with torch.no_grad():         
        image = vae.decode(latents).sample     
    
    image = (image / 2 + 0.5).clamp(0, 1)     
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()      
    images = (image * 255).round().astype("uint8")

    pil_images = [Image.fromarray(image) for image in images]        
    return pil_images

p = FastDownload().download('https://lafeber.com/pet-birds/wp-content/uploads/2018/06/Scarlet-Macaw-2.jpg')
img = load_image(p)
print(f"Dimension of this image: {np.array(img).shape}")
img

latent_img = pil_to_latents(img)
print(f"Dimension of this latent representation: {latent_img.shape}")

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for c in range(4):
    axs[c].imshow(latent_img[0][c].detach().cpu(), cmap='Greys')

decoded_img = latents_to_pil(latent_img)
decoded_img[0]

# show(axs[1])

# combined_image = np.hstack([latent_img[0][c].detach().cpu().numpy() for c in range(4)])
# combined_image = (combined_image * 255).astype("uint8")
# show(combined_image)

