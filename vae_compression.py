import cv2
import numpy as np 
import torch
import matplotlib.pyplot as plt 
from PIL import Image 
from diffusers import AutoencoderKL
from fastdownload import FastDownload  
from torchvision import transforms as tfms

vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float32).to("cuda")

# IMAGE FUNCTIONS 
def show(img):
    cv2.imshow("img",img)
    cv2.waitKey(0)

def load_image(p):
   return Image.open(p).convert('RGB').resize((512,512))

def test_np(img):
    print("Image shape:", img.shape)
    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
        show(img)
    else:
        print("Image is invalid or has an incorrect shape.")

def normalize(img):
    img=img-img.min()
    img=img/img.max()
    return np.uint8(255*img)

# VAE FUNCTIONS
def pil_to_latents(image):   
    init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0   
    init_image = init_image.to(device="cuda", dtype=torch.float32)
    init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215     
    return init_latent_dist  

# def latents_to_pil(latents):     
#     latents = (1 / 0.18215) * latents     
#     with torch.no_grad():         
#         image = vae.decode(latents).sample     
    
#     image = (image / 2 + 0.5).clamp(0, 1)     
#     image = image.detach().cpu().permute(0, 2, 3, 1).numpy()      
#     images = (image * 255).round().astype("uint8")     
#     pil_images = [Image.fromarray(image) for image in images]        
#     return pil_images

def latents_to_np(latents):
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].detach().cpu().permute(1, 2, 0).numpy() * 255
    image = image.round().astype("uint8")
    image = image[:,:,::-1] # Fix Numpy's BGR weirdness
    return image

def pil_to_np(pil_image):
    # Convert a PIL Image to a NumPy array
    np_array = np.array(pil_image)
    return np_array

p = FastDownload().download('https://upload.wikimedia.org/wikipedia/commons/1/1c/Squirrel_posing.jpg')

img = load_image(p)

# IMAGE LOAD TEST
# img_np = np.array(img)
# img_np = img_np[:,:,::-1]
# show(img_np)

print(f"Dimension of this image: {np.array(img).shape}")

latent_img = pil_to_latents(img)

print(f"Dimension of this latent representation: {latent_img.shape}")

# CHANNEL VALUES TEST
# for c in range(4):
#     img = latent_img[0, c, :, :].detach().cpu()
#     print(f"Channel {c}: Min={img.min()}, Max={img.max()}")
img2=np.array(img)
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
stacked_images = []
for c in range(4):
    img = latent_img[0, c, :, :].detach().cpu()
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
    axs[c].imshow(img, cmap='Greys')
    axs[c].axis('off')
    stacked_images.append((img * 255).numpy().astype('uint8'))
plt.show()

# Horizontally stack the images
stacked_image = np.hstack(stacked_images)
cv2.imwrite("output/vae_success.png", stacked_image)

decoded_img = latents_to_np(latent_img)
# decoded_img[0]
# show(decoded_img)
np_img = img2[:,:,::-1] # Fix BGR weirdness
stacked_img = np.hstack((np_img, decoded_img))
show(stacked_img)

show(normalize(np_img-decoded_img))

cv2.imwrite("output/comparison_img.png", stacked_img)
cv2.imwrite("output/np_img.png", np_img)
cv2.imwrite("output/decoded_img.png", decoded_img)


