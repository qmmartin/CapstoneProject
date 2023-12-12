import cv2
import numpy as np 
import torch
import matplotlib.pyplot as plt 
from PIL import Image 
from diffusers import AutoencoderKL
from fastdownload import FastDownload  
from torchvision import transforms as tfms




# Debug functions

# Tests the dimensions of an image and prints them
def dimensions_test(img, latent_img):
    print(f"Dimension of this image: {np.array(img).shape}")
    print(f"Dimension of this latent representation: {latent_img.shape}")

# Tests the values in an image's channels
def channel_vals_test(latent_img):
    for c in range(4):
        img = latent_img[0, c, :, :].detach().cpu()
        print(f"Channel {c}: Min={img.min()}, Max={img.max()}")

# Tests if an image is in a NumPy usable format
def test_np(img):
    print("Image shape:", img.shape)
    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
        show(img)
    else:
        print("Image is invalid or has an incorrect shape.")

# Tests if an image is loading properly
def image_test(img):
    img_np = np.array(img)
    img_np = img_np[:,:,::-1]
    show(img_np)

    d



# Load functions

# Loads the vae from Hugging Face
def load_vae():
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float32).to("cuda")
    return(vae)

# Loads an image from the web
def load_image(p):
   return Image.open(p).convert('RGB')

# Downloads an image from a given URL
def load_img(link):
    p = FastDownload().download(link)
    img = load_image(p)
    return(img)

# Loads an image from path
def load_np(path):
    img = cv2.imread(path)
    img = img[:,:,::-1]
    return(img)




# Image functions

# Shows an image using cv2
def show(img):
    cv2.imshow("img",img)
    cv2.waitKey(0)

# Normalizes an image
def normalize(img):
    img=img-img.min()
    img=img/img.max()
    return np.uint8(255*img)

# Calculates the difference of the images
def diff(img1, img2):
    diff = img1-img2
    return(diff)

# Horizontally stacks an array of images
def restack(unstacked_images):
    stacked_image = np.hstack(unstacked_images)
    return(stacked_image)

# Horizontally stacks two separate images
def side_by_side(img, img2):
    images = [img, img2]
    sbs = np.hstack(images)
    return(sbs)




# Matplot functions

# Creates a matplot of the latent representation of an image
def matplot_create(latent_img):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    unstacked_images = []
    for c in range(4):
        img = latent_img[0, c, :, :].detach().cpu()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        axs[c].imshow(img, cmap='Greys')
        axs[c].axis('off')
        unstacked_images.append(255 - (img * 255).numpy().astype('uint8'))
    return(unstacked_images)

# Shows a matplot
def matplot_show():
       plt.show()




# Conversion Functions

# Converts an image from a numpy array to a PIL format
def np_to_pil(image):
    pil_image = Image.fromarray(image)
    pil_image = pil_image.convert('RGB')
    return pil_image

# Converts an image from PIL to a NumPy array
def pil_to_np(pil_image):
    np_array = np.array(pil_image)
    np_array = np_array[:,:,::-1]
    return np_array

# Converts an image from PIL to latent representation
def pil_to_latents(image, scalar):   
    vae = load_vae()
    init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0   
    init_image = init_image.to(device="cuda", dtype=torch.float32)
    init_latent_dist = vae.encode(init_image).latent_dist.sample() * scalar
    return init_latent_dist  

# Converts an image from latent representation to NumPy array
def latents_to_np(latents, scalar):
    vae = load_vae()
    scalar = 0.18215
    latents = (1 / scalar) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].detach().cpu().permute(1, 2, 0).numpy() * 255
    image = image.round().astype("uint8")
    image = image[:, :, ::-1]  # Fix Numpy's BGR weirdness
    return image

# Converts an image from latent representation to PIL
def latents_to_pil(latents, scalar): 
    vae = load_vae()    
    latents = (1 / scalar) * latents     
    with torch.no_grad():         
        image = vae.decode(latents).sample     
    
    image = (image / 2 + 0.5).clamp(0, 1)     
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()      
    images = (image * 255).round().astype("uint8")     
    pil_images = [Image.fromarray(image) for image in images]        
    return pil_images

count=0

# Other functions

def compress_and_save(img, scalar):
    global count

    if not isinstance(img, Image.Image):
        img = np_to_pil(img)
    img = img.resize((512,512))

    np_img = pil_to_np(img)

    latent_img = pil_to_latents(img, scalar)
    unstacked_images = matplot_create(latent_img)
    
    decoded_img = latents_to_np(latent_img, scalar)
    stacked_img = restack(unstacked_images)

    compare_img = side_by_side(np_img, decoded_img)

    count = count+1

    img1 = (f"output/latent_rep{count:02}.png")
    img2 = (f"output/comparison_img{count:02}.png")
    img3 = (f"output/np_img{count:02}.png")
    img4 = (f"output/decoded_img{count:02}.png")

    cv2.imwrite(img1, stacked_img)
    cv2.imwrite(img2, compare_img)
    cv2.imwrite(img3, np_img)
    cv2.imwrite(img4, decoded_img)



