import cv2
import numpy as np
from vae_module import (
    show,
    normalize,
    pil_to_latents,
    latents_to_np,
    pil_to_np,
    load_img,
    matplot_create,
    matplot_show,
    restack,
    diff,
    side_by_side,
)

img_link = 'https://upload.wikimedia.org/wikipedia/commons/1/1c/Squirrel_posing.jpg'


img = load_img(img_link)
latent_img = pil_to_latents(img)
np_img = pil_to_np(img)

unstacked_images = matplot_create(latent_img)
matplot_show()

decoded_img = latents_to_np(latent_img)
stacked_img = restack(unstacked_images)
compare_img = side_by_side(np_img, decoded_img)

cv2.imwrite("output/latent_rep.png", stacked_img)
cv2.imwrite("output/comparison_img.png", compare_img)
cv2.imwrite("output/np_img.png", np_img)
cv2.imwrite("output/decoded_img.png", decoded_img)
