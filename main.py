import cv2
import numpy as np
from vae_module import (
    pil_to_latents,
    latents_to_np,
    pil_to_np,
    load_img,
    matplot_create,
    matplot_show,
    restack,
    side_by_side,
    compress_and_save,
)
from friqa_module import (
    ssim_iqa,
)


link = 'https://upload.wikimedia.org/wikipedia/commons/1/1c/Squirrel_posing.jpg'

compress_and_save(link)


# img = load_img(link)
# latent_img = pil_to_latents(img)
# np_img = pil_to_np(img)

# unstacked_images = matplot_create(latent_img)
# # matplot_show()

# decoded_img = latents_to_np(latent_img)
# stacked_img = restack(unstacked_images)
# compare_img = side_by_side(np_img, decoded_img)

# cv2.imwrite("output/latent_rep.png", stacked_img)
# cv2.imwrite("output/comparison_img.png", compare_img)
# cv2.imwrite("output/np_img.png", np_img)
# cv2.imwrite("output/decoded_img.png", decoded_img)

# ssim_iqa(np_img, decoded_img)

