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


animal_link = 'https://upload.wikimedia.org/wikipedia/commons/1/1c/Squirrel_posing.jpg'
cityscape_link = 'https://upload.wikimedia.org/wikipedia/commons/0/05/View_of_Empire_State_Building_from_Rockefeller_Center_New_York_City_dllu.jpg'
text_link = 'https://upload.wikimedia.org/wikipedia/commons/d/df/Neon.JPG'
art_link = 'https://upload.wikimedia.org/wikipedia/commons/a/aa/Van_Gogh_-_Starry_Night_2.jpg'
sax_link = 'https://upload.wikimedia.org/wikipedia/commons/e/e6/Alto_saxophone-E_1685-IMG_7092-gradient.jpg'

compress_and_save(animal_link)
compress_and_save(cityscape_link)
compress_and_save(text_link)
compress_and_save(art_link)
compress_and_save(sax_link)


# ssim_iqa(np_img, decoded_img)

