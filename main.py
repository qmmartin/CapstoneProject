import cv2
import numpy as np
from vae_module import (
    np_to_pil,
    pil_to_latents,
    latents_to_np,
    pil_to_np,
    load_img,
    matplot_create,
    matplot_show,
    restack,
    side_by_side,
    vae_compress,
    load_np,
    jpeg_compress,
    webp_compress,
    pil_to_webp,
)


animal_link = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Squirrel_posing.jpg/717px-Squirrel_posing.jpg'
cityscape_link = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/View_of_Empire_State_Building_from_Rockefeller_Center_New_York_City_dllu.jpg/798px-View_of_Empire_State_Building_from_Rockefeller_Center_New_York_City_dllu.jpg'
text_link = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Neon.JPG/799px-Neon.JPG'
art_link = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/757px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'
sax_link = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Alto_saxophone-E_1685-IMG_7092-gradient.jpg/600px-Alto_saxophone-E_1685-IMG_7092-gradient.jpg'
hand_link = 'https://upload.wikimedia.org/wikipedia/commons/5/56/Hand_female_3.jpg'

animal_img = load_img(animal_link)
cityscape_img = load_img(cityscape_link)
text_img = load_img(text_link)
art_img = load_img(art_link)
sax_img = load_img(sax_link)

scalar1 = 0.18215
vae_compress(animal_img, scalar1)
vae_compress(cityscape_img, scalar1)
vae_compress(text_img, scalar1)
vae_compress(art_img, scalar1)
vae_compress(sax_img, scalar1)

jpeg_compress(animal_img)
jpeg_compress(cityscape_img)
jpeg_compress(text_img)
jpeg_compress(art_img)
jpeg_compress(sax_img)

webp_compress(animal_img)
webp_compress(cityscape_img)
webp_compress(text_img)
webp_compress(art_img)
webp_compress(sax_img)





