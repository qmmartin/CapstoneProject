# Capstone Science Fair Project Logbook

## 9-25-2023

The project goal for today was to start implementing the VAE code while using the actual huggingface site as little as possible. 

Installed essential libraries Pytorch, Diffusers, and Transformers to python environment.

Wrote code that initializes a prompt and generates an image using it, although i met an error in that the model cannot be found.

Changed README to include more information such as the project's ultimate final goal.

## 10-2-2023

[link to test code](https://towardsdatascience.com/stable-diffusion-using-hugging-face-501d8dbdd8)

[from_pretrained help](https://huggingface.co/docs/diffusers/using-diffusers/loading)

Worked on new code to utilize model following hugging face's official from_pretrained guide. Previous code was not working due to no model found. Hugging Face's online API seems to be the best (and likely only) way to utilize the model although using it through python code has proved very difficult. I think it will be my main path forward.

The following is an image of the error I am currently recieving: 

<img src="Images\model_nf.PNG" width="800">

I believe this error is coming from my avoidance of the hugging face API for online Stable Diffusion models. It seems I have no other choice than to access the repositories held on the site instead of using a local model.

## 10-12-2023 

Created code that would log into huggingface to pull the models from online so this program can be run remotely.

Successfully initialized the model and prompt, but ran into an error when trying to pass the prompt through the model.

Added code that can tokenize a prompt and print those tokens with their english counterparts.

Installed libraries fastdownload matplotlib using pip

Added code that can perform VAE compression on an image and it seemed to work, at least in the backend side of things, despite some runtime warnings. There is currently no way to actually view this image as it is not compatible with cv2 in its current form so I will need to add some method of viewing the image. Following is the response the code gives when ran, showing that the program is working in theory:

<img src="Images\functional.PNG" width="800">

## 10-16-2023

Separated the code into its different functionalities, specifically made a new file for the token test code because it is not used in vae_compression.py but may be needed later down the line.

Attempted creating code that would show the image, and it did show an image, but I believe the pixels are being altered in some way as the result is just greyscale squares.

## 10-24-2023

Worked on code that would successfully display the compressed images and it works!

Compressed latents:

<img src="Images\vae_success.PNG" width="800">

## 10-26-2023 

Created code in vae_compression that would decompress the latent representation using Stable Diffusion 1.5's VAE.

It works but the images are slightly off as the variation from the AI alters the image slightly.

<img src="Images\comparison_img.PNG" width="800">

As you can probably tell in this photo, the squirrels look very similar, but they are slightly different, some notable differences include the glimmer on the eye, as well as many of the hairs on the tail and ears.

## 10-30-2023
My current plan is to try to change the rate at which the image is compressed and see if the compression rate alters the output image.

The method I am using to test the quality is an FR-IQA or a Full Reference Image Quality Assessment which compares 2 images by feeding it a distorted image and the original image. More specifically, I am using ssim from scikit-image.

Installed scikit-image to perform friqa on images

TO DO LIST
- Modularize code
- Image compression metrics

After Modularizing vae_compression.py into a new file vae_module.py, the code ultimately works nearly the same, however the output latent_rep has the black and white values inverted for some reason and I will have to fix this bug.

## 10-31-2023
I fixed the bug where the latent_rep image was inverted. Modularized friqa_module successfully.


Test Images: 

https://commons.wikimedia.org/wiki/File:Squirrel_posing.jpg

TO DO LIST 
- Image Compression Metrics
- Alter Compression Rate of Image


## 11-2-2023
Made the code in main.py repeatable by turning it into a function within vae_module.py. Now the code can be run multiple times with different images easily.

## 11-7-2023
Used the newly made function to perform compression on multiple images of different types.

#### Image Sources:
[animal_link](https://upload.wikimedia.org/wikipedia/commons/1/1c/Squirrel_posing.jpg), [cityscape_link](https://upload.wikimedia.org/wikipedia/commons/0/05/View_of_Empire_State_Building_from_Rockefeller_Center_New_York_City_dllu.jpg), [text_link](https://upload.wikimedia.org/wikipedia/commons/d/df/Neon.JPG), [art_link](https://upload.wikimedia.org/wikipedia/commons/a/aa/Van_Gogh_-_Starry_Night_2.jpg), [sax_link](https://upload.wikimedia.org/wikipedia/commons/e/e6/Alto_saxophone-E_1685-IMG_7092-gradient.jpg)


## 11-9-2023
Worked on procedures and created graphics of the VAE process for use in my procedures.

## 11-13-2023
Worked on procedures and researched new methods of FR-IQA, namely Strutural Similarity Index (SSI), Peak Signal-to-Noise-Ratio (PSNR), and Mean Squared Error (MSE).

## 11-14-2023
Finished rough draft of procedures and fine-tuned graphics of different functions of the code.

## 11-16-2023
Created functions to help work with different types of images that were previously unusable in the code, namely PNG and JPEG.

Altered compress_and_save() to be able to use images instead of links that way numpy arrays can be used instead of strictly downloaded PILs.

Currently attempting to remove all resizing from the code to be able to work with any images I would like to but pil_to_latents or latents_to_np appears to be changing the size of the image.

## 11-27-2023
Began working on intro to my paper. Researched image compression and it's importance as well as the basics of VAEs.

## 11-28-2023
Continued work on the introduction to my paper. Continued researching image compression as well as VAEs.

## 11-30-2023
Continued work on introduction to paper. Looked into possible avenues of altering compression quality/ratio for a human testing application on perceived quality vs objective quality.

## 12-5-2023
Continued work on introduction. Found an equation P(X)=∫P(X∣z;θ)P(z)dz that represents the fundamental concept of probabilistic generative modeling, or the back-end of VAEs. If I can get to this equation I can alter theta which would change the compression quality or ratio in some way.

## 12-7-2023
Continued work on introduction. 

## 12-11-2023
Changed code to be more seamless and work better with different image types than PIL. Also added back output images that were removed from prior iterations of the files on accident.
Now I plan to make "scalar" be a parameter of compress_and_save that alters all existences of "scale_factor."
Successfully converted code to be able to handle different scalars, however the image quality seems to drop depending on how far away the scalar is from the original scalar or 0.18215. Because of this, I will need to find another way to alter image quality or compression rate.

Added new functions to friqa_module.py that will perform MSE and PSNR FR-IQA on images respectively. Also moved those functions to vae_module.py for ease of access.