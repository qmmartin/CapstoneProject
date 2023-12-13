import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def ssim_iqa(og, new):
    # Full Reference Image Quality Assessment (FRIQA) using SSIM

    gray_original = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
    gray_decoded = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

    # Compute the SSIM score
    ssim_score = ssim(gray_original, gray_decoded)

    # Print the SSIM score
    print(f"SSIM Score: {ssim_score}")

def mse_iqa(og, new):
    # Full Reference Image Quality Assessment (FRIQA) using MSE

    gray_original = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
    gray_decoded = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

    # Compute the MSE
    mse = np.mean((gray_original - gray_decoded) ** 2)

    # Print the MSE
    print(f"MSE Score: {mse}")

def psnr_iqa(og, new):
    # Full Reference Image Quality Assessment (FRIQA) using PSNR

    gray_original = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
    gray_decoded = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

    # Compute the PSNR
    mse = np.mean((gray_original - gray_decoded) ** 2)
    if mse == 0:
        psnr_score = float('inf')  # PSNR is infinite for identical images
    else:
        max_pixel_value = 255.0
        psnr_score = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    # Print the PSNR score
    print(f"PSNR Score: {psnr_score}"+"\n")