import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

ssim_scores = []
mse_scores = []
psnr_scores = []

def ssim_iqa(og, new):
    # Full Reference Image Quality Assessment (FRIQA) using SSIM
    global ssim_ran

    gray_original = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
    gray_decoded = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

    ssim_score = ssim(gray_original, gray_decoded)

    print(f"SSIM Score: {ssim_score}")

    ssim_scores.append(ssim_score)



def mse_iqa(og, new):
    # Full Reference Image Quality Assessment (FRIQA) using MSE

    gray_original = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
    gray_decoded = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

    mse = np.mean((gray_original - gray_decoded) ** 2)

    print(f"MSE Score: {mse}")

    mse_scores.append(mse)

def psnr_iqa(og, new):
    # Full Reference Image Quality Assessment (FRIQA) using PSNR

    gray_original = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
    gray_decoded = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

    mse = np.mean((gray_original - gray_decoded) ** 2)
    if mse == 0:
        psnr_score = float('inf')  # PSNR is infinite for identical images
    else:
        max_pixel_value = 255.0
        psnr_score = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    print(f"PSNR Score: {psnr_score}"+"\n")

    psnr_scores.append(psnr_score)


def record_data():
    with pd.ExcelWriter("friqa_data.xlsx", engine='openpyxl') as writer:
        try:
            pd.DataFrame({"SSIM Score": ssim_scores}).to_excel(writer, sheet_name='SSIM', index=False)
        except ValueError:
            pd.DataFrame({"SSIM Score": ssim_scores}).to_excel(writer, sheet_name='SSIM', index=False, header=False)

        try:
            pd.DataFrame({"MSE Score": mse_scores}).to_excel(writer, sheet_name='MSE', index=False)
        except ValueError:
            pd.DataFrame({"MSE Score": mse_scores}).to_excel(writer, sheet_name='MSE', index=False, header=False)

        try:
            pd.DataFrame({"PSNR Score": psnr_scores}).to_excel(writer, sheet_name='PSNR', index=False)
        except ValueError:
            pd.DataFrame({"PSNR Score": psnr_scores}).to_excel(writer, sheet_name='PSNR', index=False, header=False)