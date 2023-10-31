import cv2
from skimage.metrics import structural_similarity as ssim


def ssim_iqa(og, new):
    # Full Reference Image Quality Assessment (FRIQA)

    # Convert images to grayscale (SSIM operates on grayscale images)
    gray_original = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
    gray_decoded = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

    # Compute the SSIM score
    ssim_score = ssim(gray_original, gray_decoded)

    # Print the SSIM score
    print(f"SSIM Score: {ssim_score}")