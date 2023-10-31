import cv2
from skimage.metrics import structural_similarity as ssim

# Full Reference Image Quality Assessment (FRIQA)
original = cv2.imread('images\np_img.png')
decoded = cv2.imread('images\decoded_img.png')

# Convert images to grayscale (SSIM operates on grayscale images)
gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
gray_decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)

# Compute the SSIM score
ssim_score = ssim(gray_original, gray_decoded)

# Print the SSIM score
print(f"SSIM Score: {ssim_score}")