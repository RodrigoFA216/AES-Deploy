import argparse
import numpy as np
import cv2


async def calculate_psnr(image1_path, image2_path):
    # Cargar imágenes
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    # Calcular PSNR
    mse = ((img1.astype(float) - img2.astype(float)) ** 2).mean()
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr




# import cv2
# import numpy as np

# def calculate_psnr(image1, image2):
#     # Convert images to numpy arrays
#     image1 = np.array(image1)
#     image2 = np.array(image2)

#     # Calculate the mean square error between the two images
#     mse = np.mean((image1 - image2) ** 2)

#     # Handle the case where the mean square error is zero (the images are identical)
#     if mse == 0:
#         return float('inf')

#     # Calculate the PSNR
#     max_pixel_value = 255.0
#     psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

#     return psnr