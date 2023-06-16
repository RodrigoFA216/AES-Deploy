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
        return float("inf")
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# Completa la matriz con ceros
def completar_matriz(matriz):
    filas = len(matriz)
    columnas = len(matriz[0])
    if columnas % 16 == 0:
        return {"success": True, "error": None, "data": matriz, "missing": 0}
    columnas_faltantes = 16 - (columnas % 16)
    matriz_completa = []
    for fila in matriz:
        fila_completa = fila + [0] * columnas_faltantes
        matriz_completa.append(fila_completa)
    return {
        "success": True,
        "error": None,
        "data": matriz_completa,
        "missing": columnas_faltantes,
    }


def calculate_ssim(image_path1, image_path2):
    # Cargar las imágenes utilizando OpenCV
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    # Calcular el SSIM utilizando la función de OpenCV
    (score, diff) = cv2.compareSSIM(img1, img2, full=True)
    # Ajustar el rango de puntuación de -1 a 1 a 0 a 1
    score = (score + 1) / 2.0
    return score
