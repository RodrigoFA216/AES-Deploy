import cv2
import numpy as np
from fastapi.responses import JSONResponse
import os

from app.functions.eval import calculate_psnr


def complete_octet(bin_str):
    # Calcula la cantidad de ceros que se deben agregar
    num_zeros = 8 - len(bin_str)
    # Agrega los ceros a la izquierda del número binario
    return "0" * num_zeros + bin_str


async def merge(path, name):
    carpeta_img = path
    open_img = name
    img = cv2.imread(carpeta_img, cv2.IMREAD_UNCHANGED)
    # Height alto Width ancho
    properties = img.shape
    if len(properties) < 3:
        h = properties[0]
        w = properties[1]
        search_h = h // 4
        search_w = w // 4
        # Convierto la matriz en vector
        img_vector = img.flatten()
        bin_func = np.vectorize(  # combierte todo a binario
            lambda x: complete_octet(format(x, "b"))
        )
        int_func = np.vectorize(lambda x: int(x, 2))  # combierte todo a decmal
        # Convierto la imágen en binario
        img_vector_bin = bin_func(img_vector)
        # Extraigo la información de la imágen
        lect = ""
        for data in img_vector_bin:
            lect = lect + data[:-1]
        # Descarto la información a la derecha de la mitad de la información
        lect = lect[: (len(lect) // 2)]
        cb_and_cr_lenght = len(lect) // 2
        return {
            "success": True,
            "erorr": None,
        }
    elif len(properties) > 3:
        return {
            "success": False,
            "error": "El archivo tiene transparencia, esta imagen no ha sido cifrada",
        }
    elif len(properties) == 3:
        return {
            "success": False,
            "error": "El archivo tiene color, esta imagen no ha sido cifrada",
        }
