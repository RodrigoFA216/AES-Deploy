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
    print(carpeta_img)
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
        mi_string = "abcdefghijklmnop"
        longitud_total = len(mi_string)
        cantidad_grupos = longitud_total // 8
        hay_incompleto = longitud_total % 8 != 0
        grupos = np.array([lect[i : i + 8] for i in range(0, longitud_total, 8)])
        return {"success": True, "error": None, "value": grupos}
        if hay_incompleto:
            return {"success": False, "error": None, "value": grupos}
        else:
            return {"success": True, "error": None, "value": grupos}
        print(lect)
        # Descarto la información a la derecha de la mitad de la información
        lect = lect[: (len(lect) // 2)]
        # Separo la primer componente
        comp_1 = lect[: (len(lect) // 2)]
        # Separo la segunda componente
        comp_2 = lect[len(comp_1) :]
        # Los hago matrices
        comp_1_reshaped = np.reshape(comp_1, (h // 4, w // 4))
        comp_2_reshaped = np.reshape(comp_2, (h // 4, w // 4))
        # Redimensionar las componentes de color
        cb_redim = cv2.resize(comp_1_reshaped, (w, h), interpolation=cv2.INTER_CUBIC)
        cr_redim = cv2.resize(comp_2_reshaped, (w, h), interpolation=cv2.INTER_CUBIC)
        # Aplicar GaussianBlur
        cb_redim_smooth = cv2.GaussianBlur(cb_redim, (3, 3), 0)
        cr_redim_smooth = cv2.GaussianBlur(cr_redim, (3, 3), 0)
        cb_h, cb_w = cb_redim_smooth.shape
        cr_h, cr_w = cr_redim_smooth.shape
        if cb_h == h & cr_h == h:
            if cb_w == w & cr_w == w:
                image_merged = cv2.merge([img, cb_redim_smooth, cr_redim_smooth])
                img_merg = f"{carpeta_img[:-4]}-M-{open_img[-4:]}"
                cv2.imwrite(carpeta_img[:-4] + "-M-" + open_img[-4:], image_merged)
                return {"success": True, "error": None, "img": img_merg}
            else:
                return {"success": False, "error": "With not match"}
        else:
            return {"success": False, "error": "Height not match"}
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
