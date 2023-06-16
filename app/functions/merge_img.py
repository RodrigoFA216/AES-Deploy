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
        print(h, w, search_h, search_w)
        # Convierto la matriz en vector
        img_vector = img.ravel()
        bin_func = np.vectorize(  # combierte todo a binario
            lambda x: complete_octet(format(x, "b"))
        )
        int_func = np.vectorize(lambda x: int(x, 2))  # combierte todo a decmal
        # Convierto la imágen en binario
        img_vector_bin = bin_func(img_vector)
        # Extraigo la información de la imágen
        print(type(img_vector_bin))
        ultimo_caracter = ""
        for string in img_vector_bin:
            ultimo_caracter += string[-1]
        # ultimo_caracter = [s[-1] for s in img_vector_bin]
        # print(len(ultimo_caracter))
        # ultimo_caracter = ultimo_caracter[: len(ultimo_caracter) // 2]
        print("Extracción: ", len(ultimo_caracter))
        resultado = "".join(ultimo_caracter)
        mi_string = resultado
        longitud_total = len(mi_string)
        cantidad_grupos = longitud_total // 8
        hay_incompleto = longitud_total % 8 != 0
        grupos = np.array([mi_string[i : i + 8] for i in range(0, longitud_total, 8)])
        print("Longitud grupos: ", len(grupos))
        print("Muestra: ", grupos[-1])
        # return {"success": True, "error": None}
        if hay_incompleto:
            return {
                "success": False,
                "error": "Hubo algún problema al extraer la información de color en la imágen dada",
            }
        else:
            info_dec = int_func(grupos)
            primer_componente = info_dec[: len(info_dec) // 2]
            segundo_componente = info_dec[len(info_dec) // 2 :]
            # primer_componente_dec = int_func(primer_componente)
            # segundo_componente_dec = int_func(segundo_componente)
            print("Longitud componente: ", len(primer_componente))
            print("Muestra: ", primer_componente[-1])
            # print("muestra 3", primer_componente_dec)
            comp_1_reshaped = np.reshape(primer_componente, (search_h, search_w))
            comp_2_reshaped = np.reshape(segundo_componente, (search_h, search_w))
            comp_1_reshaped_u = comp_1_reshaped.astype(np.uint8)
            comp_2_reshaped_u = comp_2_reshaped.astype(np.uint8)
            # Redimensionar las componentes de color
            cb_redim = cv2.resize(
                comp_1_reshaped_u, (w, h), interpolation=cv2.INTER_CUBIC
            )
            cr_redim = cv2.resize(
                comp_2_reshaped_u, (w, h), interpolation=cv2.INTER_CUBIC
            )
            # Aplicar GaussianBlur
            cb_redim_smooth = cv2.GaussianBlur(cb_redim, (3, 3), 0)
            cr_redim_smooth = cv2.GaussianBlur(cr_redim, (3, 3), 0)
            comp_image = cv2.merge(
                [
                    np.zeros_like(comp_1_reshaped_u),
                    comp_1_reshaped_u,
                    np.zeros_like(comp_1_reshaped_u),
                ]
            )
            comp_image = cv2.cvtColor(comp_image, cv2.COLOR_BGR2YCrCb)
            cb_image = cv2.merge(
                [
                    np.zeros_like(cr_redim_smooth),
                    np.zeros_like(cr_redim_smooth),
                    cb_redim_smooth,
                ]
            )
            cb_image = cv2.cvtColor(cb_image, cv2.COLOR_BGR2YCrCb)
            cv2.imwrite(carpeta_img[:-4] + "-T-" + open_img[-4:], cb_image)
            cv2.imwrite(carpeta_img[:-4] + "-F-" + open_img[-4:], comp_image)
            image_merged = cv2.merge([img, cb_redim_smooth, cr_redim_smooth])
            image_merged_rgb = cv2.cvtColor(image_merged, cv2.COLOR_YCrCb2BGR)
            img_merg = f"{carpeta_img[:-4]}-M-{open_img[-4:]}"
            cv2.imwrite(carpeta_img[:-4] + "-M-" + open_img[-4:], image_merged_rgb)
            return {"success": True, "error": None, "img": img_merg}
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

        # print(lect)
        # # Descarto la información a la derecha de la mitad de la información
        # lect = lect[: (len(lect) // 2)]
        # # Separo la primer componente
        # comp_1 = lect[: (len(lect) // 2)]
        # # Separo la segunda componente
        # comp_2 = lect[len(comp_1) :]
        # # Los hago matrices
        # comp_1_reshaped = np.reshape(comp_1, (h // 4, w // 4))
        # comp_2_reshaped = np.reshape(comp_2, (h // 4, w // 4))
        # # Redimensionar las componentes de color
        # cb_redim = cv2.resize(comp_1_reshaped, (w, h), interpolation=cv2.INTER_CUBIC)
        # cr_redim = cv2.resize(comp_2_reshaped, (w, h), interpolation=cv2.INTER_CUBIC)
        # # Aplicar GaussianBlur
        # cb_redim_smooth = cv2.GaussianBlur(cb_redim, (3, 3), 0)
        # cr_redim_smooth = cv2.GaussianBlur(cr_redim, (3, 3), 0)
        # cb_h, cb_w = cb_redim_smooth.shape
        # cr_h, cr_w = cr_redim_smooth.shape
        # if cb_h == h & cr_h == h:
        #     if cb_w == w & cr_w == w:
        #         image_merged = cv2.merge([img, cb_redim_smooth, cr_redim_smooth])
        #         img_merg = f"{carpeta_img[:-4]}-M-{open_img[-4:]}"
        #         cv2.imwrite(carpeta_img[:-4] + "-M-" + open_img[-4:], image_merged)
        #         return {"success": True, "error": None, "img": img_merg}
        #     else:
        #         return {"success": False, "error": "With not match"}
        # else:
        #     return {"success": False, "error": "Height not match"}
