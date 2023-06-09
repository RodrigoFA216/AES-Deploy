import cv2
import numpy as np

# función que agregará ceros a la izquierda para completar el octeto


def complete_octet(bin_str):
    # Calcula la cantidad de ceros que se deben agregar
    num_zeros = 8 - len(bin_str)
    # Agrega los ceros a la izquierda del número binario
    return "0" * num_zeros + bin_str


async def hide_img(path_y, path_a, path_b):
    Y = cv2.imread(path_y, cv2.IMREAD_UNCHANGED)
    A = cv2.imread(path_a, cv2.IMREAD_UNCHANGED)
    B = cv2.imread(path_b, cv2.IMREAD_UNCHANGED)
    # Height alto Width ancho
    # obtengo las caracteríasticas de las imágenes
    h, w = Y.shape
    ah, aw = A.shape
    bh, bw = B.shape
    info_port = (h * w) * 8
    info_required = ((ah * aw) * 8) + ((bh * bw) * 8)
    if info_required >= info_port:
        return {
            "success": False,
            "error": "La información de ocultamiento es mayor o igual a la información portadora: No se puede ocultar",
        }
    else:
        # Hago las matrices un vector
        Y_vector = Y.ravel()
        A_vector = A.ravel()
        B_vector = B.ravel()
        bin_func = np.vectorize(  # Función que convierte todo a binario
            lambda x: complete_octet(format(x, "b"))
        )
        int_func = np.vectorize(  # Función que combierte todo a decmal
            lambda x: int(x, 2)
        )
        # Ocupo las funciones para poder convertir los vectores en binario con la regla del octeto
        Y_vec_bin = bin_func(Y_vector)
        A_vec_bin = bin_func(A_vector)
        B_vec_bin = bin_func(B_vector)
        print("something", B_vector[-1])
        # Concateno los vectores en un string de información
        A_vec_bin_full = "".join(A_vec_bin)
        B_vec_bin_full = "".join(B_vec_bin)
        info_hide = A_vec_bin_full + B_vec_bin_full
        # Debug, imprimo las características
        print("height: ", h, h % 16, " width: ", w, w % 16)
        print("height: ", ah, ah % 16, " width: ", aw, aw % 16)
        print("height: ", bh, bh % 16, " width: ", bw, bw % 16)
        # Inicio el ocultamiento de la información
        for indice, elemento in enumerate(Y_vec_bin):
            Y_vec_bin[indice] = elemento[:-1] + info_hide[indice]
        # Convierto la información ocultada en decimal
        Y_vec_bin_hided_int = int_func(Y_vec_bin)
        # Regreso la info a sus dimenciones de matriz
        Y_vec_bin_hided_int_reshaped = np.reshape(Y_vec_bin_hided_int, (h, w))
        # Guardo la imágen y regreso los valores
        cv2.imwrite(path_y[:-7] + "-Y-h-" + path_y[-4:], Y_vec_bin_hided_int_reshaped)
        img_y_hided = f"{path_y[:-7]}-Y-h-{path_y[-4:]}"
        return {
            "success": True,
            "error": None,
            "y-hided": img_y_hided,
            "data": Y_vec_bin_hided_int_reshaped,
        }
