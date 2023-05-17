# Reemplaza los valores del array 2 en el vector arr1
# arr1 = ["100", "101", "101", "100", "100"]
# arr2 = "1001"

# for indice, elemento in enumerate(arr2):
#     aux = arr1[indice]
#     aux = aux[:-1] + elemento
#     arr1[indice] = aux

# print(arr1)

# Recorre la matriz de atras hacia adelante
# matriz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# filas = len(matriz)
# columnas = len(matriz[0])

# for i in range(filas - 1, -1, -1):
#     for j in range(columnas - 1, -1, -1):
#         print(matriz[i][j])

# Vuelve bits la cadena propuesta
# cadena = "(!+[]+[]+![])"
# bits = ""
# for caracter in cadena:
#     bits += format(ord(caracter), "08b")
# print(len(bits), bits)


# vector = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

# print(vector[: (len(vector) // 2)])

# from app.functions.eval import calculate_psnr

# img_hided = 'C:/Users/ThinkPad/Documents/TITULACION_CEDILLO/AES-Deploy/app/temp/img/IMG30/IMG30-Y-hided-.jpg'
# img_Y = 'C:/Users/ThinkPad/Documents/TITULACION_CEDILLO/AES-Deploy/app/temp/img/IMG30/IMG30-Y-.jpg'

# result = calculate_psnr( img_Y, img_hided)

# print(result)

# Hace el histograma de las imágenes
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img_hided = 'C:/Users/ThinkPad/Documents/TITULACION_CEDILLO/AES-Deploy/app/temp/img/IMG30/IMG30-Y-hided-.jpg'
# img_Y = 'C:/Users/ThinkPad/Documents/TITULACION_CEDILLO/AES-Deploy/app/temp/img/IMG30/IMG30-Y-.jpg'

# th = 0

# def compare_images(image1, image2, threshold):
#     # Convertir las imágenes a escala de grises
#     gray_image1 = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
#     gray_image2 = cv2.imread(image2, cv2.IMREAD_UNCHANGED)

#     # Calcular los histogramas de cada imagen
#     hist_image1 = cv2.calcHist([gray_image1], [0], None, [256], [0, 256])
#     hist_image2 = cv2.calcHist([gray_image2], [0], None, [256], [0, 256])

#     plt.plot(hist_image1, color = 'b')
#     plt.plot(hist_image2, color = 'r')
#     plt.xlim([0,256])

#     plt.show()

#     # closing all open windows
#     cv2.destroyAllWindows()

#     # Comparar los histogramas usando la distancia euclidiana
#     distance = cv2.norm(hist_image1, hist_image2, cv2.NORM_L2)

#     # Comprobar si la distancia está por debajo del umbral
#     if distance < threshold:
#         return {
#             "result": True,
#             "distance": distance
#         }
#     else:
#         return {
#             "result": False,
#             "distance": distance
#         }

# res = compare_images(img_hided, img_Y, th)

# print(res["result"], res["distance"])


# Completa la matriz con ceros

import numpy as np


def completar_matriz(matriz):
    filas, columnas = matriz.shape
    if columnas % 16 == 0:
        return {"success": True, "error": None, "data": matriz, "missing": 0}
    columnas_faltantes = 16 - (columnas % 16)
    matriz_completa = np.concatenate(
        (matriz, np.zeros((filas, columnas_faltantes))), axis=1
    )
    return {
        "success": True,
        "error": None,
        "data": matriz_completa,
        "missing": columnas_faltantes,
    }

    return matriz_completa


# Ejemplo de uso
# mi_matriz = [
#     [
#         1,
#         2,
#         3,
#         4,
#         5,
#         6,
#         7,
#         8,
#         9,
#         10,
#         11,
#         12,
#         13,
#         14,
#     ],
#     [
#         1,
#         2,
#         3,
#         4,
#         5,
#         6,
#         7,
#         8,
#         9,
#         10,
#         11,
#         12,
#         13,
#         14,
#     ],
# ]

my_array = np.array([[1, 2], [3, 4]])
mi_matriz = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
        ],
    ]
)
matriz_completa = completar_matriz(mi_matriz)
print(matriz_completa)
