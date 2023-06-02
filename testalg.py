# vector = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

# print(vector[: (len(vector) // 2)])

from app.functions.eval import calculate_psnr
import cv2

img_hided = "C:/Users/ThinkPad/Documents/TITULACION_CEDILLO/AES-Deploy/app/temp/img/IMG30/IMG30-Y-hided-.jpg"
img_Y = "C:/Users/ThinkPad/Documents/TITULACION_CEDILLO/AES-Deploy/app/temp/img/IMG30/IMG30-Y-.jpg"
img_cb = "C:/Users/ThinkPad/Documents/TITULACION_CEDILLO/AES-Deploy/app/temp/img/IMG30/IMG30-Cb-.jpg"
img_cr = "C:/Users/ThinkPad/Documents/TITULACION_CEDILLO/AES-Deploy/app/temp/img/IMG30/IMG30-Cr-.jpg"


imgY = cv2.imread(img_Y, cv2.IMREAD_UNCHANGED)
imgCB = cv2.imread(img_cb, cv2.IMREAD_UNCHANGED)
imgCR = cv2.imread(img_cr, cv2.IMREAD_UNCHANGED)

imgYCBCR = cv2.merge([imgY, imgCB, imgCR])
cv2.imwrite(
    "C:/Users/ThinkPad/Documents/TITULACION_CEDILLO/AES-Deploy/app/temp/img/IMG30/IMG30-m.jpg",
    imgYCBCR,
)
imgRGB = cv2.cvtColor(imgYCBCR, cv2.COLOR_YCrCb2BGR)
cv2.imwrite(
    "C:/Users/ThinkPad/Documents/TITULACION_CEDILLO/AES-Deploy/app/temp/img/IMG30/IMG30-n.jpg",
    imgRGB,
)


result = calculate_psnr(
    "C:/Users/ThinkPad/Documents/TITULACION_CEDILLO/AES-Deploy/app/temp/img/IMG30/IMG30-n.jpg",
    "C:/Users/ThinkPad/Documents/TITULACION_CEDILLO/AES-Deploy/app/temp/img/IMG30/IMG30.jpg",
)
print(result)

# ----------------------------------------------------------------
# Reemplaza los valores del array 2 en el vector arr1
# ----------------------------------------------------------------
# arr1 = ["100", "101", "101", "100", "100"]
# arr2 = "1001"

# for indice, elemento in enumerate(arr2):
#     aux = arr1[indice]
#     aux = aux[:-1] + elemento
#     arr1[indice] = aux

# print(arr1)

# ----------------------------------------------------------------
# Recorre la matriz de atras hacia adelante
# ----------------------------------------------------------------
# matriz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# filas = len(matriz)
# columnas = len(matriz[0])

# for i in range(filas - 1, -1, -1):
#     for j in range(columnas - 1, -1, -1):
#         print(matriz[i][j])

# ----------------------------------------------------------------
# # Vuelve bits la cadena propuesta
# ----------------------------------------------------------------
# cadena = "(!+[]+[]+![])"
# bits = ""
# for caracter in cadena:
#     bits += format(ord(caracter), "08b")
# print(len(bits), bits)


# ----------------------------------------------------------------
# Hace el histograma de las imágenes
# ----------------------------------------------------------------
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

# ----------------------------------------------------------------
# Completa la matriz con ceros / hay que poner un try catch
# ----------------------------------------------------------------
# import numpy as np


# def completar_matriz(matriz):
#     filas, columnas = matriz.shape
#     if columnas % 16 == 0:
#         return matriz
#     columnas_faltantes = 16 - (columnas % 16)
#     matriz_completa = np.concatenate(
#         (matriz, np.zeros((filas, columnas_faltantes))), axis=1
#     )
#     return matriz_completa


# def complete_octet(bin_str):
#     # Calcula la cantidad de ceros que se deben agregar
#     num_zeros = 8 - len(bin_str)
#     # Agrega los ceros a la izquierda del número binario
#     return "0" * num_zeros + bin_str


# # Función que convierte todo a binario
# bin_func = np.vectorize(lambda x: complete_octet(format(x, "b")))
# int_func = np.vectorize(lambda x: int(x, 2))  # Función que combierte todo a decimal

# my_array = np.array([[1, 2], [3, 4]])
# mi_matriz = np.array(
#     [
#         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
#         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
#     ]
# )
# matriz_mult = completar_matriz(mi_matriz)
# # matriz_completa = bin_func(matriz_mult)
# print(complete_octet((format(10, "b"))))

# ----------------------------------------------------------------
# Obtiene el indice máximo y mínimo de cada matriz
# ----------------------------------------------------------------
# import numpy as np
# import cv2


# def obtener_indices_maximo_minimo(image1):
#     gray_image1 = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
#     filas, columnas = gray_image1.shape
#     indices_max_min = np.zeros((filas, 2), dtype=int)
#     indice_actual = 0
#     for i in range(filas):
#         fila = gray_image1[i]
#         indice_maximo = indice_actual + np.argmax(fila)
#         indice_minimo = indice_actual + np.argmin(fila)
#         indices_max_min[i] = [indice_maximo, indice_minimo]
#         indice_actual += columnas
#     return indices_max_min


# # Ejemplo de uso
# # mi_matriz = np.array([[1, 9, 5, 1], [1, 8, 5, 1], [8, 1, 5, 1], [8, 5, 1, 1]])
# mi_matriz = "C:/Users/ThinkPad/Documents/TITULACION_CEDILLO/AES-Deploy/app/temp/img/IMG30/IMG30-Y-.jpg"
# indices_max_min = obtener_indices_maximo_minimo(mi_matriz)
# for i in range(5, 10):
#     print(indices_max_min[i])


# ----------------------------------------------------------------
# transforma string a un vector de numpy
# ----------------------------------------------------------------
# import numpy as np

# mi_string = "abcdefghijklmnop"

# longitud_total = len(mi_string)
# cantidad_grupos = longitud_total // 8
# hay_incompleto = longitud_total % 8 != 0

# grupos = np.array([mi_string[i : i + 8] for i in range(0, longitud_total, 8)])

# print(grupos, cantidad_grupos, hay_incompleto)
