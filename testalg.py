# arr1 = ["100", "101", "101", "100", "100"]
# arr2 = "1001"

# for indice, elemento in enumerate(arr2):
#     aux = arr1[indice]
#     aux = aux[:-1] + elemento
#     arr1[indice] = aux

# print(arr1)


# matriz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# filas = len(matriz)
# columnas = len(matriz[0])

# for i in range(filas - 1, -1, -1):
#     for j in range(columnas - 1, -1, -1):
#         print(matriz[i][j])


# cadena = "(!+[]+[]+![])"
# bits = ""
# for caracter in cadena:
#     bits += format(ord(caracter), "08b")
# print(bits)


vector = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

print(vector[: (len(vector) // 2)])
