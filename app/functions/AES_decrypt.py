import cv2
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

# # cargar la imagen cifrada
# with open('app/temp/img/Atmosfera/Atmosfera-cif.jpg', 'rb') as f:
#     img_cifrada = f.read()

# # crear una instancia del objeto AES
# clave = b'LlaveSecreta1234'  # la clave debe tener 16, 24 o 32 bytes de longitud
# iv = b'VectorInicial123'  # el vector inicial debe tener 16 bytes de longitud
# aes = AES.new(clave, AES.MODE_OFB, iv)

# # aplicar el descifrado
# img_descifrada = unpad(aes.decrypt(img_cifrada), AES.block_size)

# # convertir la imagen descifrada a su formato original
# img_descifrada = cv2.imdecode(
#     numpy.frombuffer(img_descifrada, numpy.uint8),
#     cv2.IMREAD_GRAYSCALE
# )

# # guardar la imagen descifrada
# cv2.imwrite('app/temp/img/Atmosfera/Atmosfera-uncif.jpg', img_descifrada)


# def decrypt_image(key, vector, path, name):
#     img_cifrada = cv2.imread(path)
#     cipher = AES.new(key, AES.MODE_OFB, iv)
#     img_descifrada = cipher.decrypt(img_cifrada)
#     img_np = np.frombuffer(img_descifrada, np.uint8)
#     img_np = img_np.reshape((img_cifrada.shape[0], img_cifrada.shape[1], -1))
#     cv2.imshow("Imagen descifrada", img_np)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# with open(path, 'rb') as F:
#     img_cifrada = F.read()
# aes = AES.new(key, AES.MODE_OFB, vector)
# img_descifrada = unpad(aes.decrypt(img_cifrada), AES.block_size)
# cv2.imshow('Imagen descifrada', img_descifrada.astype('uint8'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# img_descifrada = cv2.imdecode(
#     numpy.frombuffer(img_descifrada, numpy.uint8),
#     cv2.IMREAD_GRAYSCALE
# )
# cv2.imwrite(path[:-len(name)]+name[:-4]+'-decif'+name[-4:], img_descifrada)
# string = f'{path[:-len(name)]}{name[:-4]}-decif{name[-4:]}'
# return string


async def decipher_image(key, vector, path, name):
    with open(path, "rb") as f:
        img_cifrada = f.read()
    aes = AES.new(key, AES.MODE_OFB, vector)
    # Desencriptar la imagen
    img_desencriptada = aes.decrypt(img_cifrada)
    # Eliminar el relleno
    # img_desencriptada = unpad(img_desencriptada, AES.block_size)
    # Crear la imagen a partir de los bytes desencriptados
    img = cv2.imdecode(
        np.frombuffer(img_desencriptada, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    # Guardar la imagen descifrada
    # with open(path[:-len(name)]+name[:-4]+'-des'+name[-4:], 'wb') as f: # tiene un error
    #     f.write(img) #no puede escribir un nonetype
    print(type(img))
    string = f"{path[:-len(name)]}{name[:-4]}-des{name[-4:]}"
    return string
