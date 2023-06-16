from Crypto.Cipher import AES
from Crypto.Util import Counter
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import os


async def encrypt_image_aes_ofb(image_path, key):
    cipher = AES.new(key, AES.MODE_OFB)
    with open(image_path, "rb") as file:
        plaintext = file.read()
    ciphertext = cipher.encrypt(plaintext)
    encrypted_image_path = (
        os.path.splitext(image_path)[0] + "_encrypted" + os.path.splitext(image_path)[1]
    )
    with open(encrypted_image_path, "wb") as file:
        file.write(ciphertext)
    return encrypted_image_path
    print("Imagen cifrada correctamente. Ruta: ", encrypted_image_path)


async def decrypt_image_aes_ofb(encrypted_image_path, key):
    cipher = AES.new(key, AES.MODE_OFB)
    with open(encrypted_image_path, "rb") as file:
        ciphertext = file.read()
    decrypted_image = cipher.decrypt(ciphertext)
    decrypted_image_path = (
        os.path.splitext(image_path)[0] + "_decrypted" + os.path.splitext(image_path)[1]
    )
    with open(decrypted_image_path, "wb") as file:
        file.write(decrypted_image)
    return decrypted_image_path
    print("Imagen descifrada correctamente. Ruta: ", decrypted_image_path)
