from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import time
from typing import List
import os
from os import getcwd
import shutil
from dotenv import load_dotenv

from app.schemas.item_scheme import ItemScheme
from app.functions import divide_img, verify, merge_img
from app.functions.AES_cypher import cypher_image
from app.functions.AES_decrypt import decipher_image
from app.functions.AES_CIF_UNCIF import encrypt_image_aes_ofb, decrypt_image_aes_ofb
from app.functions.LSB import hide_img

router = APIRouter()

# Carpetas de archivos
imgFolder = "app/temp/img/"
imgCifFolder = "app/temp/imgCif/"

# Formatos válidos
imgFormats = (".png", ".jpg", ".bmp")
cifFormats = (".png", ".jpg", ".bmp", ".cif", ".aes")

# crear las instancias del objeto AES
clave = b"LlaveSecreta1234"  # la clave debe tener 16, 24 o 32 bytes de longitud
iv = b"VectorInicial123"  # el vector inicial debe tener 16 bytes de longitud


@router.post("/API/Encrypt/Image", tags=["Alfa", "Recive Imagen", "color"])
async def reciveImage(file: UploadFile = File(...)):
    try:
        if file.filename[-4:] in imgFormats:
            # Uno la ruta de imgFolder con el nombre del archivo menos la extensión
            file_folder = os.path.join(imgFolder, file.filename[:-4])
            # Creo la ruta final del archivo
            os.makedirs(file_folder, exist_ok=True)
            # Guardo el archivo dentro de la carpeta
            file_path = os.path.join(file_folder, file.filename)
            with open(file_path, "wb") as F:
                content = await file.read()
                F.write(content)
                F.close()
            res_divide = await divide_img.divide(file_path, file.filename)
            # Respondo un archivo con la dirección de guardado
            if res_divide["success"] == True:  # esto también debería ir en un try catch
                res_hide = await hide_img(
                    res_divide["img_yfile"],
                    res_divide["img_cbmin"],
                    res_divide["img_crmin"],
                )
                if res_hide["success"] == True:
                    res_merge = await merge_img.merge(
                        res_hide["y-hided"], file.filename
                    )
                    # Acá comienzo el cifrado
                    # res_cif = await cypher_image(clave, iv, file_path, file.filename)
                    res_decrypt = await encrypt_image_aes_ofb(file_path, key)
                    return FileResponse(res_decrypt)
                else:
                    return JSONResponse(
                        content={
                            "Error": res_hide["error"],
                        }
                    )
                res_uncif = await decipher_image(clave, iv, file_path, file.filename)
            else:
                return JSONResponse(
                    content={
                        "Error": res_divide["error"],
                    },
                    status_code=415,
                )
        else:
            return JSONResponse(
                content={"Error": "La extención del archivo no es válida"},
                status_code=415,
            )
    except:
        return JSONResponse(
            content={"Error": "Algo Falló con el archivo"}, status_code=200
        )


@router.post("/API/Decrypt/Image", tags=["Alfa", "Recive Imagen", "color"])
async def reciveImage(file: UploadFile = File(...)):
    try:
        if file.filename[-4:] in cifFormats:
            # Uno la ruta de imgFolder con el nombre del archivo menos la extensión
            file_folder = os.path.join(imgFolder, file.filename[:-4])
            # Creo la ruta final del archivo
            os.makedirs(file_folder, exist_ok=True)
            # Guardo el archivo dentro de la carpeta
            file_path = os.path.join(file_folder, file.filename)
            with open(file_path, "wb") as F:
                content = await file.read()
                F.write(content)
                F.close()
            res_divide = await divide_img.divide(file_path, file.filename)
            res_decrypt = await decrypt_image_aes_ofb(file_path, key)
            return FileResponse(res_decrypt)
            # Respondo un archivo con la dirección de guardado
            if res_divide["success"] == True:  # esto también debería ir en un try catch
                res_hide = await hide_img(
                    res_divide["img_yfile"],
                    res_divide["img_cbmin"],
                    res_divide["img_crmin"],
                )
                if res_hide["success"] == True:
                    res_merge = await merge_img.merge(
                        res_hide["y-hided"], file.filename
                    )
                    # Acá comienzo el cifrado
                    # res_cif = await cypher_image(clave, iv, file_path, file.filename)
                    res_decrypt = await encrypt_image_aes_ofb(file_path, key)
                    return FileResponse(res_decrypt)
                else:
                    return JSONResponse(
                        content={
                            "Error": res_hide["error"],
                        }
                    )
                res_uncif = await decipher_image(clave, iv, file_path, file.filename)
            else:
                return JSONResponse(
                    content={
                        "Error": res_divide["error"],
                    },
                    status_code=415,
                )
        else:
            return JSONResponse(
                content={"Error": "La extención del archivo no es válida"},
                status_code=415,
            )
    except:
        return JSONResponse(
            content={"Error": "Algo Falló con el archivo"}, status_code=200
        )


@router.post("/API/Hide/", tags=["Alfa", "Recive Imagen", "color"])
async def reciveImage(file: UploadFile = File(...)):
    if file.filename[-4:] in imgFormats:
        # Uno la ruta de imgFolder con el nombre del archivo menos la extensión
        file_folder = os.path.join(imgFolder, file.filename[:-4])
        # Creo la ruta final del archivo
        os.makedirs(file_folder, exist_ok=True)
        # Guardo el archivo dentro de la carpeta
        file_path = os.path.join(file_folder, file.filename)
        with open(file_path, "wb") as F:
            content = await file.read()
            F.write(content)
            F.close()
        res_divide = await divide_img.divide(file_path, file.filename)
        # Respondo un archivo con la dirección de guardado
        if res_divide["success"] == True:  # esto también debería ir en un try catch
            res_hide = await hide_img(
                res_divide["img_yfile"],
                res_divide["img_cbmin"],
                res_divide["img_crmin"],
            )
            if res_hide["success"] == True:
                res_merge = await merge_img.merge(res_hide["y-hided"], file.filename)
                return FileResponse(res_hide["y-hided"])
                # Acá comienzo el cifrado
                # res_cif = await cypher_image(clave, iv, file_path, file.filename)
            else:
                return JSONResponse(
                    content={
                        "Error": res_hide["error"],
                    }
                )
            res_uncif = await decipher_image(clave, iv, file_path, file.filename)
        else:
            return JSONResponse(
                content={
                    "Error": res_divide["error"],
                },
                status_code=415,
            )
    else:
        return JSONResponse(
            content={"Error": "La extención del archivo no es válida"}, status_code=415
        )


# este endpont es de desarrollo, aun no pidas cosas acá
@router.post("/API/Hide/dev", tags=["Recive Imagen", "color"])
async def reciveImage(file: UploadFile = File(...)):
    if file.filename[-4:] in imgFormats:
        # Uno la ruta de imgFolder con el nombre del archivo menos la extensión
        file_folder = os.path.join(imgFolder, file.filename[:-4])
        # Creo la ruta final del archivo
        os.makedirs(file_folder, exist_ok=True)
        # Guardo el archivo dentro de la carpeta
        file_path = os.path.join(file_folder, file.filename)
        with open(file_path, "wb") as F:
            content = await file.read()
            F.write(content)
            F.close()
        res_divide = await divide_img.divide(file_path, file.filename)
        # Respondo un archivo con la dirección de guardado
        if res_divide["success"] == True:  # esto también debería ir en un try catch
            res_hide = await hide_img(
                res_divide["img_yfile"],
                res_divide["img_cbmin"],
                res_divide["img_crmin"],
            )
            if res_hide["success"] == True:
                res_merge = await merge_img.merge(res_hide["y-hided"], file.filename)
                return FileResponse(res_hide["y-hided"])
                # Acá comienzo el cifrado
                # res_cif = await cypher_image(clave, iv, file_path, file.filename)
            else:
                return JSONResponse(
                    content={
                        "Error": res_hide["error"],
                    }
                )
            res_uncif = await decipher_image(clave, iv, file_path, file.filename)
        else:
            return JSONResponse(
                content={
                    "Error": res_divide["error"],
                },
                status_code=415,
            )
    else:
        return JSONResponse(
            content={"Error": "La extención del archivo no es válida"}, status_code=415
        )


@router.post("/API/PSNR/", tags=["Recive Imagen", "gray"])
async def reciveImage(file: UploadFile = File(...)):
    try:
        time.sleep(9)
        file_folder = os.path.join(imgFolder, "IMG2/IMG2.jpg")
        return FileResponse(file_folder)
    except:
        return JSONResponse(content={"Success": "Prueba lista"}, status_code=200)


@router.post("/API/Unhide/", tags=["Alfa", "Recive Imagen", "gray"])
async def decrypt_image(file: UploadFile = File(...)):
    try:
        if file.filename[-4:] in cifFormats:
            # Uno la ruta de imgCifFolder con el nombre del archivo menos la extensión
            file_folder = os.path.join(imgCifFolder, file.filename[:-4])
            # Creo la ruta final del archivo
            os.makedirs(file_folder, exist_ok=True)
            # Guardo el archivo dentro de la carpeta
            file_path = os.path.join(file_folder, file.filename)
            with open(file_path, "wb") as F:
                content = await file.read()
                F.write(content)
                F.close()
            res_merge = await merge_img.merge(file_path, file.filename)
            print(file_path)
            if res_merge["success"] == True:
                return FileResponse(res_merge["img"])
            else:
                return JSONResponse(
                    content={
                        "Error": res_merge["error"],
                    }
                )
        else:
            return JSONResponse(
                content={"Error": "la extención del archivo no es válida"},
                status_code=415,
            )
    except:
        return JSONResponse(
            content={
                "Error": "Hay un error desconocido con el archivo, intentelo de nuevo más tarde"
            },
            status_code=415,
        )


@router.post("/API/Unhide/dev", tags=["Recive Imagen", "gray"])
async def decrypt_image(file: UploadFile = File(...)):
    # try:
    if file.filename[-4:] in cifFormats:
        # Uno la ruta de imgCifFolder con el nombre del archivo menos la extensión
        file_folder = os.path.join(imgCifFolder, file.filename[:-4])
        # Creo la ruta final del archivo
        os.makedirs(file_folder, exist_ok=True)
        # Guardo el archivo dentro de la carpeta
        file_path = os.path.join(file_folder, file.filename)
        with open(file_path, "wb") as F:
            content = await file.read()
            F.write(content)
            F.close()
        res_merge = await merge_img.merge(file_path, file.filename)
        print(file_path)
        if res_merge["success"] == True:
            return FileResponse(res_merge["img"])
            # return {"Success": res_merge["success"]}
        else:
            return JSONResponse(
                content={
                    "Error": res_merge["error"],
                }
            )
    else:
        return JSONResponse(
            content={"Error": "la extención del archivo no es válida"},
            status_code=415,
        )


# except:
#     return JSONResponse(
#         content={
#             "Error": "Hay un error desconocido con el archivo, intentelo de nuevo más tarde"
#         },
#         status_code=415,
#     )
