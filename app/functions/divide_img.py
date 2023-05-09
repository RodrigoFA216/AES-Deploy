import cv2
import numpy as np
from fastapi.responses import JSONResponse
import os

from app.functions.eval import calculate_psnr


async def divide(path, name):
    carpeta_img = path
    open_img = name
    img = cv2.imread(carpeta_img, cv2.IMREAD_UNCHANGED)
    properties = img.shape
    if len(properties) == 3:
        if properties[2] == 3:
            resize_height = properties[0] // 4
            resize_width = properties[1] // 4
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y, cb, cr = cv2.split(img_yuv)
            ycb = cv2.merge([y, cb, np.zeros_like(cb)])
            ycr = cv2.merge([y, np.zeros_like(cr), cr])
            cv2.imwrite(carpeta_img[:-4] + "-Y-" + open_img[-4:], y)
            cv2.imwrite(carpeta_img[:-4] + "-Cb-" + open_img[-4:], cb)
            cv2.imwrite(carpeta_img[:-4] + "-Cr-" + open_img[-4:], cr)
            cv2.imwrite(carpeta_img[:-4] + "-YCb-" + open_img[-4:], ycb)
            cv2.imwrite(carpeta_img[:-4] + "-YCr-" + open_img[-4:], ycr)
            img_y = f"{carpeta_img[:-4]}-Y-{open_img[-4:]}"
            img_cb = f"{carpeta_img[:-4]}-Cb-{open_img[-4:]}"
            img_cr = f"{carpeta_img[:-4]}-Cr-{open_img[-4:]}"
            img_ycb = f"{carpeta_img[:-4]}-YCb-{open_img[-4:]}"
            img_ycr = f"{carpeta_img[:-4]}-YCr-{open_img[-4:]}"
            print(f"-- {name} --")
            gray_or_not = await calculate_psnr(img_y, carpeta_img)
            if gray_or_not >= 55:
                return {
                    "success": False,
                    "error": "La imágen no es válida, no posee tres canales de color",
                }
            else:
                # ---------------------------Reducir componentes--------------------------------
                # Reducir la componente Cb
                cb_redux = cv2.resize(
                    cb, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC
                )
                # Aplicar antialiasing
                cb_redux_smooth = cv2.GaussianBlur(cb_redux, (3, 3), 0)
                # Reducir la componente Cr
                cr_redux = cv2.resize(
                    cr, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC
                )
                # Aplicar antialiasing
                cr_redux_smooth = cv2.GaussianBlur(cr_redux, (3, 3), 0)
                # Guardar las imagenes resultantes
                cv2.imwrite(
                    carpeta_img[:-4] + "-cb-min-" + open_img[-4:], cb_redux_smooth
                )
                cv2.imwrite(
                    carpeta_img[:-4] + "-cr-min-" + open_img[-4:], cr_redux_smooth
                )
                cb_resize_path = f"{carpeta_img[:-4]}-cb-min-{open_img[-4:]}"
                cr_resize_path = f"{carpeta_img[:-4]}-cr-min-{open_img[-4:]}"
                return {
                    "success": True,
                    "error": False,
                    "img_crmin": cr_resize_path,
                    "img_cbmin": cb_resize_path,
                    "or_h": properties[0],
                    "or_w": properties[1],
                    "resize_h": resize_height,
                    "resize_w": resize_width,
                    "img_yfile": img_y,
                    "img_cbfile": img_cb,
                    "img_crfile": img_cr,
                    "img_ycbfile": img_ycb,
                    "img_ycrfile": img_ycr,
                }
        elif properties[2] < 3:
            # es una imagen binaria
            return {
                "success": False,
                "error": "La imágen no es válida, es una imágen con menos de tres espacios de color",
            }
        elif properties[2] > 3:
            # es una imagen contransparencia
            return {
                "success": False,
                "error": "La imágen no es válida, posee transparencia",
            }
    elif len(properties) > 3:
        # no sabemos que sea
        return {
            "success": False,
            "error": "La imágen no es válida, no se puede determinar si es una imagen",
        }
    elif len(properties) < 3:
        # tampoco sabemos que sea
        return {
            "success": False,
            "error": "La imágen no es válida, es una imagen que no posee colores",
        }
