from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


class ItemScheme(BaseModel):
    size_img: list[int] = Field(None, title='Tamaño de la imagen')
    resize_img: list[int] = Field(
        None, title='Tamaño de reescalado de la imagen')
    unzip_file: str = Field(None, title='Llave de descifrado')
    init_cb: int = Field(None, title='Inicio de codificación de Cb')
    init_cr: int = Field(None, title='Inicio de codificación de Cr')
