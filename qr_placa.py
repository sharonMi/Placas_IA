"""Genera el código QR con la información de una placa (solo OpenCV)."""
import os
import cv2 as cv

ESCALA = 10
BORDE = 40  # zona blanca que los lectores necesitan alrededor del código


def contenido_qr(matricula, estado):
    return f"Placa: {matricula}\nEstado: {estado}\nPais: Mexico"


def generar_qr(matricula, estado):
    """Devuelve el QR como imagen en escala de grises."""
    codigo = cv.QRCodeEncoder.create().encode(contenido_qr(matricula, estado))
    grande = cv.resize(codigo, None, fx=ESCALA, fy=ESCALA, interpolation=cv.INTER_NEAREST)
    return cv.copyMakeBorder(grande, BORDE, BORDE, BORDE, BORDE, cv.BORDER_CONSTANT, value=255)


def guardar_qr(matricula, estado, carpeta):
    """Guarda el QR como PNG y devuelve (ruta, imagen)."""
    os.makedirs(carpeta, exist_ok=True)
    imagen = generar_qr(matricula, estado)
    ruta = os.path.join(carpeta, f"qr_{matricula}.png")
    cv.imwrite(ruta, imagen)
    return ruta, imagen
