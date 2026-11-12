"""Mide el sistema completo con fotos reales de México que no se usaron para entrenar.

Las matrículas de ETIQUETAS se transcribieron a ojo de las fotos de valid y test.
"""
import os

import cv2 as cv

from entrenar_detector import MEX, _iou, leer_coco
from estados import interpretar
from placas_main import analizar, leer_placa
from qr_placa import contenido_qr, generar_qr

ETIQUETAS = {
    "016": "HXP-740-A", "006": "TMC-86-74", "012": "NUL-90-78", "024": "PY-5244-A",
    "026": "NXK-42-06", "035": "NJJ-64-53", "011": "FPF-82-04", "083": "GDS-074-D",
    "086": "GAH-506-D",
    # "089" se omite: la foto corta la matrícula y no se puede transcribir con certeza.
}


def _cerca(a, b):
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def main():
    from modelos import RUTA_DETECTOR, RUTA_OCR, cargar
    detector, ocr = cargar(RUTA_DETECTOR), cargar(RUTA_OCR)

    placas = detectadas = falsas = aciertos_det = aciertos_caja = etiquetadas = estados_ok = qr_ok = errores = caracteres = 0
    print(f"{'foto':5} {'real':12} {'caja anotada':12} {'sistema':12} {'conf':5} estado")
    for ruta, cajas in leer_coco("valid", MEX, True) + leer_coco("test", MEX, True):
        foto = os.path.basename(ruta)[:3]
        frame = cv.imread(ruta)
        resultados = analizar(frame, detector, ocr)

        if not cajas:
            falsas += len(resultados)
            continue

        placas += len(cajas)
        x, y, w, h = cajas[0]
        cerca = [r for r in resultados
                 if _iou((x, y, w, h), (r["caja"][0], r["caja"][1], r["caja"][2] - r["caja"][0], r["caja"][3] - r["caja"][1])) >= 0.5]
        detectadas += bool(cerca)
        falsas += len(resultados) - bool(cerca)

        real = ETIQUETAS.get(foto)
        if real is None:
            continue
        etiquetadas += 1
        compacto = real.replace("-", "")
        m = max(2, int(w * 0.06))
        lectura_caja = interpretar(leer_placa(frame[max(y - m, 0):y + h + m, max(x - m, 0):x + w + m], ocr)[0])["matricula"]
        aciertos_caja += lectura_caja == real

        if cerca:
            r = cerca[0]
            aciertos_det += r["aceptada"] and r["matricula"] == real
            errores += _cerca(r["matricula"].replace("-", ""), compacto)
            caracteres += len(compacto)
            estado_real = interpretar(real)["estado"]
            estados_ok += r["aceptada"] and r["estado"] == estado_real
            if r["aceptada"]:
                leido = cv.QRCodeDetector().detectAndDecode(generar_qr(r["matricula"], r["estado"]))[0]
                qr_ok += leido == contenido_qr(r["matricula"], r["estado"])
            print(f"{foto:5} {real:12} {lectura_caja:12} {r['matricula']:12} {r['ocr']:.2f}  {r['estado']}")
        else:
            print(f"{foto:5} {real:12} {lectura_caja:12} {'(no detectó)':12}")

    print(f"\nDetector: {detectadas}/{placas} placas encontradas, {falsas} detecciones falsas")
    print(f"OCR con la caja anotada: {aciertos_caja}/{etiquetadas} matrículas exactas")
    print(f"Sistema completo: {aciertos_det}/{etiquetadas} matrículas exactas y aceptadas")
    print(f"Estado correcto en el sistema completo: {estados_ok}/{etiquetadas}")
    print(f"QR legibles en placas aceptadas: {qr_ok}")
    print(f"Caracteres errados (sistema completo): {errores}/{caracteres}")


if __name__ == "__main__":
    main()
