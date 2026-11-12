"""
Cámara:  python placas_main.py
"""
import argparse
import os
from collections import deque

import cv2 as cv
import numpy as np

from estados import interpretar
from qr_placa import guardar_qr
from sinteticas import ALFABETO, OCR_ALTO, OCR_ANCHO

CAMARA = 0
VENTANA = "Detector de placas"
SALIDAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salidas")

UMBRAL_PLACA = 0.5
UMBRAL_LECTURA = 0.5  # confianza mínima de la lectura (ver decodificar)
MARGEN_RECORTE = 0.06
LECTURAS_MINIMAS = 5  # cuadros acumulados por placa antes de aceptarla y generar el QR
LECTURAS_MAXIMAS = 16  # solo votan los cuadros más recientes
UMBRAL_PISTA = 0.3  # solapamiento mínimo para considerar que es la misma placa
AUSENCIA_MAXIMA = 15  # cuadros sin verla antes de olvidar una placa

# Patrones válidos de 7 posiciones: D = dígito, L = letra, _ = vacío (ver estados.FORMATOS).
MASCARAS = ("LLLDDDL", "LLLDDDD", "LLDDDDL", "LLDDDDD", "DDDLLL_", "LDDLLL_")
_CLASES = {"D": np.arange(10), "L": np.arange(10, 36), "_": np.array([36])}

REJILLA, TAM = 20, 320
FUENTE = cv.FONT_HERSHEY_SIMPLEX


def cajas_desde_rejilla(salida, umbral=UMBRAL_PLACA):
    """Convierte la salida de la red en cajas relativas (x, y, w, h, confianza), sin duplicados."""
    filas, columnas = np.where(salida[..., 0] >= umbral)
    candidatas = []
    for f, c in zip(filas, columnas):
        p, tx, ty, raiz_w, raiz_h = salida[f, c]
        w, h = float(raiz_w) ** 2, float(raiz_h) ** 2
        candidatas.append(((c + tx) / REJILLA - w / 2, (f + ty) / REJILLA - h / 2, w, h, float(p)))

    candidatas.sort(key=lambda c: -c[4])
    finales = []
    for caja in candidatas:
        if all(_solapamiento(caja, otra) < 0.4 for otra in finales):
            finales.append(caja)
    return finales


def _solapamiento(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - ix * iy
    return ix * iy / union if union else 0


def _inferir(modelo, lote):
    from keras import ops
    return ops.convert_to_numpy(modelo(lote, training=False))[0]


def detectar_placas(frame, detector):
    """Devuelve una lista de (x0, y0, x1, y1, confianza) en píxeles de frame."""
    alto, ancho = frame.shape[:2]
    entrada = cv.cvtColor(cv.resize(frame, (TAM, TAM), interpolation=cv.INTER_AREA), cv.COLOR_BGR2RGB)
    salida = _inferir(detector, entrada[None].astype("float32"))

    cajas = []
    for x, y, w, h, p in cajas_desde_rejilla(salida):
        mx, my = w * MARGEN_RECORTE, h * MARGEN_RECORTE
        x0, y0 = max(int((x - mx) * ancho), 0), max(int((y - my) * alto), 0)
        x1, y1 = min(int((x + w + mx) * ancho), ancho), min(int((y + h + my) * alto), alto)
        if x1 - x0 >= 12 and y1 - y0 >= 6:
            cajas.append((x0, y0, x1, y1, p))
    return cajas


def probabilidades_ocr(recorte, ocr):
    """Devuelve la matriz 7x37 de probabilidades de la red para un recorte."""
    gris = cv.cvtColor(recorte, cv.COLOR_BGR2GRAY)
    interpolacion = cv.INTER_AREA if gris.shape[1] > OCR_ANCHO else cv.INTER_CUBIC
    entrada = cv.resize(gris, (OCR_ANCHO, OCR_ALTO), interpolation=interpolacion)
    return _inferir(ocr, entrada[None, ..., None].astype("float32"))


def decodificar(prob):
    """Convierte probabilidades en texto usando solo los formatos válidos de placa.

    En una posición de dígito solo compiten dígitos, así la O no puede ganarle al 0.
    La confianza combina dos cosas para que forzar un formato no haga pasar cualquier
    texto (por ejemplo, una placa extranjera) como válido:
      - qué tan clara fue cada elección dentro de las clases permitidas;
      - qué tanto apoyó la red ese formato (la probabilidad que dejó en las clases permitidas).
    Devuelve (texto, confianza).
    """
    mejor = None
    for mascara in MASCARAS:
        texto, puntaje, masas, claridad = "", 0.0, [], []
        for posicion, tipo in enumerate(mascara):
            clases = _CLASES[tipo]
            p = prob[posicion, clases]
            elegido = int(p.argmax())
            masa = max(float(p.sum()), 1e-9)
            texto += ALFABETO[clases[elegido]]
            puntaje += np.log(max(float(p[elegido]), 1e-9))
            masas.append(masa)
            claridad.append(float(p[elegido]) / masa)
        if mejor is None or puntaje > mejor[0]:
            apoyo = float(np.exp(np.mean(np.log(masas))))
            mejor = (puntaje, texto.rstrip("_"), min(min(claridad), apoyo))
    return mejor[1], mejor[2]


def leer_placa(recorte, ocr):
    """Devuelve (texto, confianza) de un solo recorte."""
    return decodificar(probabilidades_ocr(recorte, ocr))


def analizar(frame, detector, ocr):
    """Procesa un cuadro y devuelve un diccionario por cada placa encontrada."""
    resultados = []
    for x0, y0, x1, y1, p in detectar_placas(frame, detector):
        prob = probabilidades_ocr(frame[y0:y1, x0:x1], ocr)
        texto, confianza = decodificar(prob)
        info = interpretar(texto)
        info.update(caja=(x0, y0, x1, y1), det=p, ocr=confianza, prob=prob,
                    aceptada=info["valida"] and confianza >= UMBRAL_LECTURA)
        resultados.append(info)
    return resultados


def _a_xywh(caja):
    return caja[0], caja[1], caja[2] - caja[0], caja[3] - caja[1]


class Seguimiento:
    """Sigue cada placa entre cuadros y combina sus lecturas por votación."""

    def __init__(self):
        self.pistas = []

    def actualizar(self, resultados):
        """Asigna cada detección a su placa y devuelve las pistas visibles en este cuadro."""
        for pista in self.pistas:
            pista["ausente"] += 1
        libres = list(self.pistas)

        for r in sorted(resultados, key=lambda r: -r["det"]):
            caja = _a_xywh(r["caja"])
            pista = max(libres, key=lambda q: _solapamiento(caja, _a_xywh(q["caja"])), default=None)
            if pista is None or _solapamiento(caja, _a_xywh(pista["caja"])) < UMBRAL_PISTA:
                pista = {"lecturas": deque(maxlen=LECTURAS_MAXIMAS), "emitida": None}
                self.pistas.append(pista)
            else:
                libres = [q for q in libres if q is not pista]  # .remove() compararía arreglos NumPy

            pista.update(caja=r["caja"], ausente=0)
            pista["lecturas"].append(r["prob"])

            # Cada cuadro vota con sus probabilidades, y la suma se lee con los formatos válidos.
            texto, confianza = decodificar(np.mean(pista["lecturas"], axis=0))
            info = interpretar(texto)
            n = len(pista["lecturas"])
            info.update(caja=r["caja"], det=r["det"], ocr=confianza, n=n,
                        aceptada=info["valida"] and confianza >= UMBRAL_LECTURA and n >= LECTURAS_MINIMAS)
            pista["resultado"] = info

        self.pistas = [p for p in self.pistas if p["ausente"] <= AUSENCIA_MAXIMA]
        return [p for p in self.pistas if p["ausente"] == 0]


def dibujar(img, r):
    x0, y0, x1, y1 = r["caja"]
    color = (0, 220, 120) if r["aceptada"] else (0, 165, 255)
    cv.rectangle(img, (x0, y0), (x1, y1), color, 2, cv.LINE_AA)
    if r["aceptada"]:
        etiqueta = f"{r['matricula']}  {r['estado']}"
    elif r.get("n", LECTURAS_MINIMAS) < LECTURAS_MINIMAS:
        etiqueta = "leyendo..."
    else:
        etiqueta = "placa no reconocida"
    # Hershey no incluye acentos.
    etiqueta = etiqueta.translate(str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU"))
    (w, h), _ = cv.getTextSize(etiqueta, FUENTE, 0.6, 2)
    base = max(y0 - 8, h + 8)
    cv.rectangle(img, (x0, base - h - 6), (x0 + w + 8, base + 6), color, -1)
    cv.putText(img, etiqueta, (x0 + 4, base), FUENTE, 0.6, (0, 0, 0), 2, cv.LINE_AA)


def mostrar_qr(img, qr):
    """Coloca el QR en la esquina superior derecha."""
    lado = min(180, img.shape[0] // 3)
    miniatura = cv.cvtColor(cv.resize(qr, (lado, lado), interpolation=cv.INTER_NEAREST), cv.COLOR_GRAY2BGR)
    img[10:10 + lado, img.shape[1] - lado - 10:img.shape[1] - 10] = miniatura


def procesar_imagen(ruta, detector, ocr):
    frame = cv.imread(ruta)
    if frame is None:
        print(f"No se pudo abrir la imagen: {ruta}")
        return

    resultados = analizar(frame, detector, ocr)
    if not resultados:
        print("No se detectó ninguna placa.")

    for r in resultados:
        dibujar(frame, r)
        if r["aceptada"]:
            salida_qr, _ = guardar_qr(r["matricula"], r["estado"], SALIDAS)
            print(f"Placa: {r['matricula']} | Estado: {r['estado']} | "
                  f"confianza lectura: {r['ocr']:.2f} | QR: {salida_qr}")
        else:
            print(f"Placa detectada pero no válida como mexicana: lectura '{r['matricula']}', "
                  f"confianza {r['ocr']:.2f}")

    os.makedirs(SALIDAS, exist_ok=True)
    anotada = os.path.join(SALIDAS, "resultado_" + os.path.basename(ruta))
    cv.imwrite(anotada, frame)
    print(f"Imagen anotada: {anotada}")


def procesar_camara(indice, detector, ocr):
    video = cv.VideoCapture(indice)
    if not video.isOpened():
        print(f"No se pudo abrir la cámara {indice}")
        return

    print("Presiona 'q' para salir.")
    seguimiento, qr = Seguimiento(), None
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break

            for pista in seguimiento.actualizar(analizar(frame, detector, ocr)):
                r = pista["resultado"]
                dibujar(frame, r)
                if r["aceptada"] and r["matricula"] != pista["emitida"]:
                    pista["emitida"] = r["matricula"]
                    ruta, qr = guardar_qr(r["matricula"], r["estado"], SALIDAS)
                    print(f"Placa: {r['matricula']} | Estado: {r['estado']} | QR: {ruta}")

            if qr is not None:
                mostrar_qr(frame, qr)

            cv.imshow(VENTANA, frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video.release()
        cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagen", help="ruta de una imagen; sin esto se usa la cámara")
    parser.add_argument("--camara", type=int, default=CAMARA)
    args = parser.parse_args()

    from modelos import RUTA_DETECTOR, RUTA_OCR, cargar
    detector, ocr = cargar(RUTA_DETECTOR), cargar(RUTA_OCR)

    if args.imagen:
        procesar_imagen(args.imagen, detector, ocr)
    else:
        procesar_camara(args.camara, detector, ocr)


if __name__ == "__main__":
    main()
