"""Entrena la red que lee los caracteres de la placa. Genera RedOCR.keras."""
import argparse
import glob
import multiprocessing as mp
import os
import random

import cv2 as cv
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
ETIQUETAS_REALES = os.path.join(DIR, "datos/crudos_mex/etiquetas_ocr.txt")
REPETICIONES_REALES = 100  # variantes por placa real, para que pesen entre las sintéticas
_escenas = []


def _iniciar():
    global _escenas
    cv.setNumThreads(0)
    rutas = sorted(glob.glob(os.path.join(DIR, "datos/crudos/train/*.jpg")))
    random.Random(0).shuffle(rutas)
    _escenas = [cv.cvtColor(cv.imread(r), cv.COLOR_BGR2RGB) for r in rutas[:400]]


def _muestra(semilla):
    from sinteticas import muestra_ocr
    rng = np.random.default_rng(semilla)
    return muestra_ocr(rng, _escenas[int(rng.integers(len(_escenas)))])


def generar(n, base, pool):
    resultado = list(pool.imap(_muestra, range(base, base + n), chunksize=256))
    x = np.stack([r[0] for r in resultado])[..., None]
    y = np.array([r[1] for r in resultado], dtype="int32")
    return x, y


def leer_etiquetas_reales():
    """Une etiquetas_ocr.txt con las cajas del dataset: devuelve [(ruta, caja, matrícula)]."""
    from entrenar_detector import MEX, leer_coco
    cajas = {os.path.basename(r): (r, c[0]) for r, c in leer_coco("train", MEX)}
    reales = []
    with open(ETIQUETAS_REALES) as f:
        for linea in f:
            linea = linea.strip()
            if linea and not linea.startswith("#"):
                archivo, matricula = linea.split(",")
                ruta, caja = cajas[archivo]
                reales.append((ruta, caja, matricula))
    return reales


def _muestra_real(tarea):
    from sinteticas import muestra_real
    ruta, caja, matricula, semilla = tarea
    gris = cv.imread(ruta, cv.IMREAD_GRAYSCALE)
    return muestra_real(gris, caja, matricula, np.random.default_rng(semilla))


def generar_reales(pool, repeticiones=REPETICIONES_REALES):
    reales = leer_etiquetas_reales()
    tareas = [(*r, i * repeticiones + j) for i, r in enumerate(reales) for j in range(repeticiones)]
    resultado = list(pool.imap(_muestra_real, tareas, chunksize=64))
    print("Placas reales en el TXT:", len(reales), flush=True)
    return np.stack([r[0] for r in resultado])[..., None], np.array([r[1] for r in resultado], dtype="int32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muestras", type=int, default=80000)
    parser.add_argument("--epocas", type=int, default=40)
    args = parser.parse_args()

    with mp.Pool(14, _iniciar) as pool:
        x, y = generar(args.muestras, 0, pool)
        x_val, y_val = generar(args.muestras // 10, 10_000_000, pool)
        x_real, y_real = generar_reales(pool)
    x, y = np.concatenate([x, x_real]), np.concatenate([y, y_real])
    print("Datos:", x.shape, x_val.shape, flush=True)

    from modelos import RUTA_OCR, Lotes, cargar, crear_ocr  # primero: elige el backend de Keras
    import keras
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = crear_ocr()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(1e-3),
                  metrics=["accuracy"])
    model.summary()

    historial = model.fit(Lotes(x, y, 64, mezclar=True), validation_data=Lotes(x_val, y_val, 64),
                          epochs=args.epocas, verbose=2, callbacks=[
                              keras.callbacks.ModelCheckpoint(RUTA_OCR, save_best_only=True),
                              keras.callbacks.ReduceLROnPlateau(factor=0.4, patience=3),
                              keras.callbacks.EarlyStopping(patience=8)])

    plt.xlabel("Epocas")
    plt.ylabel("Pérdida")
    plt.plot(historial.history["loss"], label="entrenamiento")
    plt.plot(historial.history["val_loss"], label="validación")
    plt.legend()
    plt.savefig(os.path.join(DIR, "curva_ocr.png"))

    mejor = cargar(RUTA_OCR)
    pred = mejor.predict(Lotes(x_val, y_val, 32), verbose=0).argmax(-1)
    print(f"Caracteres correctos (sintético): {(pred == y_val).mean():.4f}")
    print(f"Placas completas correctas (sintético): {(pred == y_val).all(axis=1).mean():.4f}")


if __name__ == "__main__":
    main()
