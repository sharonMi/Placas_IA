
import argparse
import json
import multiprocessing as mp
import os
import shutil

import cv2 as cv
import numpy as np

from sinteticas import pegar_placa

DIR = os.path.dirname(os.path.abspath(__file__))
CRUDOS = os.path.join(DIR, "datos/crudos")
MEX = os.path.join(DIR, "datos/crudos_mex/v1")
CACHE = os.path.join(DIR, "datos/cache")
DESCARTAR_MEX = {"075", "088"}  # sin anotación, pero se alcanza a ver parte de una placa
TAM, REJILLA = 320, 20  # iguales a DET_TAM y DET_REJILLA; aquí no se importa Keras antes del fork


def leer_coco(particion, raiz=CRUDOS, incluir_vacias=False):
    carpeta = os.path.join(raiz, particion)
    with open(os.path.join(carpeta, "_annotations.coco.json")) as f:
        coco = json.load(f)
    cajas = {}
    for a in coco["annotations"]:
        x, y, w, h = (int(v) for v in a["bbox"])
        if w > 2 and h > 2:
            cajas.setdefault(a["image_id"], []).append((x, y, w, h))
    return [(os.path.join(carpeta, i["file_name"]), cajas.get(i["id"], []))
            for i in coco["images"] if incluir_vacias or i["id"] in cajas]


def objetivo(cajas, ancho, alto):
    y = np.zeros((REJILLA, REJILLA, 5), np.float32)
    for x0, y0, w, h in cajas:
        cx, cy = (x0 + w / 2) / ancho * REJILLA, (y0 + h / 2) / alto * REJILLA
        col, fil = min(int(cx), REJILLA - 1), min(int(cy), REJILLA - 1)
        y[fil, col] = (1, cx - col, cy - fil, np.sqrt(w / ancho), np.sqrt(h / alto))
    return y


def _recorte_negativo(img, cajas, rng):
    alto, ancho = img.shape[:2]
    for _ in range(25):
        w = int(ancho * rng.uniform(0.35, 0.85))
        h = int(alto * rng.uniform(0.35, 0.85))
        x, y = int(rng.integers(0, ancho - w + 1)), int(rng.integers(0, alto - h + 1))
        if all(x + w <= bx or bx + bw <= x or y + h <= by or by + bh <= y for bx, by, bw, bh in cajas):
            return img[y:y + h, x:x + w]
    return None


def _recorte_acercado(img, cajas, rng):
    """Acerca la cámara sin cortar ninguna placa, para variar el tamaño aparente."""
    alto, ancho = img.shape[:2]
    x0 = min(c[0] for c in cajas)
    y0 = min(c[1] for c in cajas)
    x1 = max(c[0] + c[2] for c in cajas)
    y1 = max(c[1] + c[3] for c in cajas)
    a = int(rng.integers(0, x0 + 1))
    b = int(rng.integers(0, y0 + 1))
    c = int(rng.integers(x1, ancho + 1))
    d = int(rng.integers(y1, alto + 1))
    return img[b:d, a:c], [(x - a, y - b, w, h) for x, y, w, h in cajas]


def _preparar(tarea):
    ruta, cajas, modo, semilla = tarea
    cv.setNumThreads(0)
    rng = np.random.default_rng(semilla)
    img = cv.imread(ruta)
    if img is None:
        return None
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    alto, ancho = img.shape[:2]
    cajas = [(x, y, min(w, ancho - x), min(h, alto - y)) for x, y, w, h in cajas if x < ancho and y < alto]
    if not cajas and modo not in ("real", "vacia"):
        return None

    if modo == "vacia":  # foto sin placa: un recorte distinto en cada repetición
        w, h = int(ancho * rng.uniform(0.6, 1)), int(alto * rng.uniform(0.6, 1))
        x, y = int(rng.integers(0, ancho - w + 1)), int(rng.integers(0, alto - h + 1))
        img = img[y:y + h, x:x + w]
    elif modo == "acercada":
        img, cajas = _recorte_acercado(img, cajas, rng)
    elif modo == "negativo":
        img = _recorte_negativo(img, cajas, rng)
        if img is None:
            return None
        cajas = []
    elif modo == "mexicana":
        for caja in cajas:
            pegar_placa(img, caja, rng)
        if rng.random() < 0.6:
            img, cajas = _recorte_acercado(img, cajas, rng)

    alto, ancho = img.shape[:2]
    return cv.resize(img, (TAM, TAM), interpolation=cv.INTER_AREA), objetivo(cajas, ancho, alto)


def construir(grupos, pool, limite=None, archivo=None):
    """Prepara las imágenes de varios grupos [(registros, modos), ...] en un solo arreglo.

    Se reserva todo de una vez porque juntar listas de imágenes duplicaba la RAM. Con
    archivo, las imágenes viven en disco y el sistema puede liberar esa memoria si la necesita.
    """
    tareas = [(r, c, modo, i * len(modos) + m)
              for registros, modos in grupos
              for m, modo in enumerate(modos) for i, (r, c) in enumerate(registros[:limite])]
    forma = (len(tareas), TAM, TAM, 3)
    if archivo:
        os.makedirs(CACHE, exist_ok=True)
        x = np.memmap(os.path.join(CACHE, archivo), np.uint8, "w+", shape=forma)
    else:
        x = np.empty(forma, np.uint8)
    y = np.empty((len(tareas), REJILLA, REJILLA, 5), np.float32)
    n = 0
    for resultado in pool.imap(_preparar, tareas, chunksize=32):
        if resultado is not None:
            x[n], y[n] = resultado
            n += 1
    return x[:n], y[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", type=int, default=40)
    parser.add_argument("--limite", type=int, default=None)
    parser.add_argument("--continuar", type=int, default=0, help="retoma RedDetector.keras desde esta época")
    args = parser.parse_args()

    with mp.Pool(14) as pool:
        # Las pocas fotos mexicanas se repiten con recortes distintos para que pesen en el entrenamiento.
        mex = leer_coco("train", MEX, incluir_vacias=True)
        con_placa = [r for r in mex if r[1]]
        sin_placa = [r for r in mex if not r[1] and os.path.basename(r[0])[:3] not in DESCARTAR_MEX]
        x, y = construir([(leer_coco("train"), ["real", "mexicana", "negativo"]),
                          (con_placa, ["real"] + ["acercada"] * 24),
                          (sin_placa, ["vacia"] * 8)], pool, args.limite, "x_train.bin")

        x_val, y_val = construir([(leer_coco("valid"), ["real", "mexicana", "negativo"])], pool, args.limite, "x_val.bin")
        x_test_real, y_test_real = construir([(leer_coco("test"), ["real"])], pool, args.limite)
        x_test_mx, y_test_mx = construir([(leer_coco("test"), ["mexicana"])], pool, args.limite)
        # valid y test de México quedan fuera del entrenamiento y del criterio de paro.
        reservadas = leer_coco("valid", MEX, True) + leer_coco("test", MEX, True)
        x_reales_mx, y_reales_mx = construir([(reservadas, ["real"])], pool)
    print("Datos:", x.shape, x_val.shape, x_test_real.shape, x_reales_mx.shape, flush=True)

    from modelos import RUTA_DETECTOR, Lotes, cargar, crear_detector, perdida_detector  # primero: elige el backend de Keras
    import keras
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from placas_main import cajas_desde_rejilla

    model = cargar(RUTA_DETECTOR) if args.continuar else crear_detector()
    # Sin XLA: reserva más de 2 GB de golpe y la GPU solo deja 1.7 GB libres para esta red.
    model.compile(loss=perdida_detector, optimizer=keras.optimizers.Adam(1e-3), jit_compile=False)
    model.summary()

    historial = model.fit(Lotes(x, y, 10, mezclar=True, aumentar=_aumentar), validation_data=Lotes(x_val, y_val, 10),
                          epochs=args.epocas, initial_epoch=args.continuar, verbose=2, callbacks=[
                              keras.callbacks.ModelCheckpoint(RUTA_DETECTOR, save_best_only=True),
                              keras.callbacks.ReduceLROnPlateau(factor=0.4, patience=4),
                              keras.callbacks.EarlyStopping(patience=12)])

    plt.xlabel("Epocas")
    plt.ylabel("Pérdida")
    plt.plot(historial.history["loss"], label="entrenamiento")
    plt.plot(historial.history["val_loss"], label="validación")
    plt.legend()
    plt.savefig(os.path.join(DIR, "curva_detector.png"))

    mejor = cargar(RUTA_DETECTOR)
    for nombre, xt, yt in [("fotos reales", x_test_real, y_test_real),
                           ("con placa mexicana sintética", x_test_mx, y_test_mx),
                           ("México real reservada", x_reales_mx, y_reales_mx)]:
        pred = mejor.predict(Lotes(xt, yt, 10), verbose=0)
        print(f"Prueba {nombre}: " + evaluar(pred, yt, cajas_desde_rejilla))

    shutil.rmtree(CACHE, ignore_errors=True)


def _aumentar(bx, by, rng):
    """Espejo horizontal y cambios de brillo distintos en cada época."""
    for j in range(len(bx)):
        if rng.random() < 0.5:
            bx[j], by[j] = bx[j, :, ::-1], by[j, :, ::-1]
            by[j, ..., 1] = by[j, ..., 0] * (1 - by[j, ..., 1])
        bx[j] = np.clip(bx[j] * rng.uniform(0.6, 1.3) + rng.uniform(-20, 20), 0, 255)
    return bx, by


def _iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - ix * iy
    return ix * iy / union if union else 0


def evaluar(pred, reales, decodificar):
    """Precisión y exhaustividad con IoU >= 0.5."""
    aciertos = detectadas = total = 0
    for p, r in zip(pred, reales):
        cajas_pred = [c[:4] for c in decodificar(p)]
        cajas_real = [c[:4] for c in decodificar(r, umbral=0.5)]
        detectadas += len(cajas_pred)
        total += len(cajas_real)
        for caja in cajas_real:
            mejor = max(cajas_pred, key=lambda c: _iou(caja, c), default=None)
            if mejor is not None and _iou(caja, mejor) >= 0.5:
                cajas_pred.remove(mejor)
                aciertos += 1
    return f"precisión {aciertos / max(detectadas, 1):.3f}, exhaustividad {aciertos / max(total, 1):.3f} ({total} placas)"


if __name__ == "__main__":
    main()
