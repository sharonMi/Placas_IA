import math
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")  # la GPU solo tiene 4 GB

import keras
import numpy as np
from keras import Input, ops
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D, Rescaling, Reshape)
from keras.models import Sequential

from sinteticas import ALFABETO, LARGO_TEXTO, OCR_ALTO, OCR_ANCHO

DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_DETECTOR = os.path.join(DIR, "RedDetector.keras")
RUTA_OCR = os.path.join(DIR, "RedOCR.keras")

DET_TAM = 320
DET_REJILLA = 20


def _bloque(filtros, dilatacion=1):
    return [Conv2D(filtros, 3, padding="same", dilation_rate=dilatacion, use_bias=False),
            BatchNormalization(),
            Activation("relu")]


def crear_detector():
    """Divide la imagen en 20x20 celdas y da 5 salidas por celda: [hay placa, cx, cy, w, h]."""
    return Sequential([
        Input(shape=(DET_TAM, DET_TAM, 3)),
        Rescaling(1 / 255),
        *_bloque(32), MaxPooling2D(),
        *_bloque(64), MaxPooling2D(),
        *_bloque(128), *_bloque(128), MaxPooling2D(),
        *_bloque(256), *_bloque(256), MaxPooling2D(),
        *_bloque(256, 2), *_bloque(256, 3),  # dilatadas: ven placas más grandes que la celda
        Dropout(0.1),
        Conv2D(5, 1, activation="sigmoid"),
    ], name="detector_placas")


def perdida_detector(y_real, y_pred):
    """Focal loss para saber dónde hay placa + Huber para ajustar la caja."""
    objeto = y_real[..., 0]
    p = ops.clip(y_pred[..., 0], 1e-6, 1 - 1e-6)
    ce = -(objeto * ops.log(p) + (1 - objeto) * ops.log(1 - p))
    pt = objeto * p + (1 - objeto) * (1 - p)
    focal = (1 - pt) ** 2 * (0.75 * objeto + 0.25 * (1 - objeto)) * ce

    error = ops.abs(y_real[..., 1:] - y_pred[..., 1:])
    huber = ops.sum(ops.where(error < 0.1, 0.5 * error ** 2 / 0.1, error - 0.05), axis=-1)

    positivos = ops.maximum(ops.sum(objeto), 1.0)
    return ops.sum(focal) / positivos + 5 * ops.sum(objeto * huber) / positivos


def crear_ocr():
    """Clasifica cada una de las 7 posiciones en 37 clases (0-9, A-Z y vacío)."""
    return Sequential([
        Input(shape=(OCR_ALTO, OCR_ANCHO, 1)),
        Rescaling(1 / 255),
        *_bloque(32), MaxPooling2D(),
        *_bloque(64), MaxPooling2D(),
        *_bloque(128), *_bloque(128), MaxPooling2D(),
        *_bloque(192), MaxPooling2D(),
        Conv2D(64, 1, activation="relu"),
        Flatten(),
        Dense(768, activation="relu"),
        Dropout(0.3),
        Dense(LARGO_TEXTO * len(ALFABETO)),
        Reshape((LARGO_TEXTO, len(ALFABETO))),
        Activation("softmax"),
    ], name="ocr_placas")


class Lotes(keras.utils.PyDataset):
    """Entrega los datos por lotes para que nunca se carguen completos en la GPU."""

    def __init__(self, x, y, lote, mezclar=False, aumentar=None):
        super().__init__()
        self.x, self.y, self.lote, self.mezclar, self.aumentar = x, y, lote, mezclar, aumentar
        self.rng = np.random.default_rng(7)
        self.orden = self.rng.permutation(len(x)) if mezclar else np.arange(len(x))

    def __len__(self):
        return math.ceil(len(self.x) / self.lote)

    def on_epoch_end(self):
        if self.mezclar:
            self.orden = self.rng.permutation(len(self.x))

    def __getitem__(self, i):
        idx = self.orden[i * self.lote:(i + 1) * self.lote]
        bx, by = self.x[idx].astype("float32"), self.y[idx].copy()
        return self.aumentar(bx, by, self.rng) if self.aumentar else (bx, by)


def cargar(ruta):
    """Carga una red para inferencia; la pérdida solo hace falta al entrenar."""
    return keras.saving.load_model(ruta, compile=False)
