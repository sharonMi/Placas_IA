"""Generador de placas mexicanas sintéticas para entrenar las redes.

No hay un dataset público de placas mexicanas con su texto, así que se dibujan placas
con etiqueta conocida y se degradan para parecerse a una foto.
"""
import os
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from estados import RANGOS, RANGOS_DOS_LETRAS, CDMX

ALFABETO = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"  # "_" rellena las posiciones vacías
LARGO_TEXTO = 7
OCR_ALTO, OCR_ANCHO = 96, 192

ANCHO_PLACA, ALTO_PLACA = 440, 220
LETRAS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_RUTAS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    # Las placas reales suelen tener trazo más delgado.
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
]
FUENTES = [r for r in _RUTAS if os.path.exists(r)]


def _serie(rng, inicio, fin):
    """Elige al azar una serie entre dos series del mismo largo, en base 26."""
    a_num = lambda s: sum(LETRAS.index(c) * 26 ** (len(s) - 1 - i) for i, c in enumerate(s))
    n = int(rng.integers(a_num(inicio), a_num(fin) + 1))
    return "".join(LETRAS[n // 26 ** (len(inicio) - 1 - i) % 26] for i in range(len(inicio)))


def texto_aleatorio(rng):
    """Devuelve (texto con guiones, estado) de una matrícula plausible."""
    digitos = lambda k: "".join(str(d) for d in rng.integers(0, 10, k))
    letras = lambda k: "".join(rng.choice(list(LETRAS), k))

    tipo = rng.choice(6, p=[0.46, 0.22, 0.08, 0.04, 0.14, 0.06])
    if tipo in (2, 3):
        inicio = digitos(3) if tipo == 2 else letras(1) + digitos(2)
        return f"{inicio}-{letras(3)}", CDMX
    if tipo >= 4:
        estado, ini, fin = RANGOS_DOS_LETRAS[int(rng.integers(len(RANGOS_DOS_LETRAS)))]
        serie = _serie(rng, ini, fin)
        if tipo == 4:
            return f"{serie}-{digitos(4)}-{letras(1)}", estado
        return f"{serie}-{digitos(2)}-{digitos(3)}", estado

    estado, ini, fin = RANGOS[int(rng.integers(len(RANGOS)))]
    serie = _serie(rng, ini, fin)
    if tipo == 0:
        return f"{serie}-{digitos(3)}-{letras(1)}", estado
    return f"{serie}-{digitos(2)}-{digitos(2)}", estado


def etiquetas(texto):
    """Convierte 'ABC-123-D' en índices del alfabeto, con relleno hasta LARGO_TEXTO."""
    compacto = texto.replace("-", "").ljust(LARGO_TEXTO, "_")
    return [ALFABETO.index(c) for c in compacto]


def _color(rng, bajo, alto):
    return tuple(int(v) for v in rng.integers(bajo, alto, 3))


def dibujar_placa(texto, estado, rng):
    """Dibuja una placa y la devuelve como arreglo RGB."""
    fondo = _color(rng, 205, 256)
    img = Image.new("RGB", (ANCHO_PLACA, ALTO_PLACA), fondo)
    d = ImageDraw.Draw(img, "RGBA")

    # Manchas de color pálido: imitan el arte de fondo de cada estado.
    for _ in range(int(rng.integers(0, 7))):
        x, y = int(rng.integers(0, ANCHO_PLACA)), int(rng.integers(0, ALTO_PLACA))
        r = int(rng.integers(20, 150))
        d.ellipse((x - r, y - r, x + r, y + r), fill=(*_color(rng, 60, 256), int(rng.integers(20, 75))))

    fuente_ruta = FUENTES[int(rng.integers(len(FUENTES)))]

    # Franja superior con el nombre del estado.
    franja = int(rng.integers(30, 50))
    if rng.random() < 0.7:
        d.rectangle((0, 0, ANCHO_PLACA, franja), fill=(*_color(rng, 0, 170), 255))
        tinta_estado = (250, 250, 250)
    else:
        tinta_estado = _color(rng, 0, 120)
    fuente = ImageFont.truetype(fuente_ruta, int(rng.integers(20, 30)))
    d.text((ANCHO_PLACA / 2, franja / 2), estado.upper(), font=fuente, fill=tinta_estado, anchor="mm")

    # Leyenda inferior.
    fuente = ImageFont.truetype(fuente_ruta, int(rng.integers(15, 24)))
    leyenda = ["MEXICO", "TRANSPORTE PRIVADO", "PARTICULAR", ""][int(rng.integers(4))]
    d.text((ANCHO_PLACA / 2, ALTO_PLACA - 20), leyenda, font=fuente, fill=_color(rng, 0, 130), anchor="mm")

    # Triángulos de colores en la parte baja, como el arte de varios estados.
    if rng.random() < 0.4:
        base = ALTO_PLACA - int(rng.integers(0, 12))
        for x in range(0, ANCHO_PLACA, int(rng.integers(35, 70))):
            pico = base - int(rng.integers(20, 48))
            d.polygon([(x, base), (x + 28, pico), (x + 56, base)], fill=(*_color(rng, 40, 256), 190))

    # Matrícula: se dibuja aparte y se estira, porque en las reales es alta y angosta.
    azar = rng.random()
    if azar < 0.12:
        texto = texto.replace("-", " ")
    elif azar < 0.6:
        texto = texto.replace("-", " - ")
    fuente = ImageFont.truetype(fuente_ruta, 120)
    x0, y0, x1, y1 = d.textbbox((0, 0), texto, font=fuente)
    capa = Image.new("L", (x1 - x0 + 8, y1 - y0 + 8), 0)
    ImageDraw.Draw(capa).text((4 - x0, 4 - y0), texto, font=fuente, fill=255)
    ancho = int(ANCHO_PLACA * rng.uniform(0.76, 0.95))
    alto = int(ALTO_PLACA * rng.uniform(0.36, 0.56))
    capa = capa.resize((ancho, alto), Image.BILINEAR)

    if rng.random() < 0.8:
        tinta = _color(rng, 0, 45)
    else:
        tinta = ((20, 30, 110), (120, 20, 30), (15, 80, 45))[int(rng.integers(3))]
    img.paste(tinta, ((ANCHO_PLACA - ancho) // 2, int(ALTO_PLACA * rng.uniform(0.46, 0.54)) - alto // 2), capa)

    # Remaches.
    for cx in (70, ANCHO_PLACA - 70):
        cy = franja + 8
        d.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(*_color(rng, 70, 170), 255))

    # Código pequeño en una esquina superior.
    if rng.random() < 0.4:
        x = ANCHO_PLACA - int(rng.integers(40, 62))
        d.rectangle((x, franja + 2, x + 26, franja + 28), fill=(30, 30, 30, 255))

    # Portaplacas de la agencia: marco oscuro con leyenda en la parte inferior.
    if rng.random() < 0.3:
        grosor = int(rng.integers(6, 20))
        marco = _color(rng, 0, 60)
        d.rectangle((0, 0, ANCHO_PLACA - 1, ALTO_PLACA - 1), outline=marco, width=grosor)
        if rng.random() < 0.6:
            alto_marco = int(rng.integers(24, 44))
            d.rectangle((0, ALTO_PLACA - alto_marco, ANCHO_PLACA, ALTO_PLACA), fill=(*marco, 255))
            fuente = ImageFont.truetype(fuente_ruta, int(rng.integers(14, 26)))
            marca = ["NISSAN", "HYUNDAI", "VOLKSWAGEN", "CHEVROLET", "TOYOTA", "RACING"][int(rng.integers(6))]
            d.text((ANCHO_PLACA / 2, ALTO_PLACA - alto_marco / 2), marca, font=fuente, fill=(235, 235, 235), anchor="mm")

    return np.array(img)


def _fotometria(img, rng):
    """Iluminación, desenfoque, ruido y compresión JPEG."""
    salida = img.astype(np.float32) / 255

    if rng.random() < 0.5:  # sombra lineal sobre la placa
        alto, ancho = salida.shape[:2]
        rampa = np.linspace(rng.uniform(0.45, 1), rng.uniform(0.45, 1), ancho, dtype=np.float32)
        rampa = rampa[None, :] if salida.ndim == 2 else rampa[None, :, None]
        salida = salida * np.broadcast_to(rampa, salida.shape)

    salida = np.clip(salida, 0, 1) ** rng.uniform(0.55, 1.7)
    salida = salida * rng.uniform(0.6, 1.15) + rng.uniform(-0.05, 0.08)

    if rng.random() < 0.7:
        k = int(rng.choice([3, 5]))
        salida = cv.GaussianBlur(salida, (k, k), rng.uniform(0.4, 1.6))
    if rng.random() < 0.25:  # movimiento horizontal
        k = int(rng.choice([3, 5, 7]))
        nucleo = np.zeros((k, k), np.float32)
        nucleo[k // 2, :] = 1 / k
        salida = cv.filter2D(salida, -1, nucleo)

    salida = salida + rng.normal(0, rng.uniform(0.003, 0.05), salida.shape)
    salida = (np.clip(salida, 0, 1) * 255).astype(np.uint8)

    calidad = int(rng.integers(25, 96))
    ok, buf = cv.imencode(".jpg", salida, [cv.IMWRITE_JPEG_QUALITY, calidad])
    return cv.imdecode(buf, cv.IMREAD_UNCHANGED)


def recorte_de_fondo(escena, rng, alto, ancho):
    """Recorta y escala una parte aleatoria de una escena."""
    h, w = escena.shape[:2]
    cw = int(rng.integers(max(16, w // 8), max(17, w // 2)))
    ch = max(8, cw * alto // ancho)
    y = int(rng.integers(0, max(1, h - ch)))
    x = int(rng.integers(0, max(1, w - cw)))
    return cv.resize(escena[y:y + ch, x:x + cw], (ancho, alto), interpolation=cv.INTER_LINEAR)


def muestra_ocr(rng, escena):
    """Devuelve (imagen gris OCR_ALTO x OCR_ANCHO, etiquetas) imitando un recorte del detector."""
    texto, estado = texto_aleatorio(rng)
    placa = dibujar_placa(texto, estado, rng)

    # Las esquinas caen con holgura variable: simula errores de la caja del detector.
    margen_x = rng.uniform(-0.02, 0.10, 2) * OCR_ANCHO
    margen_y = rng.uniform(-0.02, 0.14, 2) * OCR_ALTO
    izq, der = margen_x[0], OCR_ANCHO - margen_x[1]
    arr, aba = margen_y[0], OCR_ALTO - margen_y[1]
    destino = np.float32([[izq, arr], [der, arr], [der, aba], [izq, aba]])
    destino += rng.uniform(-0.045, 0.045, (4, 2)) * [OCR_ANCHO, OCR_ALTO]

    origen = np.float32([[0, 0], [ANCHO_PLACA, 0], [ANCHO_PLACA, ALTO_PLACA], [0, ALTO_PLACA]])
    matriz = cv.getPerspectiveTransform(origen, destino)
    lienzo = recorte_de_fondo(escena, rng, OCR_ALTO, OCR_ANCHO)
    cv.warpPerspective(placa, matriz, (OCR_ANCHO, OCR_ALTO), lienzo, cv.INTER_AREA, cv.BORDER_TRANSPARENT)

    # Baja resolución: la placa pudo verse pequeña en la foto original.
    if rng.random() < 0.5:
        f = rng.uniform(0.3, 0.8)
        chico = cv.resize(lienzo, None, fx=f, fy=f, interpolation=cv.INTER_AREA)
        lienzo = cv.resize(chico, (OCR_ANCHO, OCR_ALTO), interpolation=cv.INTER_LINEAR)

    gris = _fotometria(cv.cvtColor(lienzo, cv.COLOR_RGB2GRAY), rng)
    return gris, etiquetas(texto)


def muestra_real(gris, caja, texto, rng):
    """Variante de una placa real: recorte con holgura y perspectiva distintas en cada llamada."""
    x, y, w, h = caja
    mx = rng.uniform(-0.02, 0.10, 2) * w
    my = rng.uniform(-0.02, 0.14, 2) * h
    origen = np.float32([[x - mx[0], y - my[0]], [x + w + mx[1], y - my[0]],
                         [x + w + mx[1], y + h + my[1]], [x - mx[0], y + h + my[1]]])
    origen += rng.uniform(-0.03, 0.03, (4, 2)) * [w, h]
    destino = np.float32([[0, 0], [OCR_ANCHO, 0], [OCR_ANCHO, OCR_ALTO], [0, OCR_ALTO]])
    matriz = cv.getPerspectiveTransform(origen, destino)
    recorte = cv.warpPerspective(gris, matriz, (OCR_ANCHO, OCR_ALTO), flags=cv.INTER_CUBIC,
                                 borderMode=cv.BORDER_REPLICATE)
    # Suave a propósito: la foto ya viene degradada y debe seguir siendo legible.
    salida = (recorte.astype(np.float32) / 255) ** rng.uniform(0.8, 1.25) * rng.uniform(0.85, 1.1)
    salida += rng.normal(0, rng.uniform(0.002, 0.02), salida.shape)
    return (np.clip(salida, 0, 1) * 255).astype(np.uint8), etiquetas(texto)


def pegar_placa(img, caja, rng):
    """Sustituye en img (RGB) la zona caja=(x, y, w, h) por una placa mexicana sintética."""
    x, y, w, h = caja
    if w < 8 or h < 4:
        return
    texto, estado = texto_aleatorio(rng)
    placa = cv.resize(dibujar_placa(texto, estado, rng), (w, h), interpolation=cv.INTER_AREA)

    # Se iguala la luminosidad para que no parezca una calcomanía brillante.
    zona = img[y:y + h, x:x + w]
    factor = np.clip((zona.mean() + 25) / max(placa.mean(), 1) * rng.uniform(0.85, 1.15), 0.35, 1.15)
    placa = np.clip(placa.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        placa = cv.GaussianBlur(placa, (3, 3), rng.uniform(0.4, 1.2))
    img[y:y + h, x:x + w] = placa
