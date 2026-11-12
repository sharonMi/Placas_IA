import re

# (estado, primera serie, última serie)
RANGOS = [
    ("Aguascalientes", "AAA", "AFZ"), ("Baja California", "AGA", "CYZ"),
    ("Baja California Sur", "CZA", "DEZ"), ("Campeche", "DFA", "DKZ"),
    ("Chiapas", "DLA", "DSZ"), ("Chihuahua", "DTA", "ETZ"),
    ("Coahuila", "EUA", "FPZ"), ("Colima", "FRA", "FWZ"),
    ("Durango", "FXA", "GFZ"), ("Guanajuato", "GGA", "GYZ"),
    ("Guerrero", "GZA", "HFZ"), ("Hidalgo", "HGA", "HRZ"),
    ("Jalisco", "HSA", "LFZ"), ("Estado de México", "LGA", "PEZ"),
    ("Michoacán", "PFA", "PUZ"), ("Morelos", "PVA", "RDZ"),
    ("Nayarit", "REA", "RJZ"), ("Nuevo León", "RKA", "TGZ"),
    ("Oaxaca", "THA", "TMZ"), ("Puebla", "TNA", "UJZ"),
    ("Querétaro", "UKA", "UPZ"), ("Quintana Roo", "URA", "UVZ"),
    ("San Luis Potosí", "UWA", "VEZ"), ("Sinaloa", "VFA", "VSZ"),
    ("Sonora", "VTA", "WKZ"), ("Tabasco", "WLA", "WWZ"),
    ("Tamaulipas", "WXA", "XSZ"), ("Tlaxcala", "XTA", "XXZ"),
    ("Veracruz", "XYA", "YVZ"), ("Yucatán", "YWA", "ZCZ"),
    ("Zacatecas", "ZDA", "ZHZ"),
]

# Series de dos letras (formato AB-1234-C), con los mismos estados.
RANGOS_DOS_LETRAS = [
    ("Aguascalientes", "AA", "AF"), ("Baja California", "AG", "CD"),
    ("Baja California Sur", "CE", "CL"), ("Campeche", "CM", "CU"),
    ("Chiapas", "CV", "DC"), ("Chihuahua", "DD", "EG"),
    ("Coahuila", "EH", "FB"), ("Colima", "FC", "FJ"),
    ("Durango", "FK", "FX"), ("Guanajuato", "FY", "GW"),
    ("Guerrero", "GX", "HG"), ("Hidalgo", "HH", "HT"),
    ("Jalisco", "HU", "KK"), ("Estado de México", "KL", "MS"),
    ("Michoacán", "MT", "NT"), ("Morelos", "NU", "NZ"),
    ("Nayarit", "PA", "PG"), ("Nuevo León", "PH", "RP"),
    ("Oaxaca", "RR", "RY"), ("Puebla", "RZ", "SR"),
    ("Querétaro", "SS", "SY"), ("Quintana Roo", "SZ", "TB"),
    ("San Luis Potosí", "TC", "TP"), ("Sinaloa", "TR", "UL"),
    ("Sonora", "UM", "VK"), ("Tabasco", "VL", "VT"),
    ("Tamaulipas", "VU", "WX"), ("Tlaxcala", "WY", "XE"),
    ("Veracruz", "XF", "YM"), ("Yucatán", "YN", "YU"),
    ("Zacatecas", "YV", "ZJ"),
]

CDMX = "Ciudad de México"
FRONTERA = "Zona fronteriza (estado no determinable)"
DESCONOCIDO = "Desconocido"

# (expresión sobre el texto sin guiones, función que le pone los guiones)
FORMATOS = [
    (re.compile(r"^[A-Z]{3}\d{3}[A-Z]$"), lambda t: f"{t[:3]}-{t[3:6]}-{t[6]}"),
    (re.compile(r"^[A-Z]{3}\d{4}$"), lambda t: f"{t[:3]}-{t[3:5]}-{t[5:]}"),
    (re.compile(r"^[A-Z]{2}\d{4}[A-Z]$"), lambda t: f"{t[:2]}-{t[2:6]}-{t[6]}"),
    (re.compile(r"^[A-Z]{2}\d{5}$"), lambda t: f"{t[:2]}-{t[2:4]}-{t[4:]}"),
    (re.compile(r"^(\d{3}|[A-Z]\d{2})[A-Z]{3}$"), lambda t: f"{t[:3]}-{t[3:]}"),
]


def estado_por_serie(letras):
    """Devuelve el estado al que pertenece una serie de dos o tres letras."""
    for estado, inicio, fin in (RANGOS if len(letras) == 3 else RANGOS_DOS_LETRAS):
        if inicio <= letras <= fin:
            return estado
    return DESCONOCIDO


def interpretar(texto):
    """Valida una matrícula y devuelve {matricula, estado, valida}."""
    limpio = re.sub(r"[^A-Z0-9]", "", texto.upper())

    for patron, formatear in FORMATOS:
        if patron.match(limpio):
            matricula = formatear(limpio)
            break
    else:
        return {"matricula": limpio, "estado": DESCONOCIDO, "valida": False}

    if limpio[:3].isalpha():
        # Las placas fronterizas ZAB-12-34 usan la Z como prefijo, no como serie.
        if limpio[0] == "Z" and limpio[3:].isdigit():
            estado = FRONTERA
        else:
            estado = estado_por_serie(limpio[:3])
    elif limpio[:2].isalpha():
        estado = estado_por_serie(limpio[:2])
    else:
        estado = CDMX  # 123-ABC o A12-ABC

    return {"matricula": matricula, "estado": estado, "valida": estado != DESCONOCIDO}
