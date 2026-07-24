"""Utilidades de datos para el generador de musica en ABC notation.

Aqui esta todo lo relacionado con: descargar el dataset de thesession.org,
armar el corpus de texto, el tokenizador a nivel de caracter y la validacion
de los tunes generados con music21.
"""

import os
import requests
import pandas as pd


# CSV oficial de thesession.org (se actualiza ~1 vez por semana)
THESESSION_URL = "https://raw.githubusercontent.com/adactio/thesession-data/main/csv/tunes.csv"

# Cada tune del corpus empieza con este prompt y termina con el separador.
# Usamos una linea en blanco como separador para que el modelo aprenda
# donde termina una melodia y empieza la siguiente.
TUNE_START = "X:1\n"
TUNE_SEP = "\n\n"


def download_tunes(dest="data/tunes.csv"):
    """Descarga tunes.csv de thesession si no existe todavia."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return dest
    r = requests.get(THESESSION_URL, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)
    return dest


def mode_to_key(mode):
    """Convierte el 'mode' de thesession (ej. 'Gmajor', 'Edorian') al
    campo K: de ABC estandar (ej. 'G', 'Edor'), que es lo que entiende music21."""
    mode = str(mode).strip()
    sufijos = {"major": "", "minor": "m", "dorian": "dor", "mixolydian": "mix"}
    for palabra, corto in sufijos.items():
        if mode.lower().endswith(palabra):
            raiz = mode[: -len(palabra)]
            return raiz + corto
    # si no reconocemos el modo, devolvemos lo que venga
    return mode


def build_corpus(csv_path, max_body_len=400):
    """Arma el corpus de texto a partir del CSV.

    Cada tune se reconstruye como un bloque ABC completo con cabecera
    (compas M: y tonalidad K:) para que music21 pueda leerlo despues.
    Se descartan tunes sin ABC o demasiado largos.
    """
    df = pd.read_csv(csv_path)
    bloques = []
    for _, fila in df.iterrows():
        abc = str(fila.get("abc", "")).strip()
        if not abc or abc == "nan":
            continue
        if len(abc) > max_body_len:
            continue
        meter = str(fila.get("meter", "4/4")).strip()
        key = mode_to_key(fila.get("mode", "C"))
        bloque = TUNE_START + "M:" + meter + "\n" + "K:" + key + "\n" + abc
        bloques.append(bloque)
    corpus = TUNE_SEP.join(bloques) + TUNE_SEP
    return corpus, bloques


class CharTokenizer:
    """Tokenizador a nivel de caracter: cada caracter unico es un entero."""

    def __init__(self, texto):
        chars = sorted(set(texto))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def extract_first_tune(texto_generado):
    """Del texto generado, toma solo el primer tune completo (hasta el
    primer separador) y le agrega el prompt inicial si hace falta."""
    if not texto_generado.startswith("X:"):
        texto_generado = TUNE_START + texto_generado
    return texto_generado.split(TUNE_SEP)[0].strip()


def is_valid_abc(texto):
    """True si music21 logra parsear el tune (ABC bien formado)."""
    from music21 import converter

    try:
        pieza = converter.parse(texto, format="abc")
        # un tune valido debe tener al menos una nota
        return len(pieza.recurse().notes) > 0
    except Exception:
        return False
