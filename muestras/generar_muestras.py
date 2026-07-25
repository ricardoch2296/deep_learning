"""Genera melodias con Transformer y GRU, y crea comparacion.html.

Uso (desde la raiz del proyecto, con checkpoints entrenados):
    python3 muestras/generar_muestras.py 456

El numero es la semilla: misma semilla = misma melodia; otra semilla = otra.
"""

import glob
import json
import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(root, "src"))
sys.path.insert(0, root)
import abc_utils as A

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import GPT2Config, GPT2LMHeadModel

TEMPERATURA = 0.6
TOP_K = 40
MAX_CARACTERES = 400
INTENTOS = 5  # genera varias y se queda con la valida mas larga (como en el notebook)
PROMPT = "X:1\nM:6/8\nK:G\n"  # jig en Sol, igual que muchos tunes del corpus


def elegir_siguiente(logits, temperatura, top_k):
    logits = logits / temperatura
    if top_k:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class GPTLightning(L.LightningModule):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout, lr, max_steps, warmup_steps):
        super().__init__()
        self.save_hyperparameters()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=block_size,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.model = GPT2LMHeadModel(config)

    def forward(self, x):
        return self.model(input_ids=x).logits

    @torch.no_grad()
    def generate(self, tok, prompt=A.TUNE_START, max_new_tokens=400, temperature=0.6, top_k=40):
        self.eval()
        ids = torch.tensor([tok.encode(prompt)], device=self.device)
        for _ in range(max_new_tokens):
            ids_cond = ids[:, -self.hparams.block_size:]
            logits = self(ids_cond)[:, -1, :]
            nxt = elegir_siguiente(logits, temperature, top_k)
            ids = torch.cat([ids, nxt], dim=1)
        return tok.decode(ids[0].tolist())


class GRULightning(L.LightningModule):
    def __init__(self, vocab_size, emb_dim, hidden, num_layers, dropout, lr):
        super().__init__()
        self.save_hyperparameters()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        out, h = self.gru(self.emb(x), h)
        return self.head(out), h

    @torch.no_grad()
    def generate(self, tok, prompt=A.TUNE_START, max_new_tokens=400, temperature=0.6, top_k=40):
        self.eval()
        ids = tok.encode(prompt)
        x = torch.tensor([ids], device=self.device)
        logits, h = self(x)
        salida = list(ids)
        for _ in range(max_new_tokens):
            nxt = elegir_siguiente(logits[:, -1, :], temperature, top_k)
            salida.append(int(nxt.item()))
            logits, h = self(nxt, h)
        return tok.decode(salida)


def buscar_checkpoint(carpeta):
    archivos = sorted(glob.glob(f"checkpoints/{carpeta}/*.ckpt"))
    if not archivos:
        raise FileNotFoundError(f"No hay checkpoint en checkpoints/{carpeta}/")
    return archivos[-1]


def guardar_abc_y_midi(texto_abc, ruta_abc, ruta_midi):
    from music21 import converter
    from music21.harmony import ChordSymbol

    with open(ruta_abc, "w", encoding="utf-8") as f:
        f.write(texto_abc)
    try:
        pieza = converter.parse(texto_abc, format="abc")
        for acorde in pieza.recurse().getElementsByClass(ChordSymbol):
            acorde.writeAsChord = True
        pieza.write("midi", ruta_midi)
    except Exception as e:
        print("No se pudo crear MIDI:", ruta_midi, "-", e)


def extraer_melodia(texto):
    """Toma un tune sin cortar en el primer salto de linea del cuerpo."""
    if not texto.startswith("X:"):
        texto = A.TUNE_START + texto
    partes = texto.split(A.TUNE_SEP)
    salida = partes[0].strip()
    for p in partes[1:]:
        p = p.strip()
        if p.startswith("X:"):
            break
        if p:
            salida = salida + "\n" + p
    return salida.strip()


def generar_mejor(modelo, tok, semilla):
    """Varios intentos con la misma semilla base; elige la valida mas larga."""
    mejor = None
    for i in range(INTENTOS):
        L.seed_everything(semilla + i * 997)
        raw = modelo.generate(
            tok, prompt=PROMPT, max_new_tokens=MAX_CARACTERES,
            temperature=TEMPERATURA, top_k=TOP_K,
        )
        tune = extraer_melodia(raw)
        if not A.is_valid_abc(tune):
            continue
        if mejor is None or len(tune) > len(mejor):
            mejor = tune
    if mejor is not None:
        return mejor
    L.seed_everything(semilla)
    return extraer_melodia(modelo.generate(
        tok, prompt=PROMPT, max_new_tokens=MAX_CARACTERES,
        temperature=TEMPERATURA, top_k=TOP_K,
    ))


def poner_titulo(texto_abc, titulo):
    lineas = texto_abc.splitlines()
    for i, ln in enumerate(lineas):
        if ln.startswith("T:"):
            lineas[i] = "T:" + titulo
            return "\n".join(lineas)
    for i, ln in enumerate(lineas):
        if ln.startswith("X:"):
            lineas.insert(i + 1, "T:" + titulo)
            return "\n".join(lineas)
    return texto_abc


def crear_html(semilla, abc_transformer, abc_gru, ruta):
    abc_t = json.dumps(abc_transformer)
    abc_g = json.dumps(abc_gru)
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Comparacion Transformer vs GRU</title>
<script src="https://cdn.jsdelivr.net/npm/abcjs@6/dist/abcjs-basic-min.js"></script>
<style>
  body {{ font-family: sans-serif; margin: 2rem; max-width: 800px; }}
  h1 {{ font-size: 1.2rem; }}
  h2 {{ font-size: 1rem; margin-top: 1.5rem; }}
  pre {{ background: #f4f4f4; padding: 0.6rem; font-size: 0.75rem; overflow-x: auto; }}
</style>
</head>
<body>
<h1>Comparacion Transformer vs GRU</h1>
<p>Semilla {semilla}, temperatura {TEMPERATURA}</p>

<h2>Transformer</h2>
<div id="partitura-t"></div>
<div id="audio-t"></div>
<pre id="texto-t"></pre>

<h2>GRU</h2>
<div id="partitura-g"></div>
<div id="audio-g"></div>
<pre id="texto-g"></pre>

<script>
const abcT = {abc_t};
const abcG = {abc_g};

function cargar(idP, idA, idX, abc) {{
  document.getElementById(idX).textContent = abc;
  const obj = ABCJS.renderAbc(idP, abc, {{ responsive: "resize" }})[0];
  if (ABCJS.synth.supportsAudio()) {{
    const s = new ABCJS.synth.SynthController();
    s.load("#" + idA, null, {{ displayPlay: true }});
    s.setTune(obj, false, {{ chordsOff: false }});
  }}
}}

cargar("partitura-t", "audio-t", "texto-t", abcT);
cargar("partitura-g", "audio-g", "texto-g", abcG);
</script>
</body>
</html>
"""
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 muestras/generar_muestras.py <semilla>")
        print("Ejemplo: python3 muestras/generar_muestras.py 456")
        sys.exit(1)

    semilla = int(sys.argv[1])
    os.chdir(root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = A.download_nottingham()
    corpus, _ = A.build_corpus_nottingham(data_dir)
    tok = A.CharTokenizer(corpus)

    ckpt_gpt = buscar_checkpoint("gpt")
    ckpt_gru = buscar_checkpoint("gru")
    transformer = GPTLightning.load_from_checkpoint(ckpt_gpt).to(device)
    gru = GRULightning.load_from_checkpoint(ckpt_gru).to(device)

    muestra_t = poner_titulo(generar_mejor(transformer, tok, semilla), "Melodia generada con Transformer")
    muestra_g = poner_titulo(generar_mejor(gru, tok, semilla + 1), "Melodia generada con GRU")

    n_t = sum(p.numel() for p in transformer.parameters())
    n_g = sum(p.numel() for p in gru.parameters())
    print("Transformer:", ckpt_gpt, "| params:", f"{n_t/1e6:.2f}M")
    print("GRU:", ckpt_gru, "| params:", f"{n_g/1e6:.2f}M")
    print("Melodias distintas:", muestra_t != muestra_g)

    dir_muestras = os.path.join(root, "muestras")
    os.makedirs(dir_muestras, exist_ok=True)

    guardar_abc_y_midi(muestra_t, f"{dir_muestras}/muestra_transformer.abc", f"{dir_muestras}/muestra_transformer.mid")
    guardar_abc_y_midi(muestra_g, f"{dir_muestras}/muestra_gru.abc", f"{dir_muestras}/muestra_gru.mid")
    crear_html(semilla, muestra_t, muestra_g, f"{dir_muestras}/comparacion.html")

    print("Listo. Semilla:", semilla)
    print("Abrir: muestras/comparacion.html")
    print("Transformer valida:", A.is_valid_abc(muestra_t), "| chars:", len(muestra_t))
    print("GRU valida:", A.is_valid_abc(muestra_g), "| chars:", len(muestra_g))


if __name__ == "__main__":
    main()
