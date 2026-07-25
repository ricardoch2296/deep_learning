# Generador de musica en ABC notation

Proyecto final de Deep Learning (USFQ). Un **mini-GPT** (Transformer decoder a nivel de caracter) que genera melodias en [ABC notation](https://abcnotation.com/), comparado contra un **GRU** como baseline.

- **Dataset:** [Nottingham Music Database](https://github.com/jukedeck/nottingham-dataset) (992 tunes folk en ABC **con acordes**: `"G"`, `"D7"`, ...)
- **Tarea:** modelado de lenguaje causal (predecir el siguiente caracter)
- **Framework:** PyTorch + PyTorch Lightning (GPT2 de HuggingFace)
- **Comparacion:** loss, perplexity y % de tunes musicalmente validos (music21)

## Resultados

| Modelo | params | test loss | perplexity | % validos |
|--------|--------|-----------|------------|-----------|
| GPT    | 4.86M  | 1.093     | 2.98       | 99        |
| GRU    | 0.72M  | 1.165     | 3.21       | 100       |

El GPT logra menor perplexity (mejor modelado de la secuencia); ambos superan el 99% de tunes parseables a temperatura 0.6. El modelo genera melodia **con acordes de acompanamiento**. Entrenado en una GPU NVIDIA H200 (~10 min).

## Estructura

```
deep_learning/
  notebooks/abc_gpt.ipynb   # notebook principal (entrenamiento + evaluacion)
  src/abc_utils.py          # datos, tokenizador y validacion
  checkpoints/              # modelos entrenados (.ckpt) — local, no en git
  logs/                     # metricas de entrenamiento
  muestras/                 # melodias generadas + comparacion.html
  informe/                  # articulo en LaTeX
```

## Como ejecutar

En el servidor con GPU (H200), con el entorno del curso:

```bash
conda activate usfq
pip install -r requirements.txt   # instala music21 si falta
jupyter notebook notebooks/abc_gpt.ipynb
```

El notebook descarga los datos, entrena el Transformer y el GRU, evalua y genera melodias.

Para **generar otra comparacion** (sin reentrenar), con los checkpoints en `checkpoints/`:

```bash
conda activate usfq
python muestras/generar_muestras.py 456
```

Abrir `muestras/comparacion.html` en el navegador.

## Reproducibilidad

- Semilla fija (`seed_everything(42)`).
- Hiperparametros centralizados en una sola celda del notebook.
- Los tunes generados se guardan en `muestras/` y las figuras en `informe/figs/`.
- Para escuchar y comparar: abrir `muestras/comparacion.html` (se genera con `generar_muestras.py`).
