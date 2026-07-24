# Generador de musica en ABC notation

Proyecto final de Deep Learning (USFQ). Un **mini-GPT** (Transformer decoder a nivel de caracter) que genera melodias en [ABC notation](https://abcnotation.com/), comparado contra un **GRU** como baseline.

- **Dataset:** [thesession.org](https://github.com/adactio/thesession-data) (`tunes.csv`, ~55k tunes irlandeses)
- **Tarea:** modelado de lenguaje causal (predecir el siguiente caracter)
- **Framework:** PyTorch + PyTorch Lightning (GPT2 de HuggingFace)
- **Comparacion:** loss, perplexity y % de tunes musicalmente validos (music21)

## Estructura

```
deep_learning/
  notebooks/abc_gpt.ipynb   # notebook principal (todo el flujo)
  src/abc_utils.py          # datos, tokenizador y validacion con music21
  informe/                  # articulo en LaTeX
  samples/                  # tunes generados (.abc, .mid)
```

## Como ejecutar

En el servidor con GPU (H200), con el entorno del curso:

```bash
conda activate usfq
pip install -r requirements.txt   # instala music21 si falta
jupyter notebook notebooks/abc_gpt.ipynb
```

El notebook descarga los datos, entrena el GPT y el GRU, evalua y genera tunes. El entrenamiento va por `MAX_STEPS` (no por epochs) para controlar el tiempo.

## Reproducibilidad

- Semilla fija (`seed_everything(42)`).
- Hiperparametros centralizados en una sola celda del notebook.
- Los tunes generados se guardan en `samples/` y las figuras en `informe/figs/`.
