# Guion de defensa

Notas para defender el proyecto. No es para entregar; es para prepararme.

## Resumen en una frase
Entrene un mini-GPT (Transformer decoder char-level, 4.86M params) para generar tunes en ABC notation **con acordes** usando el Nottingham Music Database, y lo compare contra un GRU (0.72M). El GPT gana en perplexity (2.98 vs 3.21); ambos superan el 99% de tunes validos.

## Decisiones tecnicas (y por que)
- **ABC como texto -> modelado de lenguaje char-level.** Un tune es texto, generar = predecir el siguiente caracter. Por eso uso cross-entropy sobre el siguiente token.
- **Char-level (no sub-palabras).** El vocabulario de ABC es chico (86 chars); tokenizar por caracter es lo mas simple y evita un tokenizer extra.
- **Dataset Nottingham (con acordes).** A diferencia de un corpus monofonico, cada tune trae simbolos de acorde (`"G"`, `"D7"`) sobre la melodia; asi el modelo aprende a generar melodia + armonia y los acordes suenan como acompanamiento.
- **GPT vs GRU.** Comparo las dos familias del curso: recurrencia (GRU) vs atencion (GPT). El GPT ve todo el contexto a la vez con self-attention + positional encoding; el GRU arrastra un estado. Espero que el GPT capture mejor dependencias largas (tonalidad, compas, forma AABB).
- **Contexto 384.** Alcanza para un tune completo (con acordes son mas largos), asi el modelo respeta la cabecera (M:, K:) al generar el cuerpo.
- **Dropout 0.1 / 0.2.** Para no memorizar y generar melodias nuevas.
- **Entreno por steps (8000), no epochs.** Controlo el tiempo sin recorrer epochs completos.
- **Muestreo con temperatura 0.6 + top-k.** Temperatura baja = menos aleatorio y mas coherente; top-k evita elegir caracteres improbables.

## Resultados clave
- GPT: test loss 1.093, perplexity 2.98, 99% validos.
- GRU: test loss 1.165, perplexity 3.21, 100% validos.
- Ablation: la validez baja con temperatura alta (100% hasta 0.7, 92% en 0.9, 56% en 1.3). Por eso genero con 0.6.
- Las curvas train/val van juntas: no hay sobreajuste fuerte.

## Preguntas probables y respuestas
- **Por que el GPT gana?** Menor perplexity = predice mejor el siguiente caracter. La atencion accede a todo el contexto sin el cuello de botella del estado recurrente.
- **Por que ambos ~99-100% validos?** Aprender la sintaxis de ABC es facil para las dos redes; la validez sintactica no mide calidad musical. La diferencia real esta en perplexity.
- **Como suenan los acordes?** El tune trae simbolos de acorde entre comillas; music21 los convierte en acordes reales (`writeAsChord=True`) para el MIDI, y en el `preview.html` abcjs los toca como acompanamiento.
- **Por que la perplexity subio vs un corpus monofonico?** Porque ahora el modelo tambien predice los acordes, que anaden incertidumbre; es un problema mas dificil.
- **Como mido "que suene bien"?** Uso validez sintactica con music21 (parseable, con notas) y export a MIDI para escuchar. No mido calidad musical objetiva (limitacion).
- **Que es la perplexity?** exp(loss); intuitivamente, entre cuantos caracteres "duda" el modelo en promedio. Mas baja = mejor.
- **Que hace la mascara causal?** En el GPT, impide que la posicion t vea el futuro (>t); asi entrena a predecir sin trampa.
- **Por que positional encoding?** La atencion no tiene orden; hay que inyectar la posicion de cada caracter.
- **Overfitting?** Poco: val sigue a train (ver figura). Uso dropout y guardo el mejor checkpoint por val_loss.

## Limitaciones
- Validez sintactica != calidad musical.
- Modelos pequenos y corpus chico (992 tunes); sin condicionamiento por tipo/tonalidad.
- Dataset filtrado a tunes de 60-700 caracteres y solo los que tienen acordes.

## Mejoras futuras
- Tokenizacion por sub-unidades musicales.
- Modelo mas grande o mas contexto.
- Condicionar la generacion (tipo de tune, tonalidad) y evaluar con juicio humano.
