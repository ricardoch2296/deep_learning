# Guion de defensa

Notas para defender el proyecto. No es para entregar; es para prepararme.

## Resumen en una frase
Entrene un mini-GPT (Transformer decoder char-level, 4.83M params) para generar tunes en ABC notation con datos de thesession.org, y lo compare contra un GRU (0.74M). El GPT gana en perplexity (2.44 vs 2.93); ambos generan 100% de tunes validos.

## Decisiones tecnicas (y por que)
- **ABC como texto -> modelado de lenguaje char-level.** Un tune es texto, generar = predecir el siguiente caracter. Por eso uso cross-entropy sobre el siguiente token.
- **Char-level (no sub-palabras).** El vocabulario de ABC es chico (116 chars); tokenizar por caracter es lo mas simple y evita un tokenizer extra.
- **GPT vs GRU.** Comparo las dos familias del curso: recurrencia (GRU) vs atencion (GPT). El GPT ve todo el contexto a la vez con self-attention + positional encoding; el GRU arrastra un estado. Espero que el GPT capture mejor dependencias largas (tonalidad, compas, forma AABB).
- **Contexto 256.** Alcanza para un tune completo, asi el modelo respeta la cabecera (M:, K:) al generar el cuerpo.
- **Dropout 0.1 / 0.2.** Para no memorizar y generar melodias nuevas.
- **Entreno por steps (5000), no epochs.** El corpus tiene 9.2M ventanas; recorrer epochs completos es innecesario.
- **Muestreo con temperatura + top-k.** Controlo la aleatoriedad; top-k evita elegir caracteres improbables.

## Resultados clave
- GPT: test loss 0.894, perplexity 2.44, 100% validos.
- GRU: test loss 1.075, perplexity 2.93, 100% validos.
- Ablation: la validez baja con temperatura alta (100% hasta 0.9, 90% en 1.3).
- Las curvas train/val van juntas: no hay sobreajuste fuerte.

## Preguntas probables y respuestas
- **Por que el GPT gana?** Menor perplexity = predice mejor el siguiente caracter. La atencion accede a todo el contexto sin el cuello de botella del estado recurrente.
- **Por que ambos 100% validos?** Aprender la sintaxis de ABC es facil para las dos redes; la validez sintactica no mide calidad musical. La diferencia real esta en perplexity.
- **Como mido "que suene bien"?** Uso validez sintactica con music21 (parseable, con notas) y export a MIDI para escuchar. No mido calidad musical objetiva (limitacion).
- **Que es la perplexity?** exp(loss); intuitivamente, entre cuantos caracteres "duda" el modelo en promedio. Mas baja = mejor.
- **Que hace la mascara causal?** En el GPT, impide que la posicion t vea el futuro (>t); asi entrena a predecir sin trampa.
- **Por que positional encoding?** La atencion no tiene orden; hay que inyectar la posicion de cada caracter.
- **Overfitting?** Poco: val sigue a train (ver figura). Uso dropout y guardo el mejor checkpoint por val_loss.

## Limitaciones
- Validez sintactica != calidad musical.
- Modelos pequenos; sin condicionamiento por tipo/tonalidad.
- Dataset filtrado a tunes < 400 caracteres.

## Mejoras futuras
- Tokenizacion por sub-unidades musicales.
- Modelo mas grande o mas contexto.
- Condicionar la generacion (tipo de tune, tonalidad) y evaluar con juicio humano.
