# RAG-App

Aplicación educativa de RAG para consultar transcripciones de vídeos. Construye embeddings de fragmentos, recupera contexto por similitud vectorial o mediante BM25 más fusión por rango recíproco (RRF), y pide al modelo de lenguaje una respuesta con citas.

## Requisitos e instalación

Se recomienda Python 3.10 o superior. Usa un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Instala PyTorch para tu plataforma y, si vas a usar CUDA, para la versión de controlador/CUDA disponible. El selector oficial de PyTorch da el comando adecuado: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). Después instala el resto:

```powershell
python -m pip install -r requirements.txt
```

La cuantización 4-bit en CUDA requiere `bitsandbytes`, que no se incluye en la lista base porque la instalación depende del sistema operativo, CUDA y PyTorch. Instálalo siguiendo las instrucciones compatibles con tu equipo. Si no está disponible, usa `--device cpu`; el Gemma 2 de 9B sin cuantizar puede requerir mucha memoria. Los pesos de los modelos se descargan al usarlos y se guardan en la caché de Hugging Face.

## Añadir transcripciones

Puedes transcribir audios locales con autodetección de idioma:

```python
from download_channel import transcribe_audios_from_folder

transcribe_audios_from_folder(
    audio_folder="audios",
    transcription_folder="transcriptions",
    language=None,  # Whisper detecta el idioma
    device="auto",
)
```

También puedes descargar y transcribir un canal:

```python
from download_channel import download_and_transcribe_videos_from_youtube_channel

download_and_transcribe_videos_from_youtube_channel(
    channel_url="https://www.youtube.com/@canal/videos",
    output_path="audios/canal",
    transcription_folder="transcriptions/canal",
    max_videos=10,
    language=None,
    device="auto",
)
```

Los archivos incluyen el ID del vídeo en el nombre para evitar colisiones entre títulos parecidos. Las transcripciones nuevas guardan marcas de tiempo por segmento. Los archivos TXT antiguos siguen siendo válidos; sus citas no incluirán marcas temporales si no las contienen.

## Indexar y consultar

Construye o actualiza el índice sin cargar el modelo de chat:

```powershell
python cli_interface.py --index-only --transcriptions transcriptions/canal --index data/canal.parquet
```

Para usar GPT-6 Luna como generador, manteniendo local la recuperación, configura tu clave de OpenRouter y selecciona el backend. `reasoning_effort` queda en `low` por defecto. La llamada usa el modelo `openai/gpt-6-luna` y se descuenta del mismo saldo de OpenRouter que las transcripciones. El coste devuelto por OpenRouter se muestra en consola; si la respuesta no incluye coste, se imprime una estimación conservadora:

```powershell
python cli_interface.py --llm-backend openrouter --llm-model openai/gpt-6-luna --reasoning-effort low
```

Para comparar transcriptores con una muestra controlada, consulta la sección «Comparar transcriptores con una muestra controlada» más abajo.

La firma del índice incluye los hashes de las transcripciones, el modelo de embeddings y los parámetros de división. Una ejecución posterior reutiliza el índice si esas entradas no cambiaron. Para forzar la reconstrucción, añade `--rebuild-index`.

Inicia el chat con retrieval vectorial (modo predeterminado):

```powershell
python cli_interface.py --transcriptions transcriptions/canal --index data/canal.parquet
```

El chat conserva en memoria los últimos cuatro turnos para entender preguntas de seguimiento como «¿Y qué propone?». Antes de recuperar, una regla local detecta referencias probables y solo entonces reformula la consulta con el backend/modelo elegido; la respuesta sigue usando únicamente evidencia recuperada en el turno actual y conserva sus citas. Las reformulaciones de OpenRouter usan el mismo modelo y `reasoning_effort` configurados, con un máximo de 96 tokens de salida; el coste se muestra como una llamada separada y no hay llamadas de reescritura para consultas autónomas. Si la reformulación falla, se busca con la pregunta original y no se reintenta. Usa `/clear` para borrar el historial de la sesión o `--history-turns 0` para desactivar esta función. El historial vive solo mientras el proceso del CLI está abierto. Con backend OpenRouter, el historial reciente también se envía en las llamadas de reformulación y respuesta; con backend local, se queda en el equipo.

Prueba retrieval híbrido con BM25 y embeddings, seguido de RRF:

```powershell
python cli_interface.py --retrieval-mode hybrid --transcriptions transcriptions/canal --index data/canal.parquet
```

El reranking con cross-encoder es opcional y solo carga el modelo si se especifica:

```powershell
python cli_interface.py --retrieval-mode hybrid --reranker-model BAAI/bge-reranker-v2-m3
```

También puedes elegir Jev de OpenRouter como reranker del top de la recuperación híbrida:

```powershell
python cli_interface.py --retrieval-mode hybrid --reranker jev --llm-backend openrouter
```

Jev recibe la consulta y, por defecto, hasta 9 fragmentos fusionados por BM25+vectores. Hace como máximo una petición por consulta; una estimación conservadora por encima de `$0.001` detiene esa petición y mantiene el orden RRF. Si Jev falla, la consulta continúa con RRF y muestra un aviso. El límite es un preflight estimado, no una garantía de facturación. Las puntuaciones se guardan en `.rag-app-cache/jev-reranker-v1.json` bajo hashes, sin conservar consultas, fragmentos, respuestas del API ni la clave; las consultas repetidas usan la caché y no cuestan. Borrar esa caché permite solicitar de nuevo puntuaciones ya almacenadas.

Al activar esta opción se envían la consulta y los fragmentos candidatos a OpenRouter. En la evaluación de 24 consultas Jev costó `$0.002365` en total (aprox. `$0.0000985` por consulta). Su puntuación solo reordena candidatos; no sustituye el umbral vectorial de evidencia ni decide por sí sola si una respuesta puede contestarse. Jev y el cross-encoder local son opciones excluyentes.

Opciones útiles:

- `--device auto|cuda|cpu`: elige el dispositivo. Si CUDA no está disponible, el código cae a CPU.
- `--top-k 5`: fragmentos finales a usar.
- `--history-turns 4`: pares recientes de pregunta/respuesta que se conservan para seguimientos; `0` desactiva el historial.
- `--candidate-k 20`: candidatos de BM25/vector antes de la fusión.
- `--reranker-candidate-k 9`: máximo de candidatos fusionados enviados al reranker.
- `--jev-max-estimated-cost-usd 0.001`: estimación máxima por consulta antes de llamar a Jev.
- `--min-evidence-score 0.35`: umbral configurable de similitud vectorial para abstenerse. Debe calibrarse con el corpus real; no es una garantía semántica.
- `--llm-model ID` y `--embedding-model ID`: modelos alternativos disponibles localmente o en Hugging Face.

La aplicación exige citas que coincidan con fuentes recuperadas y se abstiene cuando no supera el umbral de evidencia. La verificación semántica adicional está desactivada por defecto. Actívala con `--check-faithfulness`; cada respuesta pasa entonces por GPT-6 Luna `low` en OpenRouter, incluso si el generador principal es local:

```powershell
python cli_interface.py --check-faithfulness --transcriptions transcriptions/canal --index data/canal.parquet
```

Esta opción envía la respuesta (hasta 4.000 caracteres) y hasta 8.000 caracteres de evidencia recuperada a OpenRouter. Si se superan esos límites, se abstiene sin llamar al verificador. El coste adicional suele ser aproximadamente **$0.0002–$0.001 por respuesta**, según longitud y tarifas vigentes, y se descuenta del mismo saldo. El verificador acepta solo un veredicto JSON explícito; si falla, falta o no se puede interpretar, la aplicación se abstiene. Sin esta opción, se comprueba el formato y la procedencia de las citas, pero no el apoyo semántico de cada afirmación. Para desactivarla, omite `--check-faithfulness`.

## Evaluación y tests

```powershell
python -m unittest discover -v
python -m tests.retrieval_benchmark
```

El benchmark contiene cinco documentos, tres consultas y relevancia graduada escrita para pruebas deterministas. Sus vectores son puntuaciones manuales y su reranker es un stub de solapamiento léxico; esos resultados comprueban el funcionamiento de BM25, RRF y las métricas, pero no predicen la calidad del modelo real. Para evaluar el sistema con calidad representativa hace falta añadir consultas etiquetadas sobre las transcripciones del canal.

`talk_with_db.py` se conserva como entrada compatible y delega en la CLI compartida.
# Comparar transcriptores con una muestra controlada

La comparación usa un único audio y su referencia corregida para calcular WER y CER, además de latencia y coste devuelto por OpenRouter. La referencia permanece local; el audio se envía una vez a cada modelo. Las salidas de transcripción no se guardan. El resultado resumido se añade a `MEASUREMENTS.md`.

Modelos incluidos: `microsoft/mai-transcribe-2`, `qwen/qwen3-asr-0.6b` y `openai/whisper-large-v3-turbo`. El coste estimado previo usa una instantánea de precios del 2026-09-23 ([MAI](https://openrouter.ai/microsoft/mai-transcribe-2), [Qwen](https://openrouter.ai/qwen/qwen3-asr-0.6b), [Whisper](https://openrouter.ai/openai/whisper-large-v3-turbo)); la respuesta de OpenRouter aporta el coste registrado cuando está disponible. Ejecuta la prueba solo sobre un fragmento representativo, no sobre el corpus entero.

La tarifa publicada de [GPT-6 Luna en OpenRouter](https://openrouter.ai/openai/gpt-6-luna) se consulta aparte; cualquier cambio de precio requiere actualizar la estimación local.

```powershell
$env:OPENROUTER_API_KEY = "<tu clave>"
python transcription_benchmark.py `
  --audio .\sample.mp3 `
  --reference .\sample-reference.txt `
  --duration-seconds 300 `
  --max-estimated-cost-usd 0.02
```

El idioma predeterminado es español. La duración debe ser la real; el programa limita por defecto la muestra a 5 minutos, el coste estimado a $0.02 y el archivo a 25 MiB. El coste es una estimación de preflight, no un límite de facturación garantizado. La clave se puede definir en el entorno como `OPENROUTER_API_KEY` o `OPENROUTER_APIKEY`, o en un `.env` local con cualquiera de esos nombres. `.env` está excluido de Git; no publiques ni compartas su contenido.

Para probar solo Qwen3 ASR 1.7B, añade `--models qwen/qwen3-asr-1.7b`; los tres modelos de la comparación base siguen siendo los predeterminados.
