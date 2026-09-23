# RAG-App measurement log

## Measurement protocol

All measurements belong to this checkout and are taken with the same available runtime, from the repository root. The command available for Python is the Codex-bundled Python 3.12.14 executable (the `python` and `py` commands are not on PATH). No dependencies were installed and no external model or transcript data was downloaded. Measurements that need project dependencies, GPU hardware, model weights, or a representative corpus are marked blocked rather than estimated.

Baseline was captured before implementation. After each implementation block, rerun the import/startup, unit/syntax, and applicable deterministic retrieval/index/prompt checks with the same commands and inputs. Record changes below.

## Baseline — before code changes

Environment: Windows; bundled Python 3.12.14. Module probes found `pandas` and `numpy`; `torch`, `transformers`, `langchain_huggingface`, `langchain_community`, `langchain_text_splitters`, `pyarrow`, `nltk`, `pytubefix`, and `faster_whisper` were absent. No model directories for BGE-M3 or Gemma were found in the default Hugging Face cache. `nvidia-smi` was unavailable; CUDA and GPU memory therefore could not be measured. The expected `transcriptionsBorjaBandera` corpus and `vector_db.parquet` were absent.

- Existing automated tests: `python -m unittest discover -v` reported 0 tests.
- Static syntax: AST parsing succeeded for all 5 repository Python files.
- Imports: `vector_db_manager`, `llm_interaction`, `cli_interface`, and `talk_with_db` each failed with `ModuleNotFoundError: No module named 'torch'`. In this single baseline run, measured failed-import times were 0.005220 s, 0.009705 s, 0.004726 s, and 0.005592 s respectively. These are failure timings, not working startup latency.
- CLI startup: `cli_interface.py` failed at the same missing `torch` dependency in 0.027135 s. This does not measure model loading or a usable app start.
- Retrieval quality and latency on representative data: blocked; both the embedding/runtime dependencies and local transcript/index data are unavailable.
- Model load time, inference latency, GPU/CPU memory, and real answer quality: blocked; PyTorch, model weights, and GPU probe are unavailable.
- Device portability and actual Parquet persistence: blocked in this environment because project dependencies are absent. These will receive deterministic component tests where possible.

Reproduction commands (use the bundled Python executable reported by the workspace dependency runtime):
- `python -m unittest discover -v`
- Import/startup probe: time imports of `vector_db_manager`, `llm_interaction`, `cli_interface`, and `talk_with_db`, then run `cli_interface.py` with `runpy`; record exception and elapsed time.
- Syntax check: AST-parse each root `*.py` file.
- Environment probe: inspect module availability, presence of `nvidia-smi`, expected corpus/index paths, and cached model directories. Do not print environment variables or credentials.

## After implementation blocks

### Block 1 — imports, device handling, empty/small retrieval, single embedding instance

Completed; 7/7 deterministic tests passed. Import safety, device fallback, top_k bounds, and empty-corpus behavior are covered; the CLI timing caveat is in the chronological log.

### Block 2 — persistent/versioned index, batched embeddings, source metadata

Completed; 12/12 deterministic tests passed. Fingerprint/cache reuse and 32/32/1 embedding batches were measured with test doubles; real Parquet and model latency remain unverified.

### Block 3 — cited/grounded prompts, generation decoding, CLI/setup/download workflow

Completed; final suite 32/32 passed. Hybrid BM25/RRF/evaluation, citations, and evidence gating are covered; real cross-encoder ranking and semantic validation remain pending.


### Block 1 — imports, device handling, empty/small retrieval, single embedding instance

Same bundled Python 3.12.14 runtime; same absent project dependencies and corpus as baseline.

- Imports: all four modules now import successfully without loading `torch`, `nltk`, or `transformers`. One-run import times: vector manager 0.032951 s, interaction 0.000567 s, CLI 0.003879 s, compatibility entry point 0.000423 s. These are import-only timings, not model startup.
- CLI: `--help` completed in 0.1150476 s. This is not directly comparable with the baseline CLI run (0.027135 s), which terminated early on missing `torch`; the new normal CLI reaches the expected missing-corpus check in 0.014903 s.
- Tests: `python -m unittest discover -v` ran 7 deterministic tests; all 7 passed. Coverage includes import side effects, automatic/fallback device selection, zero/small/negative top_k bounds, and prompt escaping.
- Syntax: AST parsing succeeded for 7 Python files (5 root files plus 2 test files).
- Retrieval on real embeddings, GPU/memory, model loading, Parquet, and corpus quality remain blocked for the baseline reasons above. top_k behavior is component-tested; a PyTorch-backed ranking run has not been measured.



### Block 2 — versioned index, batching, and metadata

- Tests: `python -m unittest discover -v` ran 12 tests; all passed.
- Syntax: AST parsing succeeded for 8 Python files.
- Imports remain side-effect free: all four app modules import with no optional ML/NLTK modules loaded. One-run import times: vector manager 0.059889 s, interaction 0.000714 s, CLI 0.004086 s, compatibility entry point 0.000610 s. These small import-only samples are noisy and do not measure inference startup.
- CLI `--help`: 0.1071043 s in this run.
- Cache behavior (deterministic temporary-corpus test): first unchanged corpus/configuration builds once; second run loads the cached table without calling the build path (build count remains 1); editing a transcript invalidates the fingerprint and rebuilds (count becomes 2). The test uses a fake Parquet marker/DataFrame because `pyarrow` is unavailable; actual Parquet read/write remains unverified.
- Batching (synthetic 65 chunks with a deterministic fake embedder): 3 document-batch calls of sizes 32, 32, and 1. The original implementation issued 65 `embed_query` calls for 65 chunks. This is an API-call-count comparison only, not measured model latency or throughput.
- Actual embedding, GPU memory, representative corpus retrieval, and Parquet performance remain blocked by missing dependencies, hardware probe, weights, corpus, and `pyarrow`.

### Hybrid retrieval fixture baseline — before adding hybrid retrieval code

This is a deterministic mechanical benchmark, not production retrieval quality. The vector rankings use fixed hand-authored scores (not BGE output); BM25/RRF and the lexical reranker stub are pure-Python reference calculations. There are 5 short documents and 3 queries with graded gold relevance; all metrics are macro averages at k=3. Recall counts documents with grade > 0; MRR uses the first relevant result; nDCG uses gain `2^grade - 1`. The stub reranker counts query/document token overlap and preserves candidate order on ties. It is not a cross-encoder.

Synthetic ranking fixture:
- q1 `insulina glucosa`, grades D1=3, D2=2, D4=1; vector order D4,D1,D2,D3,D5.
- q2 `ejercicio sensibilidad insulina`, grades D2=3, D1=1; vector order D4,D2,D1,D3,D5.
- q3 `memoria concentración`, grade D5=3; vector order D5,D3,D1,D2,D4.

| Method | Recall@3 | MRR | nDCG@3 |
| --- | ---: | ---: | ---: |
| BM25 reference | 1.0000 | 1.0000 | 1.0000 |
| Fixed-vector ranking | 1.0000 | 0.8333 | 0.7936 |
| Reciprocal-rank fusion (BM25 + fixed vector) | 1.0000 | 1.0000 | 0.9850 |
| Lexical overlap reranker stub over fused top 3 | 1.0000 | 1.0000 | 1.0000 |

These results only check deterministic ranking mechanics on this fixture. The repository did not have BM25, fusion, or reranking before this feature, and no semantic or real cross-encoder quality is inferred from these numbers. The fixture and metric calculation will be checked in and rerun after implementation; real-corpus evaluation remains pending.



## Registro de avances

Hora de referencia para este registro: 2026-09-22 23:10 CEST (21:10 UTC). Las primeras dos mediciones se registraron retrospectivamente; no se instrumentó la hora exacta de ejecución en esas pasadas. Los resultados y comandos anotados son los observados.

### 2026-09-22 23:10 CEST — Bloque 1: arranque y límites de recuperación

- Objetivo: retirar descargas/cargas en importación, seleccionar CPU/CUDA, limitar `top_k`, compartir una instancia de embeddings y hacer explícitas las rutas del CLI.
- Archivos: `.gitignore`, `vector_db_manager.py`, `llm_interaction.py`, `cli_interface.py`, `talk_with_db.py`, `tests/test_core_behaviors.py`.
- Comandos: bundled Python 3.12.14, `-m unittest discover -v`; AST parse; probe de importaciones; `cli_interface.py --help`.
- Antes: cuatro imports y el CLI fallaban por `torch` ausente; 0 tests; AST 5 archivos. Después: imports sanos y sin cargar `torch`, `nltk` o `transformers`; CLI help 0.115 s (no comparable directamente con el fallo previo de 0.027 s); ahora el arranque normal alcanza el error explícito de corpus ausente. 7/7 tests; AST 7 archivos.
- Límite: retrieval real aún no ejecutable sin PyTorch, modelo ni corpus.
- Siguiente: reutilizar índice y verificar invalidación/batching.

### 2026-09-22 23:10 CEST — Bloque 2: índice persistente y embeddings por lotes

- Objetivo: invalidar el índice por contenido/configuración, reutilizarlo en una segunda ejecución, persistirlo atómicamente y guardar metadatos de fuente.
- Archivos: `vector_db_manager.py`, `cli_interface.py`, `tests/test_index_cache.py`.
- Comandos: bundled Python 3.12.14, `-m unittest discover -v`; AST parse; import probe; `cli_interface.py --help`.
- Antes: cada arranque invocaba indexación; 65 fragmentos daban 65 llamadas `embed_query`. Después: prueba de corpus temporal muestra build count 1 tras primera ejecución, sigue en 1 tras repetición idéntica y sube a 2 tras editar el texto; embedder falso con 65 fragmentos realiza 3 llamadas por lotes (32/32/1); 12/12 tests; AST 8 archivos; help 0.107 s.
- Límite: el cache test simula Parquet porque falta `pyarrow`; no se midió tiempo real de embeddings.
- Siguiente: implementar retrieval híbrido y métricas reproducibles.

### 2026-09-22 23:10 CEST — Baseline híbrida y bloque 3: BM25, RRF y evaluación

- Objetivo: medir el mismo fixture con BM25, vector, RRF y reranker stub antes/después; añadir ranking híbrido opt-in, métricas y tests sin importar modelos adicionales.
- Archivos: `hybrid_retrieval.py`, `retrieval_evaluation.py`, `llm_interaction.py`, `cli_interface.py`, `tests/retrieval_fixture.py`, `tests/retrieval_benchmark.py`, `tests/test_hybrid_retrieval.py`, `tests/test_core_behaviors.py`.
- Comandos: bundled Python 3.12.14, `-m tests.retrieval_benchmark`; `-m unittest discover -v`.
- Antes de agregar el código híbrido, el fixture de 5 documentos/3 queries midió (Recall@3, MRR, nDCG@3): BM25 reference 1.000/1.000/1.000; vector con puntuaciones manuales 1.000/0.8333/0.7936; RRF 1.000/1.000/0.9850; reranker lexical stub 1.000/1.000/1.000.
- Después, las mismas consultas y gold labels dan: BM25 1.000/1.000/1.000; vector fijo 1.000/0.8333/0.7936; RRF 1.000/1.000/0.9907; stub lexical 1.000/1.000/1.000. El cambio de RRF viene del filtrado de resultados BM25 con score cero; es un resultado de fixture diminuto, no evidencia de una mejora general.
- Verificación: 24/24 tests pasan, incluyendo RRF, top_k, cross-encoder stub, Recall@k/MRR/nDCG, IDs de cita, rechazo de citas desconocidas, evidencia insuficiente y el hook de verificación semántica opt-in.
- Límites: los vectores son scores manuales, no embeddings BGE. El reranker es solapamiento léxico, no un cross-encoder. No se ha validado calidad real ni faithfulness semántica; el hook solo la marca verificada si se proporciona un comprobador. Corpus/modelos/GPU ausentes siguen bloqueando evaluación real.
- Siguiente: terminar el manejo de idioma, IDs/tiempos en transcripciones, dependencias e instrucciones; volver a medir el mismo benchmark y revisar diff/estado sin tocar el staging de `PLAN.md`.

### 2026-09-22 23:24 CEST — Bloque 4 y cierre: citas, abstención, configuración y revisión final

- Objetivo: cerrar fuentes/citas, abstención por evidencia insuficiente, opciones/configuración, guía y manejo de transcripciones; repetir pruebas, benchmark y controles del repositorio.
- Archivos: llm_interaction.py, cli_interface.py, download_channel.py, requirements.txt, README.md, tests, además de la bitácora.
- Comandos: bundled Python 3.12.14, unittest discover -v; tests.retrieval_benchmark; AST parse de todos los .py; import probe; cli_interface.py --help; git diff --check; comprobación de git status --short, git diff --cached --name-status y git diff --exit-code -- PLAN.md.
- Resultado final: 32/32 tests pasan. AST correcto en 14 archivos; 6 imports limpios y torch, nltk, transformers, sentence_transformers no se cargan; --help exit 0 en 0.159535 s (muestra única, sensible al entorno). El benchmark repite, sobre las mismas 3 consultas y gold labels: BM25 Recall@3/MRR/nDCG@3 = 1/1/1; vector fijo = 1/0.8333/0.7936; RRF = 1/1/0.9907; reranker lexical stub = 1/1/1.
- Citas incorporan source ID, título/ruta, enlace y timestamp disponible; la app rechaza referencias no presentes y puede abstenerse si no hay evidencia o el score queda bajo el umbral configurable. El hook de faithfulness es opt-in. Sin un verificador/modelo semántico real, los checks aquí prueban referencias y gating, no que cada afirmación esté semánticamente sustentada.
- Revisión de Git: git diff --check limpio; PLAN.md sigue como única entrada staged (A PLAN.md) y su contenido coincide con el índice. Sin commit.
- Límites pendientes: evaluación de calidad y latencia con corpus real, embeddings, cross-encoder real, uso GPU/memoria, lectura/escritura Parquet, y faithfulness semántica. Entorno sin torch/transformers/pyarrow, pesos, corpus de transcripciones ni GPU detectable; la evaluación cuantitativa disponible es solo el fixture determinista documentado.
- Siguiente paso: cuando haya corpus, pesos y dependencias, ejecutar benchmark real de recuperación y latencia en CPU/GPU; calibrar el umbral de evidencia y conectar un comprobador semántico para medir faithfulness.

### 2026-09-23 — Preparación de comparativa de transcripción

- Añadido `transcription_benchmark.py` para una llamada por modelo a MAI-Transcribe 2, Qwen3 ASR 0.6B y Whisper Large V3 Turbo, usando el mismo audio, idioma y referencia local.
- Métricas: WER, CER, latencia y coste devuelto por OpenRouter (o estimación de catálogo si la respuesta no informa coste). Las salidas de audio y texto no se escriben; el registro guarda solo nombre de muestra, duración y métricas.
- Límite preventivo: 15 minutos, 25 MiB y $0.10 estimados por ejecución. La estimación usa precios de catálogo consultados el 2026-09-23 y no sustituye el coste real de cada respuesta.
- No se enviaron solicitudes ni se consumió saldo: el repo no contiene audios/transcripciones de referencia y `OPENROUTER_API_KEY` no está configurada en el entorno disponible.
- Pendiente: recibir una muestra representativa con referencia corregida y configurar la clave como variable de entorno para medir resultados reales.
- La primera implementación de GPT-6 Luna se hizo directa con OpenAI; el usuario aclaró que también debía usar OpenRouter. Se corrigió en la entrada siguiente antes de realizar ninguna llamada.
- Verificación local tras la primera integración (ruta directa OpenAI, luego sustituida): `python -m unittest discover -v` — 32/32 tests pasaron; `cli_interface.py --help` muestra el backend/effort; AST parse correcto. Un mock local confirmó que el payload incluía `effort=low`; no se hizo ninguna petición HTTP. La verificación final de la ruta OpenRouter se registra abajo.

### 2026-09-23 — GPT-6 Luna enrutado por OpenRouter

- Verificado en el catálogo actual el modelo `openai/gpt-6-luna`, con tarifa de $0.10/M tokens de entrada y $0.50/M de salida.
- El backend ahora llama a `https://openrouter.ai/api/v1/chat/completions` con `reasoning_effort=low` y `OPENROUTER_API_KEY`; queda dentro del saldo cargado en OpenRouter.
- El coste reportado por OpenRouter se imprime en consola; si no hay un coste en la respuesta, calcula una estimación conservadora con los tokens y tarifas actuales. La generación limita la salida a 256 tokens por defecto.
- Verificación tras la corrección: 32/32 tests existentes pasaron; CLI help muestra `local|openrouter`; mock HTTP confirmó endpoint `/chat/completions`, slug `openai/gpt-6-luna`, `reasoning_effort=low`, `max_tokens=256` y lectura del coste. `git diff --check` limpio.
- No se hizo ninguna llamada real: el entorno no tiene `OPENROUTER_API_KEY` ni un audio/referencia disponibles. La prueba de transcriptores y una consulta RAG real siguen pendientes.

### 2026-09-23 — Carga local de clave OpenRouter

- El usuario indicó que añadió `OPENROUTER_APIKEY` a `.env` en la raíz de RAG-App. Se confirmó la presencia del archivo y del nombre de la variable sin leer ni mostrar el valor.
- `api_config.py` carga `OPENROUTER_API_KEY` o el alias `OPENROUTER_APIKEY` desde el entorno o `.env`; el backend RAG y el comparador comparten el mismo cargador.
- Se añadió una regla explícita para ignorar `.env` en Git. No se hizo ninguna llamada a OpenRouter ni se consumió saldo.
- Pendiente para ejecutar la comparación: audio representativo con transcripción corregida. El programa envía una llamada a cada modelo únicamente cuando se invoque con esos archivos.

### 2026-09-23 — Verificación de clave local y límite de gasto

- La carga local reconoce `OPENROUTER_APIKEY` en `.env` (se comprobó solo que el valor está disponible; nunca se mostró ni se registró).
- `git check-ignore .env` confirma que el archivo queda fuera de Git; `git diff --check` no detecta errores de espacios.
- Verificación local ya ejecutada: `python -m unittest discover -v` — 32/32 tests pasan; `cli_interface.py --help` y la ayuda del comparador responden correctamente. No se hizo ninguna petición HTTP ni se consumió saldo.
- El comparador requiere audio y transcripción de referencia, limita la muestra a 15 minutos y estima el coste total antes de enviar una petición; el límite se basa en precios de catálogo y no puede garantizar el coste facturado real. Para no gastar hasta disponer de la muestra, queda pendiente toda llamada real.
- A petición del usuario de extremar el cuidado con un saldo de $7, se bajaron los valores por defecto a 5 minutos y $0.02 de coste estimado para las tres llamadas. Estos son límites previos basados en catálogo; OpenRouter no garantiza que el coste real no los supere. La muestra larga puede requerir un override explícito del CLI. No se enviaron solicitudes.
- Tras el ajuste: valores por defecto del parser comprobados (`300` s, `$0.02`); `python -m unittest discover -v` — 32/32 tests pasan en 0.050 s. La suite no contactó OpenRouter.

### 2026-09-23 — Comparativa STT con clip de YouTube

- Fuente: [vídeo de YouTube](https://www.youtube.com/watch?v=XVwQCT15K3U), “Ceuta: los documentos que desmontan al Gobierno”. Se recortó `00:00:00–00:04:57.768` (297.77 s); WebM/Opus 4.23 MiB y WAV mono/16 kHz 9.53 MB.
- Referencia: subtítulos automáticos españoles `es-orig`; se conservaron solo captions completos dentro del corte y se eliminaron solapamientos de frases consecutivas. El VTT original tenía 2,212 palabras brutas y 1,463 palabras repetidas; la referencia deduplicada tiene 749 palabras. Las pistas `es` y `es-orig` son idénticas.
- La referencia no se corrigió manualmente; estas métricas miden acuerdo con subtítulos automáticos, no WER/CER contra verdad humana.
- Primera pasada inválida: se puntuó contra los captions rodantes sin deduplicar, por eso dio WER≈0.67 en los tres sistemas. Se descarta. El primer intento de MAI con WebM devolvió HTTP 400 sin `usage.cost`; al repetir solo MAI como WAV sí respondió. La impresión de la primera tabla también chocó con la codificación CP1252; se corrigió el encabezado y ahora el log se escribe antes de imprimir.
- Pasada válida: una llamada por modelo sobre el mismo WAV y la referencia corregida automáticamente; las transcripciones no se persistieron.

| Modelo | WER ↓ | CER ↓ | Latencia | Coste informado |
|---|---:|---:|---:|---:|
| MAI-Transcribe 2 | **0.0227** | 0.0193 | 3.48 s | $0.008278 |
| Qwen3 ASR 0.6B | 0.0280 | **0.0177** | 5.47 s | $0.000992 |
| Whisper Large V3 Turbo | 0.0347 | 0.0219 | **1.66 s** | $0.003308 |

- Coste informado de la pasada válida: `$0.012578`. Coste informado acumulado de las llamadas exitosas, incluidas las pruebas con referencia inválida: `$0.022840`; el intento HTTP 400 no devolvió coste, así que no se incluye y su cargo exacto no se pudo confirmar.
- El preflight conservador estimó `$0.012440` para la pasada válida; el coste real informado fue `$0.012578`. El preflight no es un límite de facturación.
- Resultado provisional: MAI tuvo el WER más bajo, Qwen el CER más bajo y Whisper la menor latencia. Diferencias pequeñas en una muestra de un solo vídeo no bastan para elegir definitivamente; validar con una referencia humana antes de tomarlo como ranking de precisión.
- Resultados detallados y metadatos: [benchmark_results.md](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/benchmark_results.md>) y [SAMPLE_INFO.md](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/SAMPLE_INFO.md>).
- Verificación tras los arreglos del comparador: 32/32 tests pasan; `git diff --check` termina con exit 0; `.env` sigue ignorado por Git. Estas comprobaciones no contactan OpenRouter.

### 2026-09-23 — Qwen3 ASR 1.7B

- OpenRouter confirma el slug `qwen/qwen3-asr-1.7b` y una tarifa de catálogo de `$0.000008/s` ([ficha del modelo](https://openrouter.ai/qwen/qwen3-asr-1.7b)). Se añadió al CLI como opción explícita; los tres modelos anteriores siguen como defaults para evitar añadir gasto inadvertido.
- Una llamada sobre el mismo WAV de 297.77 s y la misma referencia automática deduplicada. Preflight: `$0.002382`; coste devuelto por OpenRouter: `$0.002233`; latencia: 4.14 s. Un primer intento quedó bloqueado localmente antes de conectar (`WinError 10013`) y no envió ninguna petición.
- Resultado: WER `0.0254`, CER `0.0228`. Frente al Qwen 0.6B (WER `0.0280`, CER `0.0177`, `$0.000992`), mejora el WER en 0.0026, empeora CER en 0.0051 y cuesta 2.25×. La diferencia de WER equivale a unas dos palabras en esta referencia de 749 palabras.
- Elección provisional para Qwen: conservar 0.6B por relación coste/calidad; 1.7B solo si una referencia humana confirma que su pequeña mejora de WER importa. El ranking sigue limitado por una muestra y subtítulos automáticos.
- Coste informado acumulado de respuestas exitosas tras esta prueba: `$0.025073`; el HTTP 400 de la primera prueba MAI con WebM no devolvió coste y permanece fuera de la suma.
- Verificación tras añadir el modelo: 32/32 tests pasan; el CLI ofrece el slug 1.7B y mantiene los tres defaults anteriores; `git diff --check` termina con exit 0 y `.env` sigue ignorado. No hubo llamadas de red durante estas comprobaciones.


### 2026-09-23 — Transcripción Qwen guardada para revisión

- La respuesta de Qwen 0.6B de la comparativa anterior solo se mantuvo en memoria. Se hizo una llamada única sobre el WAV de 297.759 s para guardarla localmente: 749 palabras, 4,307 caracteres, 4.079 s y coste informado de $0.00099151749.
- Archivo: [qwen3-asr-0.6b_transcript_es.txt](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/qwen3-asr-0.6b_transcript_es.txt>). ID: `gen-stt-1790153685-a1Wf86PuiN3sOXa9kfR4`. La clave no se imprimió ni guardó.

### 2026-09-23 — Evaluación inicial del sistema RAG

- Corregida la rama OpenRouter en ask(): llamaba una función inexistente (_generate_with_openai) y pasaba un parámetro incorrecto. Añadida una prueba mock sin red. Después reforcé el prompt: cada afirmación factual debe citar una fuente enumerada con el formato literal [Fuente N].
- Se ejecutó una petición por lotes con embeddings openai/text-embedding-3-small (6 ventanas + 8 consultas, 1,306 tokens, 1.240 s, $0.00002612) y las implementaciones del repo BM25Index, hybrid_search/RRF y evaluate_rankings. En 7 consultas respondibles con gold del fragmento mínimo suficiente, Recall@3/MRR/nDCG@3: dense 0.714/0.540/0.537; BM25 1.000/0.690/0.770; RRF 0.857/0.524/0.609. En esta muestra BM25 fue superior; RRF no lo superó. Los embeddings remotos no sustituyen el BGE-M3 de producción.
- Tres consultas por ask() con ranking BM25 inyectado para aislar generación/citas: GPT-6 Luna low, coste $0.000309, latencias 2.715/1.545/1.648 s; tres respuestas correctas contra la transcripción con citas válidas.
- Smoke híbrido: ask() ejecutó BM25+RRF con scores dense remotos y GPT-6 Luna. Una primera respuesta se rechazó por cita inválida/ausente ($0.000106). La repetición diagnóstica del mismo prompt produjo citas válidas ($0.000107). Después de reforzar el prompt, la ruta híbrida respondió correctamente y pasó la validación de cita ($0.000105). El resultado muestra que el guard bloquea respuestas sin cita válida y que la forma de citar puede variar; 3 muestras no estiman la tasa de fallo.
- Otra llamada de embeddings para el smoke híbrido: 6 ventanas + 1 consulta, coste informado $0.0000239, latencia 1.081 s. Total del trabajo de evaluación RAG (embeddings + 6 llamadas GPT): $0.00067702.
- Suite tras las correcciones: python -m unittest discover -v — 33/33 pasan. Gasto informado conocido acumulado, incluida la llamada adicional para guardar la transcripción Qwen: $0.026742; aparte queda la petición previa a MAI que devolvió HTTP 400 sin coste.
- Informe: [RAG_EVALUATION.md](RAG_EVALUATION.md). Transcripción: [qwen3-asr-0.6b_transcript_es.txt](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/qwen3-asr-0.6b_transcript_es.txt>). Respuestas base: [rag_system_smoke_results.json](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_system_smoke_results.json>). Métricas/rankings: [rag_retrieval_embeddings_results.json](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_retrieval_embeddings_results.json>). Ejecución híbrida final: [rag_hybrid_end_to_end_prompt2.json](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_hybrid_end_to_end_prompt2.json>).
- Bloqueo para la medición productiva: faltan torch, sentence-transformers, langchain-text-splitters, pyarrow, el corpus y el índice; tampoco hay BGE-M3/cross-encoder en caché. Hay 4.3 GB de RAM libres ahora y no se detectó GPU. No instalé dependencias ni descargué modelos. Siguiente: repetir con el BGE-M3 de producción, más transcripciones y gold labels revisados; luego evaluar cross-encoder y abstención con casos positivos/negativos.

### 2026-09-23 — Retrieval con BGE-M3 y gate de abstención

- Una petición por lotes a OpenRouter para 6 ventanas y 8 consultas; modelo servido como `parasail-bge-m3`, 1.218 tokens, vectores de 1.024 dimensiones, 1.056 s y coste real informado de **$0.00001218**. No se hicieron reintentos.
- Como el texto exacto de las consultas originales no se había guardado, reconstruí una batería equivalente a partir de las etiquetas del informe y volví a medir BM25 y RRF con los mismos textos y siete gold chunks. Recall@3/MRR/nDCG@3: BGE-M3 `1.000/0.762/0.823`, BM25 `1.000/0.667/0.752`, RRF `1.000/0.643/0.736`. En esta muestra BGE ordena mejor; RRF no mejora los baselines.
- La consulta no respondible obtuvo coseno máximo `0.363503`; las siete positivas dieron entre `0.531828` y `0.722754`. El gate por defecto `0.35` aceptaría esa negativa. No ajusté el umbral por haber solo un negativo etiquetado.
- No encontré transcripciones adicionales revisadas dentro de RAG-App. Hay vídeos en `videoscrapping`, otro proyecto local y sin confirmar como corpus de RAG; no los subí a ningún servicio. La pasada de cross-encoder Cohere se registra a continuación; el modelo local sigue sin probarse.
- Resultado detallado: [rag_retrieval_bge_m3_results.json](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_retrieval_bge_m3_results.json>). Script reproducible de esta pasada: [run_bge_m3_eval.py](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/run_bge_m3_eval.py>). Informe: [RAG_EVALUATION.md](RAG_EVALUATION.md).
- Coste acumulado conocido hasta esta fase: **$0.02675418**. Queda fuera de esa suma una petición previa a MAI que devolvió HTTP 400 sin información de coste; OpenRouter no expuso el importe de esa petición.

### 2026-09-23 — BGE-M3 ampliado y prueba de abstención

- Amplié la batería sobre la misma transcripción revisada: 25 consultas (17 respondibles y 8 negativas), seis ventanas de 800 caracteres con solapamiento 80 y gold chunks asignados a mano. Una petición BGE-M3 de 31 entradas (6 documentos + 25 consultas) devolvió 1.539 tokens, latencia 3.324 s y coste informado **$0.00001539**. Preflight: $0.00002221.
- En las 17 positivas, Recall@3/MRR/nDCG@3: BGE-M3 `0.882/0.713/0.737`, BM25 `1.000/0.725/0.796`, RRF `1.000/0.706/0.782`. BM25 fue ligeramente superior en esta batería; RRF mantuvo Recall@3, pero no superó BM25.
- Gate con score mínimo `0.35`: 17/17 positivas aceptadas y 8/8 negativas aceptadas (8 falsos aceptados). Score coseno positivo `0.5146–0.7228`, negativo `0.3634–0.6454`; los rangos se solapan y ningún umbral simple separa todo este conjunto. No se cambió el umbral por defecto.
- Prueba de generación para la negativa «¿A qué hora exacta recibió el jefe de gabinete el aviso del CNI?», inyectando sus top-3 BGE-M3 en `ask()`: GPT-6 Luna low contestó que el texto solo da el día, no la hora; `[Fuente 1]` pasó el validador. Uso OpenRouter: 813 tokens de entrada + 35 de salida, coste real **$0.0000988**. Revisión manual: la respuesta coincide con el texto. El campo `semantic_faithfulness_checked` quedó en `false`; el guard actual verifica la cita, no la implicación semántica.
- Coste acumulado conocido hasta esta fase: **$0.02686837**. Sigue excluida la petición MAI HTTP 400 sin coste devuelto.
- Artefactos: [métricas BGE-M3 ampliadas](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_retrieval_bge_m3_expanded_results.json>), [respuesta negativa](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_unanswerable_gpt6_luna_result.json>) y [script de evaluación](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/run_bge_m3_eval.py>).

### 2026-09-23 — Cross-encoder Cohere Rerank v3.5

- La ruta `cohere/rerank-v3.5:free` devolvió HTTP 404 sin proveedor activo; no completó ninguna búsqueda. La ruta gratuita aparece como de coste cero pero sujeta a límites ([modelo free y límites](https://openrouter.ai/cohere/rerank-v3.5:free)). Usé la ruta normal publicada a $0.001/search ([precio y API](https://openrouter.ai/cohere/rerank-v3.5/api)).
- Integré una función de llamada al endpoint OpenRouter `/api/v1/rerank` en `hybrid_search(..., reranker=...)`. Se evaluaron las 17 preguntas respondibles, cada una sobre los seis fragmentos candidatos, con 17 llamadas; uso informado: 17 búsquedas, **$0.017**. No se reintentó ninguna petición. Latencia media del endpoint 0.343 s (mediana 0.302 s, máximo 0.847 s); el tiempo total de 57.05 s incluye pausas de 3.2 s entre llamadas para espaciar el ritmo.
- Métricas Recall@3/MRR/nDCG@3 del híbrido con reranker: `1.000/0.941/0.957`, frente a RRF `1.000/0.706/0.782` y BM25 `1.000/0.725/0.796`. El orden mejoró en este conjunto pequeño; no cambia Recall@3 porque los seis fragmentos ya eran candidatos. La evaluación de ocho negativas se registra a continuación.
- Coste acumulado conocido hasta esta fase: **$0.04386837**. La petición MAI HTTP 400 anterior continúa excluida por no haber devuelto coste.
- Resultado completo: [rag_cross_encoder_rerank_results.json](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_cross_encoder_rerank_results.json>). Script: [run_rerank_eval.py](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/run_rerank_eval.py>).

### 2026-09-23 — Scores del cross-encoder en consultas negativas

- Ocho llamadas adicionales a `cohere/rerank-v3.5`, una por negativa y con los seis fragmentos, coste informado **$0.008**. No hubo reintentos. La ruta gratuita `cohere/rerank-v3.5:free` ya había devuelto HTTP 404 sin endpoint activo y sin uso facturado.
- Score máximo de los 17 positivos: `0.4084–0.9254`; score máximo de las 8 negativas: `0.0392–0.8507`. Hay solapamiento. La pregunta de la hora exacta del aviso obtuvo `0.8507`: el reranker identifica el tema correcto aunque el contexto no contiene la hora. El score sirve para ordenar relevancia, no como prueba de suficiencia.
- Coste acumulado conocido de llamadas exitosas: **$0.05186837**, alrededor del 0.74% de los $7 indicados. La llamada anterior de MAI que acabó en HTTP 400 sigue fuera porque no informó un coste.
- Resultados: [scores negativos y comparación](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_cross_encoder_negative_results.json>) y [script](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/run_rerank_negative_eval.py>).

### 2026-09-23 — Checker semántico de fidelidad

- Añadido como opt-in y desactivado por defecto. Usa GPT-6 Luna `low`; limita la respuesta a 4.000 caracteres, la evidencia total a 8.000 caracteres, la salida a 64 tokens y el timeout a 20 s. Si se superan los límites o falla/falta el veredicto JSON booleano, la respuesta se rechaza (fail-closed).
- La suite local pasó **38/38 tests**, incluidos tests con mocks sin red. Para esta implementación se hicieron **cero llamadas facturables** a OpenRouter.
- La cifra de **$0.0002–$0.001 por respuesta** es solo una estimación documentada; no es gasto medido ni se suma al coste acumulado.

### 2026-09-23 — Auditoría de etiquetas y métricas de retrieval

- La revisión de solo lectura encontró coherentes las etiquetas answerable/unanswerable y los gold chunks con la transcripción revisada. El alcance es la evidencia en esa transcripción, no la verdad externa.
- Recall@k, MRR, nDCG y los promedios macro concuerdan con la implementación; los valores Cohere se reproducen desde los qrels y rankings guardados: `1.000/0.941/0.957` (Recall@3/MRR/nDCG@3).
- En el fixture histórico v1, q8 y q22 eran negativas casi duplicadas sobre una condena judicial; q22 ya se sustituyó en v2 por la pregunta sobre la edad del jefe de gabinete.
- El JSON BGE-M3 ampliado v1 guarda solo `vector_top3`, aunque MRR recorre el ranking completo; su MRR `0.712745` no se reproduce independientemente desde el artefacto. La salida v2 registra los seis ranks y scores vectoriales y hace auditable su MRR.
- El harness reemplazó q22 por «¿Qué edad tenía el jefe de gabinete del delegado del Gobierno en Ceuta?» (la transcripción no menciona su edad), conserva `vector_top3` y registra ranking, índice y score de los seis fragmentos. El ranking completo hace auditable el MRR v2. El histórico v1 solo conserva top 3 y sigue siendo incompleto para reproducir su MRR.

### 2026-09-23 — Resultados retrieval expanded-v2

- Nueva ejecución con OpenRouter `parasail-bge-m3`: 31 entradas (6 fragmentos + 25 preguntas), 1.540 tokens, coste informado **$0.0000154**, latencia 3.3603 s. Artefacto nuevo: [rag_retrieval_bge_m3_expanded_v2_results.json](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_retrieval_bge_m3_expanded_v2_results.json>). El JSON histórico v1 se conserva.
- Métricas macro sobre 17 positivas (Recall@3/MRR/nDCG@3): BGE-M3 `0.88235294/0.71274510/0.73668935`; BM25 `1/0.72549020/0.79551288`; RRF `1/0.70588235/0.78150462`. La salida v2 guarda los seis ranks y scores vectoriales; MRR es auditable desde el artefacto.
- Gate `0.35`: TP=17, FP=8, FN=0, TN=0; precision `0.68`, recall `1.0`. Este resultado no justifica un umbral de producción.
- Coste API conocido acumulado: **$0.05188377** (anterior $0.05186837 + $0.0000154).

### 2026-09-23 — Dos transcripciones adicionales preparadas

- Se transcribieron con Qwen3 ASR 0.6B a través de OpenRouter, en español, con segmentos de 5 minutos y 2 s de solapamiento. v1 `j2jK6ogWU8g`: 32 min 30 s, 5.076 palabras, 7 segmentos, coste reportado **$0.00653346**; [texto](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-j2jK6ogWU8g/qwen3-asr-0.6b_transcript_es.txt>) y [metadata](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-j2jK6ogWU8g/qwen3-asr-0.6b_transcript_es.metadata.json>).
- v2 `zq3fDg0MMpQ`: 1 h 10 min 46 s, 11.298 palabras, 15 segmentos, coste reportado **$0.01423241**; [texto](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-zq3fDg0MMpQ/qwen3-asr-0.6b_transcript_es.txt>) y [metadata](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-zq3fDg0MMpQ/qwen3-asr-0.6b_transcript_es.metadata.json>).
- El preflight total, incluido el solapamiento, fue **$0.018708**; OpenRouter reportó **$0.02076587**, aproximadamente un 11% por encima. Para contabilidad se usa el coste reportado. Los textos están listos para el corpus, pero todavía no se han indexado ni medido con retrieval/RRF, reranking o abstención. El acumulado conocido pasa de **$0.05188377** a **$0.07264964**.

### 2026-09-23 — Piloto de reranking Jev

- Se pasó `typesafe/jev-1.13` por OpenRouter sobre BM25 top 9 de las 12 consultas del corpus (9 positivas, 3 negativas): 108 decisiones Noul en 12 peticiones, una por consulta con 9 candidatos. El piloto puntuó suficiencia directa de evidencia y ordenó cada shortlist. Jev no añade candidatos que BM25 no haya recuperado.
- En las nueve positivas, Recall@3/MRR/nDCG@3 pasa de **0.667/0.568/0.544** (BM25 top 9) a **1.000/0.944/0.959**. Recall@5 pasa de `0.889` a `1.000`. Los gold ranks BM25→Jev fueron q01 `4→1`, q02 `2→1`, q03 `1→1`, q05 `1→1`, q06 `2→1`, q07 `1→2`, q09 `4→1`, q10 `9→1`, q11 `2→1`.
- Max Noul por positivas: `0.81–0.96`; por negativas: `0.02–0.03`. En el piloto, thresholds `0.3–0.8` separan las 12 etiquetas. No se fija threshold de producción: son solo tres negativas y preguntas curadas.
- La primera llamada corrió 120 juicios y OpenRouter informó **$0.001297464**, pero un fallo de postprocesamiento local no guardó esos scores. Para preservar el límite de gasto repetí solo los 9 candidatos por consulta (108 juicios, todos los gold seguían dentro); esta tanda reportó **$0.001178772**, 28.066 tokens de entrada + 2.087 de salida. Coste Jev total, incluida la tanda perdida: **$0.002476236**. Acumulado OpenRouter conocido: **$0.075125876**, aproximadamente **$6.924874** restantes de los $7.
- [Resultados, scores y métricas detalladas](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/rag-eval-corpus-qwen-2026-09-23/jev_rerank_pilot_results.json>); [corpus y diario local](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/rag-eval-corpus-qwen-2026-09-23/RAG_EVALUATION_DATASET.md>). El reranking solo cambia el orden del contexto para evaluar retrieval; todavía no está integrado a `ask()` ni se ha evaluado en respuestas generadas.

### 2026-09-23 — Comparación directa Cohere vs Jev

- Comparé `cohere/rerank-v3.5` y `typesafe/jev-1.13` sobre las mismas 12 consultas y los mismos 9 candidatos BM25 por consulta (108 juicios por modelo; 9 positivas y 3 negativas). Métricas calculadas con los mismos gold qrels. Cohere tardó de 0.277 a 1.130 s por llamada (media 0.447 s); fueron 12 llamadas y no hubo reintentos.
- Recall@3/MRR/nDCG@3: BM25 `0.667/0.568/0.544`, BM25+Jev `1.000/0.944/0.959`, BM25+Cohere `1.000/0.926/0.944`. En esta batería ambos rerankers recuperan los 9 gold en top 3; la pequeña diferencia agregada favorece a Jev. La q07 tiene una etiqueta gold singular (#20) pero el fragmento vecino #21 también expresa el objetivo de los agentes; esa ambigüedad afecta una de las diferencias de MRR.
- Costes para 12 consultas: BM25 local **$0 de API**; pasada Jev guardada **$0.001178772** (aprox. `$0.0000982/consulta`); Cohere **$0.012** (`$0.001/búsqueda`, precio publicado [aquí](https://openrouter.ai/cohere/rerank-v3.5/api)). El gasto Jev del experimento completo, incluyendo la primera tanda facturada con resultados perdidos por un fallo de postprocesamiento, es **$0.002476236**. Cohere sale unas **10.2×** más que una pasada Jev guardada.
- Jev: max scores positivas `0.81–0.96`, negativas `0.02–0.03`. Cohere: `0.2533–0.9112` y `0.1393–0.2289`; hay un intervalo descriptivo entre max negativa y mínima positiva, pero no debe usarse para elegir threshold de producción con solo 3 negativas.
- El primer intento local de red fue bloqueado antes de conectar y no generó gasto. Se habilitó la llamada posterior; OpenRouter informó 12 unidades, **$0.012** total. El gasto conocido acumulado pasa de **$0.075125876** a **$0.087125876**, quedando aproximadamente **$6.912874** de $7.
- [JSON de comparación completo](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/rag-eval-corpus-qwen-2026-09-23/cohere_rerank_pilot_results.json>); [script con checkpoint y cap de $0.012](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/rag-eval-corpus-qwen-2026-09-23/run_cohere_rerank_pilot.py>).
