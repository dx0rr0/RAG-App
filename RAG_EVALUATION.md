# Evaluación del sistema RAG

Fecha: 2026-09-23

## Alcance y datos

- Texto: [transcripción Qwen3 ASR 0.6B](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/qwen3-asr-0.6b_transcript_es.txt>), del clip `XVwQCT15K3U`, 749 palabras. El usuario revisó la transcripción y la considera suficientemente buena; las respuestas se juzgan contra ese texto, no contra la verdad externa.
- Se prepararon 6 ventanas consecutivas de 800 caracteres con 80 de solapamiento. Es una segmentación ligera para esta prueba; no se pudo invocar el divisor de producción (`RecursiveCharacterTextSplitter`) porque faltan dependencias.
- La comparación inicial usa siete consultas respondibles con un fragmento de evidencia directa etiquetado manualmente y una negativa. La batería ampliada posterior usa 17 consultas respondibles y ocho negativas, todas revisadas contra esta misma transcripción; sus resultados aparecen más abajo.
- Esta muestra corta y de un solo vídeo sirve como smoke test, no como estimación de calidad general.

## Recuperación vectorial, BM25 y RRF

Se ejecutaron `BM25Index`, `hybrid_search` y `evaluate_rankings` del repo sobre las seis ventanas. Para dense retrieval se generaron embeddings de los seis fragmentos y ocho consultas en una petición por lotes de OpenRouter con `openai/text-embedding-3-small` (1,306 tokens; coste informado $0.00002612; 1.240 s). OpenRouter documenta el endpoint `/api/v1/embeddings` y entradas por lotes ([API](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings)). `Recall@3`, MRR y `nDCG@3` son promedios macro de las siete consultas respondibles; cada consulta tiene como gold el fragmento mínimo que contiene la respuesta directa.

| Consulta | Gold | Ranking BM25 top 3 | Posición del gold |
|---|---:|---|---:|
| Tres versiones del Gobierno sobre los avisos | 1 | 3, 1, 2 | 2 |
| Número y fecha de los informes desclasificados | 1 | 5, 2, 1 | 3 |
| Organismos que elaboraron los informes | 1 | 5, 1, 6 | 2 |
| Quién recibió el aviso del CNI y qué cargo tenía | 2 | 2, 4, 3 | 1 |
| Qué aviso recibió el jefe de gabinete | 6 | 2, 6, 4 | 2 |
| Medios por los que llegaron los avisos y organismos destinatarios | 3 | 3, 5, 1 | 1 |
| Primera versión del ministro del Interior | 4 | 4, 3, 5 | 1 |

| Método | Recall@3 | MRR | nDCG@3 |
|---|---:|---:|---:|
| Dense (`text-embedding-3-small`) | 0.714 | 0.540 | 0.537 |
| BM25 | **1.000** | **0.690** | **0.770** |
| RRF (dense + BM25) | 0.857 | 0.524 | 0.609 |

El primer cálculo exploratorio utilizó varios fragmentos relacionados por consulta. Después se refinó el etiquetado a la evidencia mínima suficiente, que es el protocolo usado en la tabla final; los valores actuales sustituyen a los preliminares comunicados durante la ejecución. En esta muestra BM25 recupera todos los fragmentos gold en el top 3. El RRF combina las listas, pero no supera BM25: el nDCG baja de 0.770 a 0.609. Con solo siete consultas esto no demuestra que RRF empeore el sistema general; sí indica que aquí no hay mejora observada.

La consulta no respondible («¿Qué condena impuso un tribunal?») obtiene resultados BM25 léxicos y el embedding remoto `text-embedding-3-small` da similitud máxima 0.282. BM25 por sí solo no detecta falta de evidencia. Ese valor corresponde a otro modelo y a una muestra minúscula; no debe usarse para calibrar el umbral de producción.

### Repetición exploratoria con BGE-M3 (7 consultas)

El 2026-09-23 envié una sola petición por lotes a OpenRouter con las seis ventanas y las ocho consultas. OpenRouter informó 1.218 tokens, coste real de **$0.00001218** y 1.056 s; devolvió vectores de 1.024 dimensiones y el proveedor `parasail-bge-m3`. La ficha pública indica $0.01 por millón de tokens ([precio BGE-M3](https://openrouter.ai/baai/bge-m3/pricing)); se usó el endpoint de embeddings ([documentación](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings)).

Las consultas se reconstruyeron a partir de las etiquetas de la tabla anterior porque no se guardó el texto original de aquella primera pasada. Para que la comparación siga siendo pareja, recalculé BM25 y RRF sobre estas mismas consultas, las mismas seis ventanas y los mismos siete fragmentos gold. Métricas macro sobre las siete consultas respondibles:

| Método | Recall@3 | MRR | nDCG@3 |
|---|---:|---:|---:|
| BGE-M3 remoto | **1.000** | **0.762** | **0.823** |
| BM25 | **1.000** | 0.667 | 0.752 |
| RRF | **1.000** | 0.643 | 0.736 |

En esta muestra, BGE-M3 coloca la evidencia relevante más arriba que BM25; RRF vuelve a quedar por debajo de cada método individual. La muestra sigue siendo un único vídeo y siete preguntas, así que no es evidencia suficiente para elegir una estrategia de producción.

En este primer ensayo de siete positivas y una negativa, la negativa alcanzó similitud coseno máxima **0.3635**, frente a **0.5318–0.7228** de las positivas. Parecía posible separar estos ocho ejemplos con un umbral intermedio, pero la batería ampliada de abajo encuentra negativas con scores mayores y rangos solapados. No se cambia el umbral con el resultado exploratorio.

### Batería ampliada v1 (histórica): retrieval y abstención

Para reducir la dependencia de una sola negativa, amplié el conjunto a **25 preguntas** (17 respondibles y 8 negativas), etiquetadas manualmente contra la transcripción revisada. Una sola petición con seis ventanas y 25 preguntas devolvió 1.539 tokens y costó **$0.00001539**; latencia 3.324 s. BM25 y RRF usan las implementaciones del repo, y la comparación se calcula sobre exactamente las mismas preguntas y seis ventanas.

| Método | Recall@3 | MRR | nDCG@3 |
|---|---:|---:|---:|
| BGE-M3 vectorial | 0.882 | 0.713 | 0.737 |
| BM25 | **1.000** | **0.725** | **0.796** |
| RRF | **1.000** | 0.706 | 0.782 |

En esta batería BM25 es ligeramente mejor que BGE-M3 en las tres métricas; RRF conserva Recall@3 de 1.0 pero no mejora BM25. Esto reemplaza el resultado preliminar de siete preguntas para orientar decisiones, aunque sigue midiendo un solo vídeo y no representa todavía la calidad de producción.

El gate actual con `min_evidence_score=0.35` aceptó las 17 positivas y también las **8 negativas** (17 verdaderos aceptados, 8 falsos aceptados, 0 falsas abstenciones, 0 verdaderas abstenciones). Los scores positivos van de **0.5146 a 0.7228**; los negativos, de **0.3634 a 0.6454**. Los rangos se solapan: con estos ejemplos no existe un umbral coseno simple que clasifique todo correctamente. No cambié el valor por defecto; se necesita medir con más transcripciones y casos negativos variados.

Probé además una de las negativas difíciles por `ask()` y GPT-6 Luna low, con los tres primeros fragmentos BGE-M3. Para «¿A qué hora exacta recibió el jefe de gabinete el aviso del CNI?», el modelo dijo que el contexto solo indica el 29 de julio y no especifica hora. Citó `[Fuente 1]`, y la validación de procedencia aceptó la cita; OpenRouter informó 813 tokens de entrada, 35 de salida y **$0.0000988**. La revisión manual confirma que la transcripción no da la hora. `semantic_faithfulness_checked` quedó en `false`: el código comprobó el ID de cita, no el apoyo semántico automático. Este resultado prueba una abstención adecuada en un caso, no garantiza la conducta en las ocho negativas.

### Cross-encoder Cohere Rerank v3.5

Probé la integración `hybrid_search(..., reranker=...)` contra el endpoint de reranking de OpenRouter. La ruta gratuita `cohere/rerank-v3.5:free` no tenía proveedor activo (HTTP 404; no se completó ni facturó ninguna búsqueda), así que usé la ruta pagada publicada a **$0.001 por búsqueda**. Cohere indica soporte multilingüe de más de 100 idiomas ([modelo y precio](https://openrouter.ai/cohere/rerank-v3.5/api)). Para respetar coste, limité la comparación a las **17 preguntas respondibles**: una búsqueda por pregunta, sobre los seis fragmentos y rerank a top 3. OpenRouter confirmó 17 unidades y **$0.017**.

| Método | Recall@3 | MRR | nDCG@3 |
|---|---:|---:|---:|
| BGE-M3 vectorial | 0.882 | 0.713 | 0.737 |
| BM25 | 1.000 | 0.725 | 0.796 |
| RRF | 1.000 | 0.706 | 0.782 |
| RRF + Cohere Rerank v3.5 | **1.000** | **0.941** | **0.957** |

El reranker mejoró el orden de los gold chunks en esta muestra, sin cambiar Recall@3: frente a BM25, MRR sube **0.216** y nDCG@3 **0.161**; frente a RRF, suben 0.235 y 0.175. Latencia de la API: media **0.343 s**, mediana 0.302 s, máximo 0.847 s por búsqueda; las pausas entre llamadas para limitar el ritmo no se incluyen. Es un resultado prometedor, no una decisión de producción: solo hay seis fragmentos candidatos de un vídeo y las 17 preguntas positivas comparten fuente.

También pasé las ocho negativas por Cohere Rerank; coste informado **$0.008** adicional. El score máximo por pregunta en positivas fue **0.408–0.925** y en negativas **0.039–0.851**. Los rangos se solapan: una pregunta no respondible por la hora exacta obtuvo 0.851 porque los fragmentos tratan el aviso correcto, aunque no contienen la hora. Por tanto, el reranker mejora el orden de relevancia pero su score no sirve por sí solo como prueba de que exista evidencia suficiente.

## Respuestas y citas con GPT-6 Luna

Se hicieron tres llamadas a `openai/gpt-6-luna`, `reasoning_effort=low`, `max_new_tokens=120`. La ruta real `ask()` construyó el prompt, invocó OpenRouter y validó que cada cita apuntara a una fuente recuperada. Para aislar generación/citas se inyectó en `_vector_search` el ranking BM25 de esta prueba; la puntuación `0.8` solo permitió pasar el gate. **Esto no mide embeddings, búsqueda vectorial, RRF ni el umbral de evidencia.**

| Consulta | Resultado revisado contra el texto | Cita | Latencia | Coste informado |
|---|---|---|---:|---:|
| Tres versiones | «Primero, que no hubo ningún aviso; después, que hubo un WhatsApp, pero ninguna alerta formal; y, por último, que sí se produjeron avisos, aunque no llegaron a quienes debían recibirlos.» | válida; fragmento 1 | 2.715 s | $0.000115 |
| Número y fecha de informes | «Se desclasificaron 40 informes el 9 de septiembre.» | válida; fragmento 1 | 1.545 s | $0.000093 |
| Aviso del 29 de julio | «Un llamamiento en redes sociales para asaltar Ceuta por la valla y por mar.» | válida; fragmento 6 | 1.648 s | $0.000101 |

Las tres respuestas contienen los datos que pide cada consulta y citan el fragmento que los respalda. La aplicación comprobó la existencia de la cita; la revisión semántica de esta muestra fue manual. El coste real informado por OpenRouter para las tres llamadas fue **$0.000309**; no se envió el audio en estas llamadas.

## Integración híbrida en `ask()`

Se probó también `ask(retrieval_mode="hybrid")` con `hybrid_search`/RRF y BM25 reales del repo, los scores dense guardados en la llamada de embeddings y GPT-6 Luna. Para «¿Qué aviso recibió el jefe de gabinete del CNI el 29 de julio?», el RRF seleccionó los fragmentos 2, 4 y 6; el último contiene la advertencia concreta.

- Primer intento: el validador rechazó la respuesta y `ask()` devolvió la abstención segura. No se guardó el texto bruto de ese primer intento, por lo que no se sabe si omitió la cita o usó un ID inválido. Coste: $0.000106.
- Una llamada diagnóstica con el mismo contexto produjo una respuesta apoyada por los fragmentos 2 y 6, con `[Fuente 1]` y `[Fuente 3]`; `check_answer_grounding` la aceptó. Coste: $0.000107.
- Hice más estricta la instrucción de cita en el prompt y repetí el mismo camino `ask()` híbrido. Respondió «un llamamiento en redes sociales para asaltar Ceuta por la valla y por mar [Fuente 3]»; la cita se validó. Coste: $0.000105.

Esta repetición confirma que RRF recuperó la evidencia necesaria y que el flujo funciona, pero también muestra una variación puntual de formato de cita. Tres respuestas de muestra y un rechazo no bastan para estimar una tasa de fallo; conviene seguir midiendo citas inválidas en más consultas.

## Fallo encontrado y verificación

La rama OpenRouter de `ask()` llamaba a `_generate_with_openai`, que no existía, y enviaba `max_output_tokens` aunque el helper espera `max_tokens`. Corregí la llamada y el modelo por defecto para uso directo a `openai/gpt-6-luna`. Añadí una prueba que verifica la integración sin red. La ruta real de `ask()` completó las tres llamadas y devolvió citas válidas.

`python -m unittest discover -v`: **33/33 pruebas pasan**.

## Verificador semántico de fidelidad (2026-09-23)

Se conectó un verificador semántico opcional al flujo `ask()`. Está desactivado por defecto; `--check-faithfulness` lo activa con GPT-6 Luna `low` en OpenRouter. La petición limita la respuesta a 4.000 caracteres, la evidencia total a 8.000, la salida a 64 tokens y el timeout a 20 s. Si se exceden los límites o falta/falla un veredicto JSON booleano explícito, `ask()` se abstiene (fail-closed). La prueba automatizada completa pasa **38/38 tests**, incluidos los mocks del verificador; no se hicieron llamadas reales ni facturables a OpenRouter para esta implementación. El coste adicional **$0.0002–$0.001 por respuesta** es una estimación documentada, no gasto medido.

## Límites y siguiente fase

En este entorno no están instalados `torch`, `sentence-transformers`, `langchain-text-splitters` ni `pyarrow`; tampoco se detectó el índice de producción del RAG. BGE-M3 se midió como servicio remoto de OpenRouter, no a través del indexador local; las ventanas usadas no las produjo el divisor de producción. Sí se midió Cohere Rerank v3.5 como servicio alojado, pero no el cross-encoder local `BAAI/bge-reranker-v2-m3`. El comprobador semántico está conectado en `ask()` como opt-in mediante `--check-faithfulness` y se validó con 38 pruebas mock, pero aún no tiene una evaluación semántica real. Las dos transcripciones nuevas descritas abajo están listas para incorporarse al corpus de evaluación; las métricas RAG todavía no se han repetido sobre ellas.

Siguiente fase: indexar las dos transcripciones aprobadas con el chunker del proyecto, preparar consultas con gold labels revisados y repetir vector/BM25/RRF y Cohere Rerank sobre los mismos candidatos. Medir el gate con más positivos y negativos, y revisar manualmente corrección factual, fidelidad, citas y abstenciones. Hasta tener más negativos no fijar un umbral de producción.

Respuestas, tokens, costes por llamada y tiempos completos: [salida JSON del smoke test](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_system_smoke_results.json>). Rankings dense/BM25/RRF: [recuperación inicial](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_retrieval_embeddings_results.json>), [BGE-M3 exploratorio](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_retrieval_bge_m3_results.json>) y [batería ampliada BGE-M3](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_retrieval_bge_m3_expanded_results.json>). [Resultados cross-encoder positivos](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_cross_encoder_rerank_results.json>) y [negativos](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_cross_encoder_negative_results.json>); [smoke de respuesta negativa con GPT-6 Luna](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_unanswerable_gpt6_luna_result.json>). Ejecución `ask()` híbrida: [salida JSON end-to-end](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_hybrid_end_to_end_prompt2.json>).

## Auditoría de evaluación (2026-09-23)

La revisión manual encontró coherentes las etiquetas de respondibilidad y los fragmentos gold con la transcripción Qwen revisada; esta comprobación se limita a ese texto. La implementación de Recall@k, MRR y nDCG y los promedios macro concuerdan con la definición usada. Las métricas Cohere también se reproducen desde los rankings y qrels guardados: Recall@3 `1.000`, MRR `0.941` y nDCG@3 `0.957`.

En el fixture histórico v1, q8 («¿Qué condena impuso un tribunal?») y q22 («¿Qué condena impuso un tribunal a los responsables de la entrada en Ceuta?») eran negativas casi duplicadas. El harness v2 sustituyó q22 por una pregunta sobre la edad del jefe de gabinete. El JSON v1 solo guarda `vector_top3`; por eso su MRR BGE-M3 `0.712745` no puede reproducirse íntegramente desde el artefacto, aunque Recall@3 y nDCG@3 sí coinciden con lo calculable desde el top 3. El v2 guarda los seis ranks; sus resultados se registran a continuación.

### Batería ampliada v2: resultados actualizados

Se volvió a ejecutar el harness con 25 preguntas: 17 respondibles y 8 negativas. q22 ahora pregunta «¿Qué edad tenía el jefe de gabinete del delegado del Gobierno en Ceuta?»; la transcripción aprobada no menciona su edad. OpenRouter devolvió embeddings del proveedor `parasail-bge-m3` para 31 entradas (6 fragmentos y 25 consultas): 1.540 tokens, **$0.0000154** y 3.3603 s. El nuevo JSON registra el ranking vectorial completo de los seis fragmentos con scores y ranks, de modo que el MRR se puede auditar.

| Método | Recall@3 | MRR | nDCG@3 |
|---|---:|---:|---:|
| BGE-M3 vectorial | 0.882353 | 0.712745 | 0.736689 |
| BM25 | 1.000000 | 0.725490 | 0.795513 |
| RRF | 1.000000 | 0.705882 | 0.781505 |

Los valores son promedios macro de las 17 positivas. Gate con `min_evidence_score=0.35`: 17 verdaderos aceptados, 8 falsos aceptados, 0 falsas abstenciones y 0 verdaderas abstenciones; precision `0.68`, recall `1.0`. El artefacto v1 `rag_retrieval_bge_m3_expanded_results.json` se conserva como histórico, pero su MRR no es reproducible independientemente porque solo guardó `vector_top3`; el JSON v2 sí contiene los seis ranks. El coste acumulado conocido sube de **$0.05186837** a **$0.05188377**.

Resultados detallados: [fixture expanded-v2](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_retrieval_bge_m3_expanded_v2_results.json>); el [fixture v1 histórico](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_retrieval_bge_m3_expanded_results.json>) no se sobrescribió.

## Transcripciones adicionales preparadas para el corpus (2026-09-23)

Se transcribieron dos vídeos del usuario con Qwen3 ASR 0.6B a través de OpenRouter, en español, en segmentos de 5 minutos con 2 segundos de solapamiento:

| Vídeo | Duración | Palabras | Segmentos | Coste reportado | Artefactos locales |
|---|---:|---:|---:|---:|---|
| v1 `j2jK6ogWU8g` | 32 min 30 s | 5.076 | 7 | $0.00653346 | [transcripción](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-j2jK6ogWU8g/qwen3-asr-0.6b_transcript_es.txt>) · [metadata](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-j2jK6ogWU8g/qwen3-asr-0.6b_transcript_es.metadata.json>) |
| v2 `zq3fDg0MMpQ` | 1 h 10 min 46 s | 11.298 | 15 | $0.01423241 | [transcripción](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-zq3fDg0MMpQ/qwen3-asr-0.6b_transcript_es.txt>) · [metadata](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-zq3fDg0MMpQ/qwen3-asr-0.6b_transcript_es.metadata.json>) |

Ambos textos y sus metadatos están disponibles para preparar el corpus, pero todavía no se han indexado ni usado para repetir las métricas de retrieval, reranking o abstención. El coste de estas transcripciones suma **$0.02076587**; el acumulado conocido actualizado es **$0.07264964**.

## Piloto de reranking con Jev (2026-09-23)

Probé `typesafe/jev-1.13` por OpenRouter como segunda etapa tras BM25. Jev puntúa si cada pasaje contiene evidencia directa para responder; después se ordena la shortlist por esa puntuación. Este patrón de shortlist + decisión por candidato es el que describe la [guía de reranking de TypeSafe](https://docs.typesafe.ai/cookbooks/rerank_typesafe). Como cualquier reranker, Jev solo puede reordenar los candidatos recibidos, no recuperar fragmentos que BM25 dejó fuera.

La prueba usa las mismas 12 consultas textualmente revisadas del corpus local: 9 respondibles y 3 sin respuesta en el texto, procedentes de los tres vídeos. Pasé a Jev los primeros 9 candidatos BM25 de cada consulta (108 juicios) y comparé el orden anterior y posterior. Las nueve respuestas gold estaban dentro de esas nueve posiciones antes de rerank.

| Métrica | BM25 top 9 | Jev reranked |
|---|---:|---:|
| Recall@3 (9 positivas) | 0.667 (6/9) | 1.000 (9/9) |
| MRR | 0.568 | 0.944 |
| nDCG@3 | 0.544 | 0.959 |
| Recall@5 | 0.889 (8/9) | 1.000 (9/9) |

Rangos del fragmento gold por consulta: q01 `4→1`, q02 `2→1`, q03 `1→1`, q05 `1→1`, q06 `2→1`, q07 `1→2`, q09 `4→1`, q10 `9→1`, q11 `2→1`. Jev mejoró seis rangos, dejó dos iguales y bajó uno un puesto; no cambió la cobertura porque todos los gold ya estaban en la shortlist BM25 top 9.

Las puntuaciones máximas Jev en las 9 consultas respondibles fueron `0.81–0.96`; en las 3 no respondibles, `0.02–0.03`. En esta muestra, los umbrales probados de `0.3`, `0.5`, `0.7` y `0.8` separan todos los casos. Es una señal prometedora para evaluar abstención, no una calibración: 12 consultas curadas no bastan para elegir un umbral de producción.

OpenRouter reportó **$0.001178772** para la ejecución guardada (28.066 tokens de entrada, 2.087 de salida). La primera tanda de 120 juicios se cobró **$0.001297464**, pero un error local al procesar su respuesta impidió guardar los scores; repetí con 108 juicios top 9. Incluyo ambos importes en el coste del experimento: **$0.002476236**. El total conocido pasa de **$0.07264964** a **$0.075125876** de los $7 cargados; quedan aproximadamente **$6.924874**. El precio publicado para Jev 1.13 es $0.042/M tokens de entrada y $0/M de salida ([OpenRouter](https://openrouter.ai/typesafe/jev-1.13/)); para el gasto de esta prueba usé el coste real informado por OpenRouter.

Resultados por consulta, rankings, puntuaciones, tokens y tiempos: [JSON local del piloto](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/rag-eval-corpus-qwen-2026-09-23/jev_rerank_pilot_results.json>). Esto evalúa recuperación de evidencia en el texto transcrito; no mide la calidad de respuestas generadas ni valida las transcripciones contra el audio.
