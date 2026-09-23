# Plan de mejoras de RAG-App

## Objetivo

Hacer que la aplicación RAG arranque de forma reproducible, reutilice su índice y responda con referencias verificables a las transcripciones.

## Prioridad alta: arranque y compatibilidad

- [ ] Eliminar descargas de NLTK y cargas de modelos al importar módulos; inicializar cada recurso explícitamente y reutilizar una sola instancia del modelo de embeddings.
- [ ] Centralizar rutas, modelos y dispositivo en configuración. Elegir CUDA solo cuando esté disponible y usar tipos de precisión compatibles con el hardware.
- [ ] Manejar el corpus vacío y limitar `top_k` al número real de fragmentos disponibles.
- [ ] Añadir dependencias y una guía de instalación/ejecución con requisitos de modelo y hardware.

## Prioridad alta: ciclo de vida del índice

- [ ] Separar indexación de consultas: crear/actualizar el índice cuando cambien las transcripciones y cargarlo directamente en ejecuciones posteriores.
- [ ] Calcular embeddings por lotes, guardar versión del modelo y parámetros de división, y reconstruir el índice cuando esos datos cambien.
- [ ] Evitar el ciclo de generar y recargar el Parquet completo en cada arranque.

## Prioridad media: respuestas verificables y seguras

- [ ] Conservar metadatos de origen durante la recuperación y mostrar citas con vídeo, título y marca temporal cuando estén disponibles.
- [ ] Delimitar las transcripciones como datos no confiables y pedir respuestas basadas en el contexto, con reconocimiento explícito cuando falte evidencia.
- [ ] Decodificar únicamente los tokens generados y usar la plantilla de conversación del modelo cuando esté disponible.

## Prioridad media: simplificar el flujo de trabajo

- [ ] Poner el código experimental de `talk_with_db.py` detrás de un punto de entrada explícito o integrarlo en los módulos compartidos; evitar consultas, descargas e impresiones al importar.
- [ ] Incorporar el ID de vídeo a los nombres de archivos descargados para evitar colisiones entre títulos sanitizados iguales.
- [ ] Hacer configurable o detectable el idioma de transcripción en vez de fijarlo a español.

## Extensión de recuperación y evaluación

### Implementado o probado

- [x] BM25, reciprocal rank fusion y modo híbrido; suite local de 33 pruebas superada tras la integración.
- [x] Contrato de reranker opcional y helper local de cross-encoder en `hybrid_retrieval.py`.
- [x] Métricas de retrieval: Recall@k, MRR y nDCG; batería registrada en [RAG_EVALUATION.md](RAG_EVALUATION.md).
- [x] Citas con IDs de fuente y abstención por score mínimo. Una prueba manual con GPT-6 Luna se abstuvo correctamente cuando faltaba la hora exacta.
- [x] Evaluación hospedada de Cohere Rerank v3.5: mejoró el orden en este único vídeo, por $0.025 en 25 búsquedas; no se considera validación de producción.
- [x] Comprobador semántico de fidelidad opt-in en `ask()` vía `--check-faithfulness`, con límites conservadores y abstención fail-closed. Verificado con mocks en 38 pruebas; no se ha hecho una llamada real al checker.
- [x] Chat multivuelta con historial limitado a cuatro turnos, reformulación de seguimientos probables antes de recuperar, citas basadas en evidencia del turno actual, `/clear` y opción para desactivarlo. Suite con mocks; sin llamadas facturadas.

### Pendiente

- [ ] Cuando se autorice gasto, validar semánticamente con llamadas controladas y ejemplos revisados que incluyan respuestas respaldadas y no respaldadas; hasta entonces solo hay pruebas mock, sin llamadas reales al checker.
- [x] Obtener dos transcripciones aprobadas en español para ampliar el corpus de evaluación; los textos y metadata están disponibles en `video-sample-j2jK6ogWU8g` y `video-sample-zq3fDg0MMpQ`.
- [ ] Indexar esas dos transcripciones con el chunker del proyecto, preparar consultas y gold labels revisados, y repetir retrieval/RRF, Cohere reranking y evaluación del gate. Las métricas actuales siguen siendo del fixture de un solo vídeo; el umbral `0.35` aceptó sus ocho negativas.
- [x] Sustituir q22 por una negativa de tipo distinto sobre la edad del jefe de gabinete; el harness verifica que el dato no aparece en la transcripción aprobada.
- [x] Guardar en nuevas salidas del harness el ranking BGE-M3 completo de los seis fragmentos con sus ranks e índices, conservando `vector_top3`.
- [x] Ejecutar el harness `expanded-v2`; [resultados y ranking completo](<C:/Users/Daniel Chorro/Documents/Codex/2026-09-22/ay/video-sample-XVwQCT15K3U/rag_retrieval_bge_m3_expanded_v2_results.json>) registrados y MRR auditable.
- [ ] Repetir la comparación con el indexador/chunker real y el corpus de producción.
- [ ] Comparar el cross-encoder local con Cohere Rerank y medirlo también en preguntas sin respuesta antes de decidir su configuración.

## Criterios de finalización

- Importar los módulos no hace llamadas de red ni carga modelos.
- Una segunda ejecución reutiliza el índice si las entradas y la configuración no han cambiado.
- La consulta funciona con cero, pocos o muchos fragmentos y en los dispositivos documentados.
- Las respuestas muestran de qué transcripciones procede la evidencia.
