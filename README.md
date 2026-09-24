# Video RAG

A local web app for building a searchable library from YouTube channels. It discovers channel videos, asks for explicit approval before transcription, and answers questions with links to transcript sources.

## Architecture

The Python application uses four English-named layers under `src/rag_app`:

- `domain`: retrieval calculations, entities, ports, and application errors.
- `application`: channel catalog, ingestion, retrieval, chat, chunking, and worker use cases.
- `adapters`: SQLite persistence, YouTube metadata/audio, OpenRouter, and configuration.
- `presentation`: the FastAPI web interface, templates, and static assets.

SQLite stores channel/video metadata, transcription jobs, transcripts, chunks and their vectors, and chat history. Hybrid retrieval is the default: BM25 and BGE-M3 vector rankings are combined with reciprocal rank fusion (RRF). Jev reranking is optional. This app does not use Qdrant or a local vector database service.

## AI and cost controls

All AI inference is remote through OpenRouter; the application never loads local LLM, ASR, embedding, or Whisper models:

- Transcription: `qwen/qwen3-asr-0.6b`.
- Embeddings: `baai/bge-m3` through OpenRouter.
- Chat and faithfulness checks: `openai/gpt-6-luna`, `reasoning_effort=low`.
- Optional reranking: `typesafe/jev-1.13`.

Channel discovery and refresh do not transcribe. The user must select videos and confirm a batch. The default transcription cap is `$0.10` per approved batch, estimated at `$0.000003` per audio second; a batch over the cap is rejected before creating jobs. Set `RAG_MAX_BATCH_USD` to lower it. Requests are not automatically retried after provider failures. Audio is downloaded to a temporary directory and removed after the remote transcription request. Chat output is limited to 384 tokens; the faithfulness check is a second remote request and is shown as such in the UI. Jev is off by default and has a `$0.001` estimated per-query cap.

OpenRouter charges your account. Review the cost estimate before approving each batch. Embedding prices are estimated from the configured `$0.01/M tokens` rate when the response omits cost; transcription estimates use audio duration. Chat costs are shown only when reported by OpenRouter. If the provider does not report a cost, the UI says so instead of showing a false `$0`.

## Run locally

Requires Python 3.10+ and an OpenRouter API key. No model weights or local inference runtimes are needed.

```powershell
cd "path\to\RAG-App"
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

Put your key in `.env` as `OPENROUTER_API_KEY=...`. The app also accepts the existing `OPENROUTER_APIKEY` name. Never commit `.env`.

```powershell
video-rag
```

Or use `python -m rag_app.presentation.web` after installing the package. Open <http://127.0.0.1:8000>. The server binds to localhost only. SQLite is created at `data/rag_app.sqlite3`; set `RAG_APP_DATABASE` to choose another path. The web process starts a SQLite-backed background worker. If the process stops while a paid request is in flight, the job is marked failed; it requires explicit reapproval, and any already-saved transcript or vectors are reused. For a single local instance, do not run multiple web worker processes against the same database.

## Retrieval evaluation

Run the deterministic, no-network fixture benchmark:

```powershell
python -m tests.retrieval_benchmark
```

The fixture checks ranking mechanics and reports Recall@k, MRR, and nDCG. It is not a claim about semantic quality on a production corpus. See [the retrieval evaluation notes](docs/RAG_EVALUATION.md) and [the measurement log](docs/MEASUREMENTS.md). All metrics on a real library should use reviewed query-to-source relevance labels.

## Project map

```text
src/rag_app/
  domain/
  application/
  adapters/
  presentation/
tests/
scripts/
docs/
```
