# Production-oriented implementation plan

## Delivered in this refactor

- [x] Separate English-named domain, application, adapters, and presentation layers.
- [x] Replace flat CLI-only flow with a local web app for saved YouTube channels, video discovery, explicit video/batch selection, ingestion status, and scoped chat.
- [x] Keep newly discovered videos pending; refreshing a channel never triggers paid transcription.
- [x] Use Qwen3 ASR 0.6B, BGE-M3, GPT-6 Luna (`low`), and optional Jev through OpenRouter; remove local inference dependencies and Whisper.
- [x] Make hybrid BM25 + vector retrieval with RRF the default and keep Jev optional.
- [x] Use SQLite for metadata, transcript, chunks, embeddings, durable jobs, and conversations; do not add Qdrant.
- [x] Require explicit approval, estimate audio cost before queueing, cap batch spend, avoid automatic paid retries, and retain completed transcript/embedding work for explicit retry after an interrupted request.
- [x] Add source-linked answers, insufficient-evidence abstention, and an optional-by-request faithfulness check that is enabled in the UI with its extra model call disclosed.
- [x] Preserve a deterministic retrieval benchmark and log baseline/current measurements.

## Follow-up work

- [ ] Validate the app against a handful of manually selected public channels and short videos, including unavailable/private/deleted videos, without approving a transcription batch by default.
- [ ] Review one user-approved transcript and build a small labeled retrieval set from the existing approved video corpus; report Recall@k, MRR, nDCG and abstention false-accept/false-reject rates.
- [ ] Check OpenRouter's current audio limits and support safe chunked transcription for long/high-bitrate audio before using it on long videos; preserve partial results to prevent a failed retry from paying twice.
- [ ] Add explicit library deletion/export and data retention controls for transcripts, vectors, chats, and temporary media.
- [ ] Add authentication only if the app is ever exposed beyond localhost; bind to loopback by default.
- [ ] Calibrate the evidence threshold and faithfulness checker on reviewed examples; neither is a formal guarantee.
- [ ] Add migration tests and backup/restore documentation before storing a large or important library.

## Acceptance measurements

- Local deterministic suite and fixture benchmark pass without network access or model downloads.
- Starting the web app imports no ML frameworks and makes no billable requests.
- Listing/refreshing channels makes no transcription or embedding requests.
- Approval over the configured estimate fails before the worker contacts an AI provider.
- A worker restart marks in-flight work for review instead of automatically repeating a potentially billable request; explicit reapproval resumes saved transcript/embedding stages.
- Retrieval defaults to hybrid BM25 + BGE-M3 + RRF; chat citations resolve to source video URLs.
