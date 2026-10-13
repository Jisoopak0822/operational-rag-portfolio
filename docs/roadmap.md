# Stage 2–6 implementation roadmap

This file separates implemented Stage 1 features from planned work.

## Stage 2: retrieval backends and reproducible evaluation
- Replace the TF-IDF channel with multilingual sentence embeddings (`requirements-semantic.txt` can already test this locally).
- Add Milvus or a smaller local vector backend with persistent index and collection lifecycle.
- Replace in-process BM25 with OpenSearch BM25 / Nori for robust Korean tokenization.
- Enlarge evaluation set to >50 manually labeled queries from multiple source types.
- Compare BM25-only, semantic-only, RRF at `k=5/10/25`; persist scores and per-query failures.

## Stage 3: cross-encoder and retrieval experiments
- Add multilingual CrossEncoder to rerank top 25 hits.
- Run identical gold sets for off/on/interleave; record regressions, latency and model version.
- Measure Hit@k, Recall@k, MRR; do not claim quality improvement solely because a reranker exists.

## Stage 4: answer generation and conversation
- Stream SSE status/chunk/done/error through a common internal event protocol.
- Re-write follow-up queries for retrieval only while preserving original user intent.
- Add question type routing and bounded HyDE / Multi-Query with cost and latency controls.
- Add an actual conversational web client; maintain session/user boundary.

## Stage 5: robust knowledge and operations
- Implement validated YAML entity cards, revision manifests, alias collision checks, source lineage and self-generated-source exclusions.
- Add document-level authorization **before** retrieval; current verified flag is not an access-control substitute.
- Add failure injection for vector timeout, keyword failure, LLM failure; log fallback cause.
- Capture p50/p95, failure counts, request traces, token costs, context truncation.
- Keep answer quality evaluation separate from retrieval evaluation; freeze judge prompts and versions.

## Stage 6: deployment and portfolio
- Containerize and isolate tenant config, indices, volumes, ports and secrets.
- Add pilot collection validation and dry-run ingest, then a repeatable on-boarding script.
- Add CI tests and deployment documentation; no public deployment before authentication, rate limiting, and secrets management.
- Capture measured before/after metrics from the **same** gold set, code version, index snapshot, and environment.
