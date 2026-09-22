# Integration tests

Real-container tests driven by `deploy/compose/<engine>.yml`. Build tag
`integration` keeps them out of `go test ./...` so the unit suite stays
fast.

## Layout

```
tests/integration/
├── shared/         reusable harness (compose lifecycle + readiness + API probes)
├── llamacpp/       CI-running test, including a TinyLlama Q4 download
├── ollama/         environment-gated local test
├── vllm/           environment-gated, requires an NVIDIA GPU
├── sglang/         environment-gated, requires an NVIDIA GPU
├── embed/          text embedding models
├── clipembed/      CLIP dual-tower image and text embeddings
├── rerank/         rerank-server POST /v1/rerank
├── download/       URL failure matrix
├── ollamaurl/      ollama-url failure matrix
├── negative/       real upstream negative cases
└── baseline.json   reference timings; never asserted on
```

## Run locally

```sh
# CI-equivalent: llama.cpp on CPU with TinyLlama
go test -tags integration -timeout 10m ./tests/integration/llamacpp/...

# Ollama (CPU-friendly)
LLM_INIT_INTEGRATION_OLLAMA=1 \
    go test -tags integration -timeout 15m ./tests/integration/ollama/...

# vLLM / SGLang (NVIDIA GPU)
LLM_INIT_INTEGRATION_VLLM=1 \
    go test -tags integration -timeout 30m ./tests/integration/vllm/...

LLM_INIT_INTEGRATION_SGLANG=1 \
    go test -tags integration -timeout 30m ./tests/integration/sglang/...

# Embed / IREmbeddingServer
LLM_INIT_INTEGRATION_EMBED=1 \
    EMBED_IMAGE=embed-server:ov-intel-amd64 \
    MODEL_ID=embeddinggemma-300m-ov \
    MODEL_NAME=embeddinggemma-300m \
    MODEL_SOURCE=hf://google/embeddinggemma-300m \
    go test -tags integration -timeout 20m ./tests/integration/embed/...

# ClipEmbed / Jina CLIP v2
LLM_INIT_INTEGRATION_CLIPEMBED=1 \
    MODEL_ID=jina-clip-v2-split-ov \
    MODEL_NAME=jina-clip-v2 \
    MODEL_SOURCE='hf://beclab/jina-clip-v2-split --revision main --exclude "onnx/**" --subdir openvino' \
    go test -tags integration -timeout 45m -run TestClipEmbed \
    ./tests/integration/clipembed/...

# Rerank
LLM_INIT_INTEGRATION_RERANK=1 \
    ACCELERATOR=cpu \
    RERANK_IMAGE=rerank-server:onnx-cpu-amd64 \
    MODEL_SOURCE='hf://beclab/bge-reranker-v2-m3 --revision main --exclude "openvino/**" --subdir onnx' \
    HF_ENDPOINT=https://huggingface.co \
    go test -tags integration -timeout 45m ./tests/integration/rerank/...
```

Each test reuses a production Compose template so deployment artifacts and the
integration harness cannot drift apart. The harness asserts that the published
port is free before starting Compose.

## Baselines

`baseline.json` is committed as a reference. Every run rewrites the row for
its own engine; CI uploads the file as an artifact for release comparison.

## Download failure matrix

These environment-gated tests exercise container-level download failures
against a real Model Console image. They cover successful downloads, recoverable
retries, exhausted retries, non-retriable failures, and integrity failures.
Assertions use `GET /api/progress` rather than `/readyz` because the failure
stacks do not start a real inference engine.

`faultsrv` and the `llm-init` image are built on demand. Set
`FAULTSRV_IMAGE` or `LLM_INIT_IMAGE` to reuse prebuilt images.

| Variable | Purpose |
| --- | --- |
| `LLM_INIT_INTEGRATION_DOWNLOAD=1` | Enable URL and ollama-url failure matrices |
| `LLM_INIT_INTEGRATION_NEGATIVE=1` | Enable live Hugging Face and Ollama negative cases |
| `LLM_INIT_IMAGE` | Image under test; built locally when unset |
| `FAULTSRV_IMAGE` | Fault server image; built locally when unset |
| `LLM_INIT_PORT` | Published host port; defaults to 8080 |

```sh
# URL and ollama-url failure matrices
make test-integration-download

# Equivalent direct invocation
LLM_INIT_INTEGRATION_DOWNLOAD=1 \
    go test -tags integration -timeout 30m -v \
    ./tests/integration/download/... ./tests/integration/ollamaurl/...

# Live upstream negative cases; requires network access
make test-integration-negative

# Equivalent direct invocation
LLM_INIT_INTEGRATION_NEGATIVE=1 \
    go test -tags integration -timeout 20m -v \
    ./tests/integration/negative/...
```

Cases run serially because they share host port 8080.

### URL channel

| Case | faultsrv behavior | Expected phase | `last_error` contains |
| --- | --- | --- | --- |
| `normal` | 200 with a correct `#sha256=` | `ready` | empty |
| `retriable_recover` | two 503 responses, then success | `ready` | empty |
| `retriable_stuck` | persistent 503 | `degraded` | `retry` |
| `non_retriable_404` | persistent 404 | `failed` | `not found` |
| `sha_mismatch` | 200 with an incorrect digest | `degraded` | `verify` |

### ollama-url channel

`ENGINE_KIND=ollama` starts a real Ollama daemon because it is an
AliveBeforeBoot engine. These cases fail during download or verification, before
registration. There is no normal case because faultsrv's random payload is not
a valid GGUF file; the positive Ollama path is covered by
`tests/integration/ollama`.

| Case | faultsrv behavior | Expected phase | `last_error` contains |
| --- | --- | --- | --- |
| `download_retriable_stuck` | persistent 503 | `degraded` | `retry` |
| `download_non_retriable_404` | persistent 404 | `failed` | `not found` |
| `sha_mismatch` | 200 with an incorrect digest | `degraded` | `verify` |

### Live upstream negative cases

These reuse `deploy/compose/<engine>.yml` and assert the user-facing
`last_error` produced from real upstream failures.

| Case | Compose file | `MODEL_SOURCE` condition | Expected phase | `last_error` contains |
| --- | --- | --- | --- | --- |
| `hf_repo_not_found` | `llamacpp.yml` | missing repository | `failed` | `repository not found` |
| `hf_bad_revision` | `llamacpp.yml` | invalid revision | `failed` | `revision not found` |
| `ollama_invalid_tag` | `ollama.yml` | missing library tag | `degraded` | `not found in the library` |

The faster, offline, in-process download matrix lives in `tests/download/`.
