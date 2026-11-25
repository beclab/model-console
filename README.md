# Model Console

Model Console is a lightweight Go sidecar for running local AI models behind
one stable network interface. It coordinates model acquisition, engine startup,
API compatibility, and Router discovery without tying callers to a particular
inference runtime.

## Core responsibilities

### 1. Model download

Model Console downloads full Hugging Face repositories, individual GGUF files,
Ollama Library tags, and direct HTTPS resources. Downloads support resuming,
integrity verification, shared caches, progress reporting, and explicit retry
or re-download operations.

### 2. Standardized APIs

Applications use one port instead of addressing an engine directly. Model
Console exposes OpenAI-compatible `/v1/*` routes, Anthropic Messages, selected
Ollama-native routes, and capability-specific Audio, OCR, Translate, Embedding,
Rerank, and Music endpoints. Requests are translated or proxied to Ollama,
vLLM, llama.cpp, SGLang, FreeToken, or a sibling capability engine as appropriate.

### 3. Engine lifecycle

The sidecar gates engine startup until model files are ready, publishes engine
arguments and wrapper helpers, supervises restarts, and combines download and
runtime health into `/livez`, `/readyz`, `/healthz`, progress, diagnostics, and
the built-in dashboard at `/`.

### 4. Router metadata

`GET /api/model-spec` exposes model identity, capability flags, pricing,
parameter rules, and engine configuration before the data plane becomes ready.
Additional control-plane endpoints report available operations, effective
engine capacity, configuration, and runtime diagnostics for Router discovery.

## Quick start

For local development, copy the example environment and start one of the
provided engine stacks:

```sh
cp deploy/compose/.env.example .env
# Set MODEL_NAME, MODEL_MODE, and MODEL_SOURCE in .env.
docker compose -f deploy/compose/ollama.yml --env-file .env up
```

Then open `http://127.0.0.1:8080/` or query the service:

```sh
curl -s http://127.0.0.1:8080/readyz
curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/api/model-spec
```

The `deploy/compose/` and `deploy/k8s/` directories contain engine-specific
examples. Production deployments should pin an explicit image version instead
of using `latest`.

### FreeToken

Set `ENGINE_KIND=freetoken` to use the shared OpenAI proxy with the sibling
service `http://freetoken:1919`. FreeToken `ft serve` flags are accepted through
`ENGINE_ARGS`, including MoE placement flags and `--max-running-requests`.
Unknown flags are preserved. `--kv-reserve-tokens` does not declare a context
window or a fixed KV pool capacity.

### Strata

Set `ENGINE_KIND=strata` to proxy the sibling service `http://strata:8080`.
`ENGINE_ARGS` contains Strata **setup.py** options; `--context` and `--parallel`
declare context and concurrency. KV pool capacity remains unknown. The wrapper
uses the downloaded GGUF directory, prepares Strata's pack/MTP cache, and
relaunches the API server when Model Console changes engine arguments.

The NVIDIA manual-test chart and build instructions are in
[strataqwen38iq4xsv3/README.md](strataqwen38iq4xsv3/README.md).

### ComfyUI text-to-image

To publish H3 Image, start `llm-init` with `ENGINE_KIND=comfyui`,
`MODEL_MODE=image`, and `MODEL_NAME=h3-image`. Set `COMFY_MODELS_DIR` to the
ComfyUI weights directory. In the JSON file selected by `COMFY_APP_CONFIG`, declare
one `apiWorkflows` entry with the Studio flow `id` (H3 uses `t2i`), its supported
`aspectRatios`, a `defaultAspect` from that list, and its `resolution` (`2K`).
Supply a shared secret of at least 32 bytes as `COMFY_API_TOKEN` and as
`LLM_GATEWAY_PROVIDER_STUDIO_MEDIA_TOKEN` in Router.

Router discovers the model at `GET /api/model-spec` and checks readiness at
`GET /api/progress`. Once ready, `POST /api/v1/generations` creates a Studio job
as the authenticated user and returns HTTP 202 with the Studio job ID. Router
polls `GET /api/v1/generations/{id}` and downloads a completed image from
`GET /api/v1/generations/{id}/content?outputId=…`. Easy UI shows the same job.

For a weight-download console, start `llm-init comfyui`. This entrypoint
serves download progress and model-group management.

## Development

Go 1.23 or newer is required by the module. The release and security checks use
the Go version pinned in CI.

```sh
make build          # build bin/llm-init
make test           # race-enabled unit tests
make test-download  # offline download matrix
make test-wrappers  # wrapper tests
make lint           # golangci-lint
make verify         # local release gates
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor workflow and testing
details.

## License

Model Console is licensed under the
[GNU Affero General Public License v3.0](LICENSE).
