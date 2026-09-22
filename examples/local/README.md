# examples/local

Fifteen hand-run scenarios for observing the Model Console UI and runtime behaviour
across every supported engine + source combination, using the smallest
viable model in each row. Production deployment templates live in
[deploy/compose/](../../deploy/compose/) and are reused unmodified.

## What you observe

Every `.env` here pins `LLM_INIT_IMAGE=llm-init:local`. Both `run.sh`
and `run.ps1` auto-build that image from the current source tree on
`up` whenever it is missing or out of date, so you are always
observing the binary that matches your checkout — never whatever
Docker Hub ships as `:latest`.

Rebuild triggers (on `up` only, default tag only):

- image not present at all;
- image's `org.opencontainers.image.revision` label != `git rev-parse HEAD`;
- uncommitted edits under `cmd/`, `internal/`, `go.mod`, `go.sum`, `Dockerfile`;
- image was previously built from a dirty tree (`version` ends in `-dirty`).

The actual build is `docker build` against the repo root with the
same `VERSION` / `COMMIT` / `BUILD_TIME` build-args the Makefile
uses. Docker's layer cache decides how much work each call does —
usually seconds when nothing under `cmd/` or `internal/` changed.

Bypass the auto-build:

```sh
# bash
SKIP_REBUILD=1 bash examples/local/run.sh A1

# PowerShell
$env:SKIP_REBUILD='1'; .\examples\local\run.ps1 A1
```

Bring your own image (the runner will not rebuild it, only check
that it exists):

```sh
# bash
LLM_INIT_IMAGE=llm-init:dev bash examples/local/run.sh A1

# PowerShell
$env:LLM_INIT_IMAGE='llm-init:dev'; .\examples\local\run.ps1 A1
```

## Quickstart

```sh
# bash / WSL / git-bash
bash examples/local/run.sh A2          # auto-builds llm-init:local if needed
bash examples/local/run.sh A2 down     # tear down + remove volumes

# PowerShell (functionally identical port)
.\examples\local\run.ps1 A2
.\examples\local\run.ps1 A2 down
```

Then open the URL the runner prints under `url :` in the banner.
That is normally http://localhost:8080, but if 8080 is already bound
on the host (Tomcat, Jenkins, another stack, …) both `run.sh` and
`run.ps1` pick a free port in `28080..28099` (or a random ephemeral
one) and inject it via `$env:LLM_INIT_PORT`. Override explicitly with
`LLM_INIT_PORT=18080 bash ...` or `$env:LLM_INIT_PORT='18080'; .\...`.

### Resetting / cleaning state

Each scenario keeps its model bytes in a per-project Docker volume
(`-p llmlocal-<id>`): file-source engines (A2-A4, B3, C2, D*, E1) use
`llmlocal-<id>_hf-hub-cache` (the HF hub cache), and ollama scenarios
(A1, B1, B2, C1) use `llmlocal-<id>_ollama-data`. F1 uses the same HF
cache shape through `translate.yml`. There is also an
ephemeral `llmlocal-<id>_run-state` (sentinel + model_path).

F1 is the strict-mode Translate contract and uses the locally built
`TRANSLATE_LLM_INIT_IMAGE=llm-init:local` image. The runner builds it from the
current checkout; the published `v1.2.1` image does not support this scenario.
After release, replace only `TRANSLATE_LLM_INIT_IMAGE` with the fixed tag.

Full teardown (removes that scenario's volumes too — the simplest clean
restart, since `down` runs `down -v`):

```sh
# bash
bash examples/local/run.sh A3 down

# PowerShell
.\examples\local\run.ps1 A3 down
```

Wipe only the HF cache for one scenario (forces a fresh re-download so
you can watch the download / phase flow again), then bring it back up:

```sh
docker volume ls --filter name=hf-hub-cache   # see what's cached
docker compose -p llmlocal-a3 stop            # or Ctrl+C the `up`
docker volume rm llmlocal-a3_hf-hub-cache     # delete just this scenario's cache
bash examples/local/run.sh A3                 # up again → re-downloads
```

Wipe every scenario's HF cache at once (the volume must not be in use —
stop the stacks first):

```sh
# bash
docker volume rm $(docker volume ls -q --filter name=hf-hub-cache)
```

```powershell
# PowerShell
docker volume ls -q --filter name=hf-hub-cache | ForEach-Object { docker volume rm $_ }
```

For ollama scenarios (A1, B1, B2, C1) the model lives in
`llmlocal-<id>_ollama-data` instead of `hf-hub-cache`. Wipe one
scenario's pulled models, then bring it back up to re-pull:

```sh
docker volume ls --filter name=ollama-data    # see what's cached
docker compose -p llmlocal-a1 stop            # or Ctrl+C the `up`
docker volume rm llmlocal-a1_ollama-data      # delete just this scenario's models
bash examples/local/run.sh A1                 # up again → re-pulls
```

Wipe every scenario's ollama data at once (stop the stacks first — the
volume must not be in use):

```sh
# bash
docker volume rm $(docker volume ls -q --filter name=ollama-data)
```

```powershell
# PowerShell
docker volume ls -q --filter name=ollama-data | ForEach-Object { docker volume rm $_ }
```

### GPU mode

`deploy/compose/{llamacpp,ollama}.yml` ships with the GPU device
reservation commented out so the CI integration tests can run on a
CPU-only host. For local development, GPU is the common case, so the
runner **auto-enables GPU** when `nvidia-smi -L` succeeds on the host
and layers the right override for the scenario's engine:

| Engine   | Override                    | What changes |
|----------|-----------------------------|--------------|
| llamacpp | `_llamacpp-gpu.compose.yml` | image → `ghcr.io/ggml-org/llama.cpp:server-cuda`, plus the nvidia device reservation on the llamacpp service |
| ollama   | `_ollama-gpu.compose.yml`   | nvidia device reservation on the ollama service (the `ollama/ollama` image already bundles CUDA, ROCm and CPU paths and picks one at boot from what is visible) |

The banner makes the choice explicit, for example:

```
gpu      : on  (:server-cuda; NVIDIA auto-detected on host)
gpu      : on  (NVIDIA auto-detected on host)
gpu      : off (no nvidia-smi on host; set USE_GPU=1 to force)
```

`USE_GPU` is a tri-state override:

```sh
# Force on (skip host probe; useful when nvidia-smi is not on PATH
# but Docker still has --gpus access, e.g. some CI runners).
USE_GPU=1 bash examples/local/run.sh A2
$env:USE_GPU='1'; .\examples\local\run.ps1 A2

# Force off (match the upstream compose defaults, exercise the CPU path).
USE_GPU=0 bash examples/local/run.sh A2
$env:USE_GPU='0'; .\examples\local\run.ps1 A2
```

Pre-flight that Docker can actually reach your GPU (only needed when
you suspect the auto-detection result is wrong):

```sh
docker run --rm --gpus all nvidia/cuda:12.9.1-runtime-ubuntu20.04 nvidia-smi -L
```

A3 / A4 (vllm / sglang) already request GPUs in their base compose
files; `USE_GPU` does nothing there because GPU is the only supported
path. After `up`, expect `/api/diag/gpu` to report a GPU-resident
source (e.g. `vllm_metrics` / `sglang_server_info` / `ollama_ps`) and,
once the model is loaded, a non-zero `size_vram` (ollama `/api/ps`) or
`kv_cache_usage_perc` under inference (others' `/metrics`).

The first `up` against a fresh checkout takes a few minutes for the
initial `docker build`. Subsequent `up`s reuse Docker's layer cache
and are seconds. See [What you observe](#what-you-observe) above for
the exact rebuild triggers and bypasses.

Each scenario runs under a distinct compose project (`-p llmlocal-<id>`)
so volumes never collide. The host port is per-scenario auto-picked
when 8080 is taken, so two scenarios can in principle run side by side
on different ports — but they each load a model into the same Docker
Desktop VM, so resource pressure is the practical limit.

## Scenario matrix

> **Unified config.** Every `.env` here uses the single `MODEL_SOURCE`
> env, with comma-separated entries when a scenario preloads more than
> one source (and `--role mmproj` on the segment that carries a vision
> projector), the single
> `ENGINE_ARGS` env, and `MODEL_MODE` (+ optional `MODEL_SUPPORTS`) to seed
> the model-spec. Unsupported environment variables are ignored.

| ID | Engine | Mechanism | Source / spec | What to look for |
|----|--------|-----------|---------------|------------------|
| A1 | ollama   | ollama-pull          | `ollama://qwen3:0.6b` + chat spec | Daemon `/api/pull` progress, Overview phase `pulling`, Ollama-native `/api/tags` `/api/ps` `/api/show` |
| A2 | llamacpp | hf-cli (single GGUF) | `hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF --include ...gguf` | hfwrap progress events, sentinel handoff, proxy `/v1/chat/completions` |
| A3 | vllm     | hf-cli (full repo)   | `hf://Qwen/Qwen2.5-0.5B-Instruct` | HF tree multi-file concurrent download, `MAX_CONCURRENT_DOWNLOADS=4`, GPU residency on `/api/diag/gpu` |
| A4 | sglang   | hf-cli (full repo)   | Same as A3 | SGLang reverse proxy + `/server_info` mirror |
| B1 | ollama   | ollama-url           | `ollama://https://huggingface.co/.../qwen.gguf` | URL-overload mode: Model Console range-downloads, ollama `/api/blobs` + `/api/create` (from-only Modelfile; metadata self-described) |
| B2 | ollama   | ollama-url           | Same shape as B1, different URL | URL branch + optional `#sha256=` fragment verification |
| B3 | llamacpp | url-escape-hatch     | `https://...` + `MODEL_SOURCE_LOCAL` | URL path under proxy engine (no Modelfile) |
| C1 | ollama   | ollama-pull          | `ollama://nomic-embed-text` + embedding spec (`mode: embedding`) | `/api/endpoints` emphasizes `/v1/embeddings`; catch-all chat acceptance remains an upstream decision |
| C2 | llamacpp | hf-cli (single GGUF) | `hf://Qwen/Qwen3-0.6B-GGUF` + thinking spec (`supports_reasoning: true`) | `chat_template_kwargs.enable_thinking` passthrough; demonstrates `LLAMA_ARG_*=val` env-form `ENGINE_ARGS` |
| D1 | llamacpp | hf-cli (single GGUF) | `hf://this-org-does-not-exist/never-published` | PhaseFailed + LastError, Overview `Retry` button, hfwrap exit code 30 |
| D2 | llamacpp | hf-cli (single GGUF) | reuse A2 cache, request a nonexistent revision | `phase: failed` with the HF revision guidance; cached snapshots remain keyed by repo/revision (see [D2-source-mismatch.md](D2-source-mismatch.md)) |
| D3 | llamacpp | hf-cli (single GGUF) | A2 model via `HF_ENDPOINT=https://hf-mirror.com` | Mirror endpoint round-trip parity with A2; demonstrates `--endpoint` flag is blacklisted in `MODEL_SOURCE` |
| E1 | llamacpp | comma preload        | `MODEL_SOURCE=hf://...Qwen2.5...,hf://...Qwen3...` + `MODEL_SOURCE_LOCAL=,` | First model is served; second model is downloaded to the shared cache only; combined progress covers both |
| E2 | ollama   | comma + URL overload | `MODEL_SOURCE=ollama://...IQ2_XXS.gguf,ollama://...mmproj-BF16.gguf` + `MODEL_SOURCE_LOCAL=,` | Ollama engine; second segment is `extra` (preload only, not attached to the engine); download total prefetched via the Hub tree API (~4.1 GB) |
| F1 | llamacpp | hf-cli (single GGUF) | `tencent/Hy-MT2-1.8B-GGUF` `Hy-MT2-1.8B-Q4_K_M.gguf` + translate spec (`mode: translate`) | Runnable Hy-MT2 translation-model example for MTran-compatible `/languages` and `/translate`; uses `TRANSLATE_LLM_INIT_IMAGE=llm-init:local` because published v1.2.1 lacks strict mode; download and runtime smoke have not been run locally |

### Model spec files

Each scenario's `.env` sets `MODEL_MODE` (chat / embedding / translate), which seeds
the model-spec on boot. To also demonstrate a *fuller* spec (label,
`context_size`, richer `supports`), every scenario layers an override that
mounts one of four reusable specs and points `MODEL_SPEC_PATH` at it so
the mounted file wins over the env seed. The specs ship in
[examples/local/specs/](specs/):

- [`specs/chat.json`](specs/chat.json) -- `mode: chat`. Used by A1-A4, B1-B3, D1, D2, D3, E1, E2.
- [`specs/embedding.json`](specs/embedding.json) -- `mode: embedding`. Used by C1.
- [`specs/thinking.json`](specs/thinking.json) -- `mode: chat` + `supports_reasoning: true`. Used by C2.
- [`specs/translate.json`](specs/translate.json) -- `mode: translate` + `extensions.translate`. Used by F1.

`run.sh` and `run.ps1` both auto-layer the right override for each
scenario. For manual `docker compose` invocations, add the matching
`_spec-*.compose.yml` (chat / embedding / thinking / translate), which both mounts the
file and sets `MODEL_SPEC_PATH`:

```yaml
# examples/local/_spec-chat.compose.yml (and friends)
services:
  llm-init:
    environment:
      MODEL_SPEC_PATH: /etc/llm-init/model-spec.json
    volumes:
      - ./specs/chat.json:/etc/llm-init/model-spec.json:ro
```

Drop the override entirely and the scenario still works: `MODEL_MODE`
(plus optional `MODEL_SUPPORTS`, e.g. `supports_reasoning` for C2) seeds a
valid spec, persisted at `MODEL_SPEC_PATH` and editable from the dashboard
API tab.

### `ENGINE_ARGS` cheat sheet per engine

```sh
# ollama (KEY=VALUE list of OLLAMA_* daemon env)
ENGINE_ARGS=OLLAMA_NUM_CTX=8192 OLLAMA_KEEP_ALIVE=30m

# vllm (cmdline)
ENGINE_ARGS=--max-model-len 8192 --gpu-memory-utilization 0.9

# llama.cpp (cmdline OR LLAMA_ARG_*=val env; auto-detected by wrapper)
ENGINE_ARGS=-c 8192 -ngl all -fa on
ENGINE_ARGS=LLAMA_ARG_CTX_SIZE=8192 LLAMA_ARG_N_GPU_LAYERS=all

# sglang (cmdline)
ENGINE_ARGS=--mem-fraction-static 0.8 --max-running-requests 256
```

Model Console parses `ENGINE_ARGS` by `ENGINE_KIND` and exposes
`{raw, known, unknown}` on `/api/config`. Unknown tokens are silently
passed through to the engine rather than treated as an error.

## Common verification snippets

> Replace `8080` below with the port from the banner's `url :` line if
> the runner had to pick a free port. Or set `PORT=…` once and use
> `$PORT` in the snippets.

```sh
# Lifecycle
curl -s http://localhost:8080/livez
curl -s http://localhost:8080/readyz
curl -s http://localhost:8080/api/progress | jq   # dashboard polls this once per second

# Discovery
curl -s http://localhost:8080/api/endpoints | jq
curl -s http://localhost:8080/api/build-info | jq
curl -s http://localhost:8080/v1/models | jq

# OpenAI chat (A1-A4, B1-B3, C2, D3 once ready)
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<MODEL_NAME from .env>","messages":[{"role":"user","content":"hi"}]}'

# Embedding (C1 only)
curl -s http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"nomic-embed-text","input":"hello"}'

# Translate (F1 only)
curl -s http://localhost:8080/languages | jq
curl -s http://localhost:8080/translate \
  -H 'Content-Type: application/json' \
  -d '{"from":"es","to":"en","text":"Hola, mundo"}'

# Anthropic Messages (works on every backend)
curl -s http://localhost:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"<MODEL_NAME>","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'

# Ollama-native (A1, C1)
curl -s http://localhost:8080/api/tags | jq
curl -s http://localhost:8080/api/ps | jq

# Operator
curl -s -XPOST http://localhost:8080/api/retry        # D1
curl -s -XPOST 'http://localhost:8080/api/retry?force=true&level=sha256'
```

## UI tab expectations per scenario

| Tab          | A1 (ollama-name) | A2-A4 / B/D (file source)    | C1 (embedding) |
|--------------|------------------|------------------------------|----------------|
| Overview     | Phase `pulling` → `ready`; Lifecycle card shows `engine_cold_start_ms` after ready | Phase `init → resolving → downloading → ready`; per-file progress | Same as A1 |
| API          | Lists `/v1/chat/completions`, `/v1/embeddings`, Ollama-native paths | Lists `/v1/chat/completions`, `/v1/messages`, `/v1/responses` | Emphasizes `/v1/embeddings` from model-spec mode |
| Metrics      | req/s + p95 charts populate after first request | Same | Same |

## Source-of-truth files

- [deploy/compose/ollama.yml](../../deploy/compose/ollama.yml) — A1, B1, B2, C1
- [deploy/compose/llamacpp.yml](../../deploy/compose/llamacpp.yml) — A2, B3, C2, D1, D3, E1
- [deploy/compose/vllm.yml](../../deploy/compose/vllm.yml) — A3
- [deploy/compose/sglang.yml](../../deploy/compose/sglang.yml) — A4
- [deploy/compose/translate.yml](../../deploy/compose/translate.yml) — F1

The pre-v1.1.0 `_ollama-files.compose.yml` override is no longer
needed: B1 / B2 use the `ollama://<URL>` form which the standard
`ollama.yml` already supports through Model Console's URL-overload path.

## Notes

- Windows users have two equivalent runners: `run.sh` for bash / WSL /
  git-bash, and `run.ps1` for native PowerShell. Both share the same
  scenario table, image-existence guard, and freshness check (see
  below). The `.env` files also work with plain `docker compose
  --env-file` if you prefer driving each scenario by hand — every
  `.env` header shows the exact command.
- Volumes are scoped per scenario (`llmlocal-a2_hf-hub-cache` +
  `llmlocal-a2_run-state` for file sources; `llmlocal-a1_ollama-data`
  for ollama). Clean a single scenario with
  `bash examples/local/run.sh A2 down`, or wipe just the model cache —
  see [Resetting / cleaning state](#resetting--cleaning-state) above.
- `LOG_LEVEL=debug` and `LOG_FORMAT=text` are set in every `.env` to
  make the boot trail readable on a local terminal; production
  defaults are `info` / `json`.
- After editing Go code, just re-run the scenario — the runner picks
  up the change and rebuilds before `docker compose up`. No manual
  `make docker` step needed. Set `SKIP_REBUILD=1` if you intend to
  reuse the existing image despite source changes.
- The runner only rebuilds the default tag `llm-init:local`. When
  `LLM_INIT_IMAGE` points elsewhere it is left untouched (and run
  will refuse to start if the tag is missing).
