# Download test suite (offline)

This suite validates the **model download** machinery — every
`MODEL_SOURCE` channel and its abnormal conditions — *before* any real
inference engine is wired up. It runs fully offline: no HuggingFace, no
Ollama daemon, no network, no docker. Three in-process mocks stand in
for the outside world:

| Mock | Replaces | Lives in |
| --- | --- | --- |
| `faultsrv` | an HTTP(S) file server (direct URL + Ollama-URL blobs) | `faultsrv/`, CLI `cmd/faultsrv` |
| `mockollama` | the Ollama daemon (`/api/pull`, `/api/blobs`, `/api/create`, `/api/tags`) | `mockollama/`, CLI `cmd/mockollama` |
| `fakehf` | the HuggingFace `hf` CLI (`hf download …`) | CLI `cmd/fakehf` |

The Go tests drive the **real** `lifecycle.Manager` (its actual `Run`
loop), the **real** `fetch.RangeDownloader`, the **real**
`internal/adapter/ollama` client, and the **real** `internal/adapter/hfwrap`
subprocess runner against these mocks. Each test asserts both the
on-disk result *and* the operator-facing message the frontend reads from
`GET /api/progress` (`last_error`) — because a download that fails for a
bad repo, an unauthorized token, or an invalid Ollama tag must tell the
user exactly what to fix.

## Run the automated suite

```bash
make test-download
# or:
go test -tags download -timeout 5m ./tests/download/...
```

The suite is gated behind the `download` build tag so the default
`go test ./...` stays fast. It takes ~20s (one case deliberately
exhausts the real HTTP retry budget).

## What is covered

### Channel: `url` (`https://…`, direct file)

| Test | Scenario | Expected frontend outcome |
| --- | --- | --- |
| `TestURL_HappyPath_WithSHA` | clean download + SHA-256 verify | `ready`; file on disk; `.part` cleaned up |
| `TestURL_ResumeAfterConnectionReset` | connection dropped mid-stream | resumes via `Range`; `ready` |
| `TestURL_ServerIgnoresRange_RewritesFromZero` | server ignores `Range` (200) | stale `.part` discarded; `ready` |
| `TestURL_ETagRotation_DiscardsStalePart` | ETag changed since last attempt | clean re-download; `ready` |
| `TestURL_5xxStormThenRecover` | first 2 GETs are 503 | inner retry rides it out; `ready`; `transport_retries > 0` |
| `TestURL_Permanent404_FailsWithoutRetry` | wrong URL (404) | `failed`; `last_error` = "Model URL not found (HTTP 404)…"; **no** register |
| `TestURL_429_YieldsDegraded` | upstream throttles (429) | `degraded` (retriable) with a populated `last_error` |
| `TestURL_SHAMismatch_RemovesFileAndDegrades` | bytes don't match `#sha256=` | `degraded`; corrupt file removed |
| `TestURL_MissingLocalPath_FailsPermanently` | `MODEL_SOURCE_LOCAL` unset | `failed`; `last_error` mentions `MODEL_SOURCE_LOCAL` |
| `TestURL_DestAlreadyComplete_Skips` | file already present | `ready`; zero GETs |
| `TestURL_HTTPS_WithTrustingClient` | TLS transport | `ready` |
| `TestURL_ForceRedownloads` | `POST /api/retry?force=true` | wipes + refetches |

### Channel: `hf` (`hf://owner/repo`)

The user's key requirement — **an invalid repo or an unauthorized token
must produce a correct frontend prompt** — is covered two ways:

1. `TestHF_ErrorMessages_ForFrontend` drives the lifecycle dispatch with
   each `hfwrap` error code and asserts the `last_error` the dashboard
   shows:

   | `hfwrap` code | Phase | `last_error` contains |
   | --- | --- | --- |
   | `CodeRepoNotFound` | `failed` | "repository not found" |
   | `CodeGated` | `failed` | "gated" (request access + set `HF_TOKEN`) |
   | `CodeTokenMissing` | `failed` | "authentication failed" |
   | `CodeRevision` | `failed` | "revision not found" |
   | `CodeDiskFull` | `failed` | "disk full" |
   | `CodePermission` | `failed` | "permission denied" |
   | `CodeNetwork` | `degraded` | "network error" (retriable) |

2. `TestHF_Subprocess_ErrorClassification` runs the **real** `hfwrap.Run`
   subprocess against `fakehf`, feeding it the stderr each real failure
   emits, and asserts `classify.go` maps it to the right code +
   retriability. Together they prove the full chain: real `hf` stderr →
   classification → operator message.

Other HF tests: `TestHF_HappyPath`, `TestHF_TokenAndEndpointPlumbedToWrapper`
(HF_TOKEN / HF_ENDPOINT reach the wrapper), `TestHF_MultiSource_MainPlusMmproj`,
`TestHF_Subprocess_HappyPath` (cache layout + resolved commit),
`TestHF_Subprocess_SingleFile`.

### Channel: `ollama` (`ollama://tag`) and `ollama-url`

| Test | Scenario | Expected frontend outcome |
| --- | --- | --- |
| `TestOllamaName_PullHappyPath` | valid tag | `ready` |
| `TestOllamaName_UnknownTag_404` | invalid tag (404) | `degraded`; `last_error` = "Ollama model … not found in the library…" |
| `TestOllamaName_Denied_403` | pull denied (403) | `degraded`; `last_error` = "Ollama denied access…" |
| `TestOllamaName_NoSuccessFrame_RecoversViaTags` | stream ends early but model present | `ready` (re-check `/api/tags`) |
| `TestOllamaName_NoSuccessFrame_AbsentModel_Degrades` | stream ends early, model absent | `degraded` |
| `TestOllamaURL_DownloadThenRegister` | URL blob → blob push → create | `ready` |
| `TestOllamaURL_CreateNoSuccess_Fails` | `/api/create` ends without success | `failed` (terminal, loading phase); `last_error` mentions create |
| `TestOllamaURL_BlobPushDenied_Fails` | `/api/blobs` returns 403 | `failed` (terminal, loading phase) |

### Config + control plane

`TestConfig_InvalidInput_GivesClearErrors` asserts that bad `MODEL_SOURCE`
input (wrong scheme, `hf://noslash`, URL without `MODEL_SOURCE_LOCAL`,
inline flags on url/ollama, malformed `#sha256=`, multi-source without a
`ROLE=main`, …) is rejected at boot with an actionable message.
`TestControlPlane_*` asserts `GET /api/progress` exposes `last_error`,
`POST /api/retry?force=true` forwards the force flag, and the retry rate
limiter returns `429` once the burst is spent.

## Mock CLIs (for manual reproduction)

Each mock has a standalone binary so a colleague can reproduce any
scenario by hand against the real `llm-init` binary.

```bash
# Fault HTTP server: resume, 5xx storm, 404, 429, throttling, ETag churn.
go run ./tests/download/cmd/faultsrv -addr 127.0.0.1:9000 -size 1048576 \
    -support-range -fail-first 2          # first 2 GETs 503, then OK
go run ./tests/download/cmd/faultsrv -addr 127.0.0.1:9000 -force-status 404
go run ./tests/download/cmd/faultsrv -addr 127.0.0.1:9000 -drop-after 65536 -drop-first 1

# Mock Ollama daemon: unknown tag, denied pull, missing success frame.
go run ./tests/download/cmd/mockollama -addr 127.0.0.1:11434 -pull-status 404
go run ./tests/download/cmd/mockollama -addr 127.0.0.1:11434 -pull-status 403

# Fake hf CLI: drop it on PATH and the real hfwrap runner shells out to it.
go build -o /tmp/hfbin/hf ./tests/download/cmd/fakehf
FAKEHF_STDERR='GatedRepoError: Cannot access gated repo' FAKEHF_EXIT=1 \
    /tmp/hfbin/hf download owner/repo --cache-dir /tmp/cache
```

The helper scripts in `scripts/` wire these together end-to-end against
the real binary; see the next section.

## Manual reproduction with the real binary

The real binary needs `MODEL_MODE` (to seed its model-spec) and a writable
`RUN_DIR` (the spec is seeded to `${RUN_DIR}/model-spec.json`). The scripts
under `scripts/` export `MODEL_MODE=chat`, start the relevant mock, boot
`llm-init`, and poll `GET /api/progress` so you can watch the phase +
`last_error` the dashboard would render:

```bash
bash tests/download/scripts/url-faults.sh     404      # permanent 404
bash tests/download/scripts/hf-errors.sh      gated    # gated repo prompt
bash tests/download/scripts/ollama-errors.sh  404      # invalid tag prompt
```

Each script prints the polled `/api/progress` JSON so you can confirm the
operator-facing message matches the table above. They are Linux/bash and
require `go` on PATH; the model-spec is seeded from `MODEL_MODE` into the
per-run temp `RUN_DIR`, so no `/etc` file or `sudo` is needed.
