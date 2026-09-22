# Ollama on Kubernetes

The Ollama daemon owns the model bytes -- no shared HF cache needed
on the daemon container; Model Console talks to the daemon via HTTP API
only. The Pod uses engine-native `ENGINE_ARGS` and can seed a model spec
from environment variables.

## Apply order

1. Apply the Ollama daemon PVC from `shared-pvc.example.yaml`. The PVC is
   RWO (one per Pod) -- the Ollama daemon's lock semantics around
   `/root/.ollama` are not officially supported across multiple daemons
   sharing a volume. The model-spec ConfigMap in that file is OPTIONAL:
   `MODEL_MODE` (+ `MODEL_SUPPORTS`) seeds the spec, so you only need the
   ConfigMap (+ its mount + `MODEL_SPEC_PATH`) to ship a fuller spec.

2. Create the wrapper ConfigMap once per cluster:

   ```sh
   kubectl create configmap llm-init-wrappers \
     --from-file=deploy/wrappers/common.sh \
     --from-file=deploy/wrappers/ollama.sh
   ```

3. Replace placeholders in `pod.yaml`:
   - `<MODEL_NAME>` -- client alias the OpenAI `model` field carries.
   - `<MODEL_SOURCE>` -- `ollama://<library-tag>` (daemon pulls), or
     `ollama://<URL>` (URL-overload mode: Model Console range-downloads
     and POSTs to `/api/blobs` + `/api/create`).
   - `<ENGINE_ARGS>` -- KEY=VALUE list of `OLLAMA_*` daemon env. Common:
     `OLLAMA_NUM_CTX=8192 OLLAMA_KEEP_ALIVE=30m OLLAMA_NUM_PARALLEL=2`.
   - `<OLLAMA_PVC>` / `<LLM_INIT_IMAGE>` (`<SPEC_CONFIGMAP>` only if you
     uncomment the optional model-spec volume/mount + `MODEL_SPEC_PATH`).
     `MODEL_MODE` in `pod.yaml` defaults to `chat`; set `embedding` as needed.
4. `kubectl apply -f deploy/k8s/ollama/pod.yaml -f deploy/k8s/ollama/service.yaml`

## Switching models

Update `MODEL_NAME` (client alias) and `MODEL_SOURCE` (e.g.
`ollama://qwen2.5:7b-instruct` -> `ollama://llama3:8b`), plus
`MODEL_MODE`/`MODEL_SUPPORTS` if the profile changed. If you ship a full
spec via the optional ConfigMap, replace its content too. The daemon keeps
the old model blobs on the PVC so a rollback is "swap `MODEL_SOURCE` back".

The pre-v1.1 distinction between `MODEL_NAME` and `OLLAMA_MODEL` envs
is now folded into `MODEL_SOURCE`: the URL part of `ollama://<tag>` is
the upstream tag, `MODEL_NAME` is the client alias, and they're free
to differ (or to be set equal for a single-name deploy).

## Production hardening

See [vLLM "Production hardening"](../vllm/README.md#production-hardening).
Two specifics:

- `storageClassName` on the Ollama daemon PVC: pick the right RWO class.
- `imagePullPolicy`: pin explicitly when running on a pinned Ollama tag
  so node-cached images survive restarts.
