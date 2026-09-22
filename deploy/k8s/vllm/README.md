# vLLM on Kubernetes

The deployment uses engine-native `ENGINE_ARGS`, a shared model cache, and
an optional model-spec ConfigMap.

## Apply order

1. **Apply the shared HF cache PVC.** Use `shared-pvc.example.yaml` as a
   starting point. The model-spec ConfigMap in that file is OPTIONAL --
   `MODEL_MODE` (+ optional `MODEL_SUPPORTS`) in `pod.yaml` seeds the spec
   on boot, so you only need the ConfigMap (and the matching mount +
   `MODEL_SPEC_PATH` env) to ship a fuller spec (label/context_size/pricing):

   ```sh
   sed \
     -e 's|<your-rwx-storage-class>|cephfs|' \
     -e 's|<MODEL_NAME>|qwen2.5-7b-instruct|' \
     deploy/k8s/vllm/shared-pvc.example.yaml | kubectl apply -f -
   ```

2. **Create the wrapper ConfigMap once per cluster.** The same ConfigMap
   serves all four engines, so this is a one-time step:

   ```sh
   kubectl create configmap llm-init-wrappers \
     --from-file=deploy/wrappers/common.sh \
     --from-file=deploy/wrappers/vllm.sh
   ```

3. **Replace placeholders in `pod.yaml`** using `sed -i` or your
   templating tool of choice:

   - `<MODEL_NAME>` -- e.g. `qwen2.5-7b-instruct` (client alias)
   - `<MODEL_SOURCE>` -- e.g. `hf://Qwen/Qwen2.5-7B-Instruct`
   - `<ENGINE_ARGS>` -- e.g. `--max-model-len 8192 --gpu-memory-utilization 0.9`
     (engine-native command line)
   - `<HF_PVC>` -- e.g. `hf-hub-cache` (matches the PVC applied in step 1)
   - `<SPEC_CONFIGMAP>` -- (optional) e.g. `model-spec-qwen2.5-7b-instruct`,
     only if you uncomment the model-spec volume/mount + `MODEL_SPEC_PATH`
   - `<LLM_INIT_IMAGE>` -- pinned image, e.g. `docker.io/beclab/llm-init:v1.1.0`

   The `MODEL_MODE` env in `pod.yaml` defaults to `chat`; set it to
   `embedding` for an embedding model, and list enabled capabilities in
   `MODEL_SUPPORTS` (comma-separated) if needed.

4. **Apply the manifests:**

   ```sh
   kubectl apply -f deploy/k8s/vllm/pod.yaml
   kubectl apply -f deploy/k8s/vllm/service.yaml
   ```

5. **Tail the boot:**

   ```sh
   kubectl logs -f <MODEL_NAME> -c llm-init
   kubectl logs -f <MODEL_NAME> -c engine
   ```

6. **Port-forward and probe:**

   ```sh
   kubectl port-forward svc/<MODEL_NAME> 8080
   curl localhost:8080/readyz
   curl localhost:8080/v1/models
   curl localhost:8080/api/config | jq '.engine.args'   # see ENGINE_ARGS parsed
   ```

## Switching models

1. Update `MODEL_NAME` (alias clients send in `model`), `MODEL_SOURCE`, and
   `MODEL_MODE`/`MODEL_SUPPORTS` if the capability profile changed.
2. If you ship a full spec via the optional ConfigMap, replace its content
   with the new model's `model-spec.json`. (The runtime spec also lives at
   `MODEL_SPEC_PATH` and is editable via the dashboard / `PUT /api/model-spec`.)
3. `kubectl rollout restart pod/<old-name>` (or delete + apply if you
   renamed the Pod).

The HF cache PVC keeps every previously-downloaded model under
`/cache/hf/hub/models--owner--repo/...`, so a rollback is "swap
`MODEL_SOURCE` back; the snapshot dir is still cached".

## Production hardening

The bundled `pod.yaml` is a starting template, not a production preset.
Two fields commonly need to change before going live:

### `storageClassName` on the PVC

The shipped `shared-pvc.example.yaml` placeholder is unset; pick a
RWX-capable provider for cross-Pod dedup. RWO is fine for single-Pod.

| Cloud / runtime | RWX-capable values |
|-----------------|--------------------|
| GKE             | `standard-rwx`, Filestore CSI |
| EKS             | `efs-sc` (EFS CSI driver) |
| AKS             | Azure Files CSI |
| OpenShift       | `ocs-storagecluster-cephfs` |
| On-prem (Ceph)  | `rook-cephfs` |
| On-prem (NFS)   | the StorageClass your `nfs-client-provisioner` exposes |

### `imagePullPolicy`

Defaults under k8s are surprising: `:latest` -> `Always`, any other tag
-> `IfNotPresent`. For deterministic rollouts pick once and pin:

- **`:latest` floating tag (dev / fast iteration):** keep the implicit
  `Always`.
- **Pinned semver (`:v1.1.0`, `:v1.0.9`, etc.) for production:**
  set `imagePullPolicy: IfNotPresent` explicitly so node-cached
  images are reused across pod restarts and the registry isn't a
  rollout dependency.

`pod.yaml` sets `IfNotPresent` for both containers; bump to `Always`
deliberately if you actually want re-pulls on restart.
