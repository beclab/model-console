# SGLang on Kubernetes

Same workflow as [vllm](../vllm/README.md), using engine-native
`ENGINE_ARGS`, a shared model cache, and an optional model-spec ConfigMap.

## Apply order

1. Apply the shared HF cache PVC from `shared-pvc.example.yaml`. The
   model-spec ConfigMap there is OPTIONAL: `MODEL_MODE` (+ `MODEL_SUPPORTS`)
   seeds the spec; only ship the ConfigMap (+ mount + `MODEL_SPEC_PATH`)
   for a fuller spec.
2. Create the wrapper ConfigMap once per cluster (same ConfigMap as
   vLLM / llamacpp / ollama):

   ```sh
   kubectl create configmap llm-init-wrappers \
     --from-file=deploy/wrappers/common.sh \
     --from-file=deploy/wrappers/sglang.sh
   ```

3. Replace placeholders in `pod.yaml`:
   - `<MODEL_NAME>` / `<MODEL_SOURCE>`
   - `<ENGINE_ARGS>` -- e.g. `--mem-fraction-static 0.8 --max-running-requests 256`
   - `<HF_PVC>` / `<LLM_INIT_IMAGE>` (`<SPEC_CONFIGMAP>` only if you
     uncomment the optional model-spec volume/mount + `MODEL_SPEC_PATH`).
     `MODEL_MODE` in `pod.yaml` defaults to `chat`.
4. `kubectl apply -f deploy/k8s/sglang/pod.yaml -f deploy/k8s/sglang/service.yaml`

## Differences from vLLM

- Engine port is 30000 (SGLang default) vs vLLM's 8000.
- Wrapper invokes `python -m sglang.launch_server`.
- All other plumbing (probe paths, sentinel, model_path file) is identical.

## Production hardening

See [vLLM "Production hardening"](../vllm/README.md#production-hardening).
Same trade-offs apply.
