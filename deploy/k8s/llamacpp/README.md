# llama.cpp on Kubernetes

CPU-friendly deployment using engine-native `ENGINE_ARGS`, a shared model
cache, and an optional model-spec ConfigMap.

## Apply order

1. Apply the shared HF cache PVC from `shared-pvc.example.yaml` (single
   GGUFs are small, the example defaults to 20 GiB). The model-spec
   ConfigMap in that file is OPTIONAL: `MODEL_MODE` (+ `MODEL_SUPPORTS`)
   seeds the spec; only ship the ConfigMap (+ mount + `MODEL_SPEC_PATH`)
   for a fuller spec.
2. Create the wrapper ConfigMap once per cluster:

   ```sh
   kubectl create configmap llm-init-wrappers \
     --from-file=deploy/wrappers/common.sh \
     --from-file=deploy/wrappers/llamacpp.sh
   ```

3. Replace placeholders in `pod.yaml`:
   - `<MODEL_NAME>` / `<MODEL_SOURCE>` (e.g.
     `hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --include tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`)
   - `<ENGINE_ARGS>` -- cmdline OR `LLAMA_ARG_*=val` env (auto-detected
     by the wrapper). Common: `-c 4096 -ngl all -fa on`.
   - `<HF_PVC>` / `<LLM_INIT_IMAGE>` (`<SPEC_CONFIGMAP>` only if you
     uncomment the optional model-spec volume/mount + `MODEL_SPEC_PATH`).
     `MODEL_MODE` in `pod.yaml` defaults to `chat`.
4. `kubectl apply -f deploy/k8s/llamacpp/pod.yaml -f deploy/k8s/llamacpp/service.yaml`

## GPU mode

Uncomment the `nvidia.com/gpu` resources block in `pod.yaml`. The
ggml-org server image automatically detects CUDA. Add `-ngl all` (or
`LLAMA_ARG_N_GPU_LAYERS=all`) to `ENGINE_ARGS` to offload all layers.

## Production hardening

See [vLLM "Production hardening"](../vllm/README.md#production-hardening).
llama.cpp models tend to be small (single GGUF, often <10 GB), so
even modest block storage works.
