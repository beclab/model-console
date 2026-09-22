# Scenario D2 - invalid HF revision with a warm cache

The HF hub cache is keyed by repository and revision, not by
`MODEL_NAME`. Reusing A2's cache with a different source identity does
not trigger a local manifest-name check. This procedure demonstrates the
current failure path with a revision that does not exist.

## Procedure

1. Run A2 until it reaches `ready`, then stop it without deleting its
   cache:

   ```sh
   bash examples/local/run.sh A2
   curl -s http://localhost:8080/api/progress | jq '.phase'
   docker compose -f deploy/compose/llamacpp.yml \
     -f examples/local/_spec-chat.compose.yml \
     --env-file examples/local/A2-llamacpp-hf-gguf.env \
     -p llmlocal-a2 stop
   ```

2. Restart the same project and cache with an invalid revision. Keep the
   A2 spec override mounted so the model card is identical to the normal
   scenario:

   ```sh
   MODEL_SOURCE='hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF --include qwen2.5-0.5b-instruct-q4_k_m.gguf --revision revision-does-not-exist' \
     docker compose -f deploy/compose/llamacpp.yml \
     -f examples/local/_spec-chat.compose.yml \
     --env-file examples/local/A2-llamacpp-hf-gguf.env \
     -p llmlocal-a2 up
   ```

## Expected behaviour

- `/api/progress` reaches `phase: "failed"`.
- `last_error` starts with `HuggingFace revision not found: check
  --revision points to a valid commit, branch, or tag`.
- The existing cached A2 snapshot is left intact, but it does not satisfy
  the requested nonexistent revision.
- The llama.cpp service remains behind the sentinel gate.

## Recovery

Restart A2 with its verified revision, or remove the scenario volumes:

```sh
bash examples/local/run.sh A2 down
```
