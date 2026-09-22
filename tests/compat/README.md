# Compatibility tests

Python smoke suite that confirms third-party clients work against
Model Console unchanged. Three tests:

| File | What it covers |
| --- | --- |
| `test_openai_sdk.py` | `openai.Client` chat (streaming + non-streaming), `models.list`, and conditional embeddings. |
| `test_langchain.py`  | `langchain_openai.ChatOpenAI(...).invoke([HumanMessage(...)])`. |
| `test_openwebui.py`  | Streaming `POST /api/chat/completions` alias. |

The suite reuses `deploy/compose/llamacpp.yml` so it runs the exact
deployment template release notes recommend. CI runs it on tag pushes
in [.github/workflows/ci.yml](../../.github/workflows/ci.yml) under the
`compat` job.

## Run locally

```sh
python -m pip install -r tests/compat/requirements.txt

# Bring up the stack yourself, then:
LLM_INIT_URL=http://localhost:8080 \
    pytest tests/compat -v

# Or let the conftest fixture manage the stack (default):
pytest tests/compat -v
```

`LLM_INIT_URL` set ⇒ the conftest skips the compose-up and assumes
something else owns the stack.

## Why pinned versions?

The `requirements.txt` pins are deliberate. Bumping a pin means
re-running the suite end-to-end against the llamacpp compose stack
(see CI matrix in commit M5.7).
