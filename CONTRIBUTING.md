# Contributing to Model Console

Thank you for improving Model Console. This guide covers the repository's
development and review workflow. Runtime behavior is defined by the source,
tests, deployment examples, and the public HTTP handlers.

## Development environment

Install the Go version pinned in `.tool-versions`, Docker with Compose, and the
version of golangci-lint used by `.github/workflows/ci.yml`.

```sh
go mod download
go mod verify
make build
./bin/llm-init --version
```

The module path intentionally remains `github.com/llm-init/llm-init` for
compatibility. Do not rename the module, command, image, environment variables,
runtime paths, or metrics as part of unrelated changes.

## Testing

Run the narrowest relevant tests while developing, then run the complete local
release gates before submitting a change:

```sh
go test ./internal/config/...
make verify
make compose-validate
make k8s-validate
```

`make verify` runs vet, race-enabled unit tests, the offline download suite,
wrapper tests, and golangci-lint. Integration and SDK compatibility suites use
Docker and are documented beside their test code:

- `tests/integration/README.md`
- `tests/compat/README.md`
- `tests/download/README.md`

Changes to public behavior must include tests for successful requests, invalid
input, lifecycle transitions, and compatibility with existing configuration.
Do not weaken coverage or security gates to make a change pass.

## Pull requests

- Create a focused branch from `main`.
- Use Conventional Commits with English commit messages.
- Keep generated files and unrelated formatting out of the change.
- Explain behavior changes, compatibility impact, and verification performed.
- Update README, deployment examples, and tests when their behavior changes.
- Never commit credentials, tokens, model weights, or private endpoints.

## Code style

Use `gofmt` and `goimports` through `make fmt`. Prefer small packages, explicit
errors, deterministic tests, and standard-library facilities where practical.
Public HTTP responses and logs must not expose secrets or authorization data.
