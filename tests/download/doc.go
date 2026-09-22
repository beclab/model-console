// Package download holds the offline, end-to-end download test suite.
//
// The tests exercise every MODEL_SOURCE channel (url, hf, ollama,
// ollama-url) against the local mocks in faultsrv/ and mockollama/ plus
// the fakehf CLI, covering the abnormal conditions and input modes
// documented in README.md. They are gated behind the `download` build
// tag so the default `go test ./...` stays fast; run them with
// `make test-download` (or `go test -tags download ./tests/download/...`).
package download
