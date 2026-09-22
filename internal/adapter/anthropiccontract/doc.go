// Package anthropiccontract holds cross-engine contract tests that lock
// the shared POST /v1/messages behaviour of the two adapters that
// implement it: proxy (OpenAI /v1/chat/completions backends — vLLM /
// llama.cpp / SGLang) and ollama (NDJSON /api/chat).
//
// The package has no production code. It exists because each adapter's
// own *_test.go can only exercise that adapter in isolation, so there
// was no place asserting the two produce the SAME Anthropic-shaped
// output for the same input. That equivalence is the contract clients
// (anthropic-sdk-python / Claude Code) rely on when they point at
// llm-init without knowing which engine is behind it.
package anthropiccontract
