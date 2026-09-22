package translate

// Package-level string constants keep goconst quiet on repeated wire
// codes and JSON keys (same pattern as internal/controlplane/handlers.go).
const (
	langZhHans = "zh-Hans"
	langZhHant = "zh-Hant"

	codeInvalidRequest  = "invalid_request"
	codePayloadTooLarge = "payload_too_large"
	codeTranslationErr  = "translation_error"
	codeRequestCanceled = "request_canceled"

	msgToRequired = "to is required"

	jsonKeyError   = "error"
	jsonKeyCode    = "code"
	jsonKeyMessage = "message"
)
