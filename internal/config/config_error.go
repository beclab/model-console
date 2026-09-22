package config

import (
	"errors"
	"strings"
)

// FieldError pairs one validation failure with the env/field it concerns.
// Field is best-effort: it's the "FOO:"-style prefix the loaders already
// put on their messages, extracted so structured CLI output and the legacy
// "; "-joined string stay byte-identical (Error just delegates to Err).
type FieldError struct {
	Field string
	Err   error
}

func (e FieldError) Error() string { return e.Err.Error() }
func (e FieldError) Unwrap() error { return e.Err }

// ConfigError aggregates every validation failure from a single Load()
// pass. Error() reproduces the historical "; "-joined message so callers
// and tests that substring-match keep working unchanged; Fields plus
// Unwrap() []error expose the same failures structurally for errors.Is /
// errors.As and for structured CLI rendering.
type ConfigError struct {
	Fields []FieldError
}

func (e *ConfigError) Error() string {
	msgs := make([]string, len(e.Fields))
	for i, fe := range e.Fields {
		msgs[i] = fe.Error()
	}
	return strings.Join(msgs, "; ")
}

func (e *ConfigError) Unwrap() []error {
	errs := make([]error, len(e.Fields))
	for i, fe := range e.Fields {
		errs[i] = fe.Err
	}
	return errs
}

// newConfigError wraps accumulated loader errors into a ConfigError,
// tagging each with its best-effort field prefix. It returns nil for an
// empty slice so callers can invoke it unconditionally. A nested
// *ConfigError (e.g. from validateCrossField / model-spec parsing) is
// flattened into its component FieldErrors rather than re-wrapped, so the
// top-level aggregate stays a single flat list.
func newConfigError(errs []error) error {
	if len(errs) == 0 {
		return nil
	}
	var fields []FieldError
	for _, err := range errs {
		var ce *ConfigError
		if errors.As(err, &ce) {
			fields = append(fields, ce.Fields...)
			continue
		}
		fields = append(fields, FieldError{Field: fieldOf(err), Err: err})
	}
	return &ConfigError{Fields: fields}
}

// joinErrors aggregates field-level validation errors into a structured
// ConfigError whose Error() is the historical "; "-joined string. It is
// the shared sink for the per-section validators (validateCrossField,
// model-spec parsing) and Load's top-level accumulator.
func joinErrors(errs []error) error { return newConfigError(errs) }

// fieldOf extracts the "FOO:" prefix a loader prepends to its message
// (e.g. "MODEL_NAME: required" → "MODEL_NAME"). It returns "" when the
// message has no single-token prefix (cross-field checks phrased as prose).
func fieldOf(err error) string {
	msg := err.Error()
	if i := strings.IndexByte(msg, ':'); i > 0 {
		if head := msg[:i]; !strings.ContainsAny(head, " \t\n") {
			return head
		}
	}
	return ""
}
