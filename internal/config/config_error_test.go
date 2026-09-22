package config

import (
	"errors"
	"strings"
	"testing"
)

// A Load() that fails on several fields at once must surface a structured
// *ConfigError carrying one FieldError per failure, while its Error()
// stays byte-compatible with the old "; "-joined string callers/tests
// substring-match against.
func TestLoad_StructuredConfigError(t *testing.T) {
	env := minimalEnv(nil)
	delete(env, "MODEL_NAME")
	delete(env, "MODEL_SOURCE")

	_, err := Load(mapEnv(env))
	if err == nil {
		t.Fatal("Load: want error, got nil")
	}

	var ce *ConfigError
	if !errors.As(err, &ce) {
		t.Fatalf("err = %T, want *ConfigError", err)
	}
	if len(ce.Fields) < 2 {
		t.Fatalf("Fields = %d, want >= 2 (MODEL_NAME + MODEL_SOURCE)", len(ce.Fields))
	}

	var sawName, sawSource bool
	for _, fe := range ce.Fields {
		switch fe.Field {
		case "MODEL_NAME":
			sawName = true
		case "MODEL_SOURCE":
			sawSource = true
		}
	}
	if !sawName || !sawSource {
		t.Fatalf("Fields lack expected prefixes: %+v", ce.Fields)
	}

	// Error() text must still substring-match both fields, like before.
	for _, want := range []string{"MODEL_NAME", "MODEL_SOURCE"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("Error() = %q, missing %q", err.Error(), want)
		}
	}

	// Unwrap() []error exposes the same failures for errors.As traversal.
	if got := len(ce.Unwrap()); got != len(ce.Fields) {
		t.Errorf("Unwrap len = %d, want %d", got, len(ce.Fields))
	}
}

// A nested *ConfigError (validateCrossField / model-spec parsing feed one
// into Load's accumulator) must be flattened, not double-wrapped.
func TestNewConfigError_FlattensNested(t *testing.T) {
	inner := &ConfigError{Fields: []FieldError{
		{Field: "A", Err: errors.New("A: bad")},
		{Field: "B", Err: errors.New("B: bad")},
	}}
	out := newConfigError([]error{inner, errors.New("C: bad")})

	var ce *ConfigError
	if !errors.As(out, &ce) {
		t.Fatalf("out = %T, want *ConfigError", out)
	}
	if len(ce.Fields) != 3 {
		t.Fatalf("Fields = %d, want 3 (flattened)", len(ce.Fields))
	}
	if want := "A: bad; B: bad; C: bad"; ce.Error() != want {
		t.Errorf("Error() = %q, want %q", ce.Error(), want)
	}
}
