package obs

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"
)

func TestRedact_TableDriven(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"empty", "", ""},
		{"plain text", "nothing to redact", "nothing to redact"},
		{
			"bearer header",
			"Authorization: Bearer abc123def456",
			"Authorization: Bearer ***",
		},
		{
			"bearer mixed case",
			"bearer ABC.def-XYZ_123",
			"Bearer ***",
		},
		{
			"hf token",
			"token=hf_abcdefghijklmnopqrstuvwxyz1234",
			"token=hf_***",
		},
		{
			"hf token bare",
			"got hf_abcdefghijklmnopqrstuvwxyz1234 from env",
			"got hf_*** from env",
		},
		{
			"openai sk token",
			"key=sk-abcdefghij1234567890klmn",
			"key=sk-***",
		},
		{
			"query token",
			"https://example.com/path?token=abc&x=1",
			"https://example.com/path?token=***&x=1",
		},
		{
			"query api_key",
			"https://example.com/?api_key=secret&v=1",
			"https://example.com/?api_key=***&v=1",
		},
		{
			"query sig",
			"https://s3/bucket?x-amz-signature=deadbeef&x-amz-date=20260101",
			"https://s3/bucket?x-amz-signature=***&x-amz-date=20260101",
		},
		{
			"multiple in one string",
			"call Bearer xxx with hf_abcdefghijklmnopqrstuvwxyz1234",
			"call Bearer *** with hf_***",
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := Redact(tc.in)
			if got != tc.want {
				t.Errorf("\n got: %q\nwant: %q", got, tc.want)
			}
		})
	}
}

func TestRedact_BearerWithoutFollowingTokenChars(t *testing.T) {
	t.Parallel()
	// When nothing token-like follows "Bearer" the regex does not match;
	// e.g. punctuation breaks the [a-z0-9._\-]+ run.
	const in = "use Bearer."
	if got := Redact(in); got != in {
		t.Errorf("got %q, want %q", got, in)
	}
}

func TestRedactingHandler_RedactsMessage(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	logger := slog.New(NewRedactingHandler(slog.NewJSONHandler(&buf, nil)))

	logger.Info("calling Bearer abcdef123 with hf_abcdefghijklmnopqrstuvwxyz1234")

	got := buf.String()
	if strings.Contains(got, "abcdef123") {
		t.Errorf("Bearer token leaked: %s", got)
	}
	if strings.Contains(got, "hf_abcdefghijklmnopqrstuvwxyz1234") {
		t.Errorf("HF token leaked: %s", got)
	}
}

func TestRedactingHandler_RedactsAttrs(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	logger := slog.New(NewRedactingHandler(slog.NewJSONHandler(&buf, nil)))

	logger.Info("ok",
		slog.String("auth", "Bearer abcdef123456"),
		slog.Int("retry", 2),
	)

	var out map[string]any
	if err := json.Unmarshal(buf.Bytes(), &out); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got := out["auth"]; got != RedactedBearer {
		t.Errorf("auth = %v, want %q", got, RedactedBearer)
	}
	if got := out["retry"]; got != float64(2) {
		t.Errorf("retry = %v, want 2 (untouched)", got)
	}
}

func TestRedactingHandler_RedactsGroupAndWithAttrs(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	base := slog.New(NewRedactingHandler(slog.NewJSONHandler(&buf, nil)))
	logger := base.With(slog.String("static_token", "hf_abcdefghijklmnopqrstuvwxyz1234"))
	logger.Info("ok",
		slog.Group("hf",
			slog.String("token", "hf_zyxwvutsrqponmlkjihgfedcba9999"),
		),
	)

	out := buf.String()
	if strings.Contains(out, "hf_abcdefghijklmnopqrstuvwxyz1234") {
		t.Errorf("static token leaked: %s", out)
	}
	if strings.Contains(out, "hf_zyxwvutsrqponmlkjihgfedcba9999") {
		t.Errorf("group token leaked: %s", out)
	}
}

func TestRedactingHandler_Enabled(t *testing.T) {
	t.Parallel()
	inner := slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelWarn})
	h := NewRedactingHandler(inner)
	if h.Enabled(context.Background(), slog.LevelDebug) {
		t.Error("Debug should be filtered when inner is Warn")
	}
	if !h.Enabled(context.Background(), slog.LevelError) {
		t.Error("Error should pass through")
	}
}

func TestRedactingHandler_WithGroup(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	logger := slog.New(NewRedactingHandler(slog.NewJSONHandler(&buf, nil))).WithGroup("api")
	logger.Info("ok", slog.String("auth", "Bearer abcdef0123"))
	if !strings.Contains(buf.String(), `"api":`) {
		t.Errorf("group missing: %s", buf.String())
	}
	if strings.Contains(buf.String(), "abcdef0123") {
		t.Errorf("group leak: %s", buf.String())
	}
}

func TestHasSensitiveMarker(t *testing.T) {
	t.Parallel()
	if HasSensitiveMarker("hello") {
		t.Error("plain string should not match")
	}
	if !HasSensitiveMarker("Bearer xxx") {
		t.Error("Bearer should match")
	}
	if !HasSensitiveMarker("hf_xxx") {
		t.Error("hf_ should match")
	}
	if !HasSensitiveMarker("sk-xxx") {
		t.Error("sk- should match")
	}
}
