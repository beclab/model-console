package anthropic

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestWriter_Emit_HeadersOnce(t *testing.T) {
	t.Parallel()
	rec := httptest.NewRecorder()
	w := NewWriter(rec)
	if err := w.Emit(EventMessageStart, map[string]any{"x": 1}); err != nil {
		t.Fatalf("emit1: %v", err)
	}
	if err := w.Emit(EventMessageStop, map[string]any{"y": 2}); err != nil {
		t.Fatalf("emit2: %v", err)
	}
	if got := rec.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Errorf("Content-Type=%q", got)
	}
	if rec.Code != 200 {
		t.Errorf("status=%d", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "event: message_start\ndata: {") {
		t.Errorf("missing message_start frame: %s", body)
	}
	if !strings.Contains(body, "event: message_stop\ndata: {") {
		t.Errorf("missing message_stop frame: %s", body)
	}
}

func TestWriter_EmitError(t *testing.T) {
	t.Parallel()
	rec := httptest.NewRecorder()
	w := NewWriter(rec)
	if err := w.EmitError(ErrAPIError, "boom"); err != nil {
		t.Fatalf("emit error: %v", err)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"type":"error"`) {
		t.Errorf("expected type=error in body: %s", body)
	}
	if !strings.Contains(body, `"message":"boom"`) {
		t.Errorf("expected message in body: %s", body)
	}
}

func TestWriteError(t *testing.T) {
	t.Parallel()
	rec := httptest.NewRecorder()
	WriteError(rec, 400, ErrInvalidRequest, "bad")
	if rec.Code != 400 {
		t.Errorf("status=%d", rec.Code)
	}
	body := rec.Body.String()
	for _, want := range []string{`"type":"error"`, `"invalid_request_error"`, `"bad"`} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q in body: %s", want, body)
		}
	}
}
