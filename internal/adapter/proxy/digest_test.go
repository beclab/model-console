package proxy

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/config"
)

func TestParamDigest(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "the reasoning level a request carried",
			body: `{"model":"m","reasoning_effort":"low","messages":[{"role":"user","content":"hi"}]}`,
			want: "reasoning_effort=low messages=1",
		},
		{
			name: "sampling knobs render by value in a fixed order",
			body: `{"max_tokens":32768,"temperature":0.7,"model":"m","top_k":40,"stream":true}`,
			want: "temperature=0.7 top_k=40 max_tokens=32768 stream=true",
		},
		{
			name: "content is counted, never rendered",
			body: `{"messages":[{"role":"user","content":"my bank password is hunter2"},` +
				`{"role":"assistant","content":"noted"}],"tools":[{"type":"function"}]}`,
			want: "messages=2 tools=1",
		},
		{
			name: "an unlisted key is invisible even when it is a scalar",
			body: `{"model":"m","user":"alice@example.com","prompt":"secret","temperature":1}`,
			want: "temperature=1",
		},
		{
			name: "response_format keeps its type and drops its schema",
			body: `{"response_format":{"type":"json_schema","json_schema":{"name":"patient_record"}}}`,
			want: "response_format=json_schema",
		},
		{
			name: "chat_template_kwargs is reported by key, sorted",
			body: `{"chat_template_kwargs":{"thinking":true,"enable_thinking":false}}`,
			want: "chat_template_kwargs={enable_thinking,thinking}",
		},
		{
			name: "an object under a scalar key is left to the engine to reject",
			body: `{"tool_choice":{"type":"function","function":{"name":"lookup"}}}`,
			want: "",
		},
		{
			name: "a bare string Responses input is not counted as one message",
			body: `{"input":"summarize this","max_output_tokens":64}`,
			want: "max_output_tokens=64",
		},
		{
			name: "a body with nothing on the allowlist says nothing",
			body: `{"model":"m","metadata":{"trace":"abc"}}`,
			want: "",
		},
		{
			name: "not an object",
			body: `["messages"]`,
			want: "",
		},
		{
			name: "not JSON",
			body: `hunter2`,
			want: "",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := paramDigest([]byte(tc.body)); got != tc.want {
				t.Fatalf("paramDigest = %q, want %q", got, tc.want)
			}
		})
	}
}

// A value that could hold whitespace must not be able to split itself into two
// log fields, and one that runs long is cut rather than carried.
func TestParamDigestValuesStaySingleFields(t *testing.T) {
	t.Parallel()
	long := strings.Repeat("x", digestMaxValueLen+10)
	got := paramDigest([]byte(`{"reasoning_effort":"low and slow","verbosity":"` + long + `"}`))
	want := "reasoning_effort=low_and_slow verbosity=" +
		strings.Repeat("x", digestMaxValueLen) + "…"
	if got != want {
		t.Fatalf("paramDigest = %q, want %q", got, want)
	}
	if fields := strings.Fields(got); len(fields) != 2 {
		t.Fatalf("digest split into %d fields: %q", len(fields), got)
	}
}

// The line exists to answer "what did the engine get asked for", so it has to
// carry the parameters and both sides of the model rewrite -- and nothing of
// the conversation.
func TestProxyLogsRequestParamsAtDebug(t *testing.T) {
	u := newUpstream(t)
	a := newProxy(t, u, "target-model", config.EngineLlamaCpp)

	var logs bytes.Buffer
	restore := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(restore) })

	body := `{"model":"alias","reasoning_effort":"low","temperature":0.7,` +
		`"messages":[{"role":"user","content":"my medical history is"}]}`
	w := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(w,
		httptest.NewRequest(http.MethodPost, pathChatCompletions, strings.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}

	line := findLogLine(t, logs.Bytes(), "proxy: forwarded")
	if params, _ := line["params"].(string); !strings.Contains(params, "reasoning_effort=low") ||
		!strings.Contains(params, "temperature=0.7") ||
		!strings.Contains(params, "messages=1") {
		t.Errorf("params = %q, want the level, the temperature and the message count", params)
	}
	if line["model_in"] != "alias" || line["model_out"] != "target-model" {
		t.Errorf("model_in/model_out = %v/%v, want alias/target-model", line["model_in"], line["model_out"])
	}
	if line["upstream_status"] != float64(http.StatusOK) {
		t.Errorf("upstream_status = %v, want 200", line["upstream_status"])
	}
	if strings.Contains(strings.ToLower(logs.String()), "medical") {
		t.Fatalf("the conversation reached the log: %s", logs.String())
	}
}

// At the default level the body is not walked at all: the digest is inside the
// same guard as the line, so an info-level deployment pays nothing.
func TestProxyStaysSilentAboveDebug(t *testing.T) {
	u := newUpstream(t)
	a := newProxy(t, u, "target-model", config.EngineLlamaCpp)

	var logs bytes.Buffer
	restore := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelInfo})))
	t.Cleanup(func() { slog.SetDefault(restore) })

	w := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(w, httptest.NewRequest(http.MethodPost,
		pathChatCompletions, strings.NewReader(`{"model":"alias","reasoning_effort":"low"}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	if strings.Contains(logs.String(), "proxy: forwarded") {
		t.Fatalf("a per-request line escaped above debug: %s", logs.String())
	}
}

// The log wrapper sits in front of every forwarded byte, so the thing to prove
// is that it did not break the request it is describing.
func TestProxyStillRewritesTheModelWhileLogging(t *testing.T) {
	u := newUpstream(t)
	a := newProxy(t, u, "target-model", config.EngineLlamaCpp)

	restore := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(restore) })

	w := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(w, httptest.NewRequest(http.MethodPost,
		pathChatCompletions, strings.NewReader(`{"model":"alias","reasoning_effort":"low"}`)))

	if len(u.gotRequests) != 1 {
		t.Fatalf("upstream saw %d requests, want 1", len(u.gotRequests))
	}
	var forwarded map[string]json.RawMessage
	if err := json.Unmarshal(u.gotRequests[0].body, &forwarded); err != nil {
		t.Fatalf("forwarded body not JSON: %v (%s)", err, u.gotRequests[0].body)
	}
	if string(forwarded["model"]) != `"target-model"` {
		t.Errorf("model = %s, want the configured name", forwarded["model"])
	}
	if string(forwarded["reasoning_effort"]) != `"low"` {
		t.Errorf("reasoning_effort = %s, want it forwarded untouched", forwarded["reasoning_effort"])
	}
	if !strings.Contains(w.Body.String(), "data:") {
		t.Errorf("the SSE stream did not reach the client: %s", w.Body.String())
	}
}

func findLogLine(t *testing.T, logs []byte, msg string) map[string]any {
	t.Helper()
	for _, raw := range strings.Split(strings.TrimSpace(string(logs)), "\n") {
		if raw == "" {
			continue
		}
		var line map[string]any
		if err := json.Unmarshal([]byte(raw), &line); err != nil {
			t.Fatalf("log line is not JSON: %v (%s)", err, raw)
		}
		if line["msg"] == msg {
			return line
		}
	}
	t.Fatalf("no %q line in %s", msg, logs)
	return nil
}
