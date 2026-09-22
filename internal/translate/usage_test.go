// The MTran routes are the only place a token count for a translation exists.
// The gateway forwards the answer without seeing the completion behind it, so a
// route that does not declare its usage is recorded as zero tokens — which is
// what every one of these routes did.

package translate

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// usageMux mounts the routes over an engine that reports a fixed cost per
// completion, and counts the completions so a batch can be told from one call.
func usageMux(t *testing.T, prompt, completion, total int) (*http.ServeMux, *int) {
	t.Helper()
	calls := 0
	engine := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		usage := map[string]any{}
		if prompt != 0 || completion != 0 || total != 0 {
			usage = map[string]any{
				"prompt_tokens":     prompt,
				"completion_tokens": completion,
				"total_tokens":      total,
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "en"}},
			},
			"usage": usage,
		})
	})
	mux := http.NewServeMux()
	Mount(mux, Options{Completer: Completer{Handler: engine, ModelName: "translate-model"}})
	return mux, &calls
}

func post(t *testing.T, mux *http.ServeMux, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("%s status=%d body=%s", path, rec.Code, rec.Body.String())
	}
	return rec
}

func assertUsageHeaders(t *testing.T, rec *httptest.ResponseRecorder, prompt, completion, total string) {
	t.Helper()
	for header, want := range map[string]string{
		"X-Model-Usage-Prompt-Tokens":     prompt,
		"X-Model-Usage-Completion-Tokens": completion,
		"X-Model-Usage-Total-Tokens":      total,
	} {
		if got := rec.Header().Get(header); got != want {
			t.Errorf("%s=%q want %q", header, got, want)
		}
	}
}

func TestTranslateDeclaresWhatTheCompletionCost(t *testing.T) {
	mux, _ := usageMux(t, 11, 7, 18)
	rec := post(t, mux, "/translate", `{"from":"zh-CN","to":"en","text":"你好"}`)
	assertUsageHeaders(t, rec, "11", "7", "18")
}

// A batch is one billable request made of N completions. Declaring only the
// last would price a 64-text batch as one translation.
func TestBatchDeclaresEveryCompletionItRan(t *testing.T) {
	mux, calls := usageMux(t, 11, 7, 18)
	rec := post(t, mux, "/translate/batch", `{"from":"auto","to":"en","texts":["你好","世界","再见"]}`)
	if *calls != 3 {
		t.Fatalf("completions=%d want 3", *calls)
	}
	assertUsageHeaders(t, rec, "33", "21", "54")
}

func TestDetectDeclaresWhatTheModelAnswerCost(t *testing.T) {
	// Latin text, so the pure-CJK heuristic does not short-circuit and a
	// model actually answers.
	mux, calls := usageMux(t, 5, 2, 7)
	rec := post(t, mux, "/detect", `{"text":"hello there"}`)
	if *calls != 1 {
		t.Fatalf("completions=%d want 1", *calls)
	}
	assertUsageHeaders(t, rec, "5", "2", "7")
}

// A heuristic answer never reaches a model, so there is nothing to declare and
// the absent headers say exactly that.
func TestDetectDeclaresNothingWhenAHeuristicAnswered(t *testing.T) {
	mux, calls := usageMux(t, 5, 2, 7)
	rec := post(t, mux, "/detect", `{"text":"你好世界"}`)
	if *calls != 0 {
		t.Fatalf("completions=%d want none — CJK is answered without a model", *calls)
	}
	assertUsageHeaders(t, rec, "", "", "")
}

// "Not reported" and "reported as zero" have to stay distinguishable: an engine
// that says nothing must not be written down as a translation that cost zero.
func TestAnEngineThatReportsNothingWritesNoHeaders(t *testing.T) {
	mux, _ := usageMux(t, 0, 0, 0)
	rec := post(t, mux, "/translate", `{"from":"zh-CN","to":"en","text":"你好"}`)
	assertUsageHeaders(t, rec, "", "", "")
}

// An engine that reports a breakdown but no total gets one derived rather than
// having the whole declaration dropped over the missing field.
func TestATotalIsDerivedWhenTheEngineOmitsIt(t *testing.T) {
	mux, _ := usageMux(t, 11, 7, 0)
	rec := post(t, mux, "/translate", `{"from":"zh-CN","to":"en","text":"你好"}`)
	assertUsageHeaders(t, rec, "11", "7", "18")
}
