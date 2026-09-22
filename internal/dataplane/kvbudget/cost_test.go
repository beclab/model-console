package kvbudget

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOutputReserve(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		cardMax int
		want    int
	}{
		{
			// The reserve a request has to be charged when it says nothing
			// is the one number here nobody can derive. Charging the rest
			// of the window would let one caller hold the whole pool for a
			// completion that usually ends in a few hundred tokens.
			name: "silent request gets the default",
			body: `{"messages":[]}`,
			want: defaultOutputReserve,
		},
		{
			name: "max_tokens is honoured",
			body: `{"max_tokens":512}`,
			want: 512,
		},
		{
			// Newer OpenAI clients send only this spelling, so reading
			// max_tokens alone would charge every one of them the default.
			name: "max_completion_tokens is honoured",
			body: `{"max_completion_tokens":700}`,
			want: 700,
		},
		{
			// max_tokens is the older name and the one an engine reads
			// when both arrive, so it decides here too.
			name: "max_tokens wins over max_completion_tokens",
			body: `{"max_tokens":300,"max_completion_tokens":900}`,
			want: 300,
		},
		{
			// Reserving tokens the engine will not generate takes pool
			// away from callers who could have used it.
			name:    "a request cannot reserve past the card's ceiling",
			body:    `{"max_tokens":100000}`,
			cardMax: 4096,
			want:    4096,
		},
		{
			name:    "a low ceiling also caps the default",
			body:    `{}`,
			cardMax: 256,
			want:    256,
		},
		{
			name:    "a high ceiling does not raise the default",
			body:    `{}`,
			cardMax: 65536,
			want:    defaultOutputReserve,
		},
		{
			// Streaming clients send max_tokens: 0 to mean "no limit".
			name: "zero is not a ceiling",
			body: `{"max_tokens":0}`,
			want: defaultOutputReserve,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var shape requestShape
			if err := json.Unmarshal([]byte(tc.body), &shape); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got := outputReserve(shape, tc.cardMax); got != tc.want {
				t.Errorf("outputReserve() = %d, want %d", got, tc.want)
			}
		})
	}
}

// TestPromptUpperBoundIsABound is the property the fast path rests on. If a
// body can tokenize to more tokens than this, the fast path admits requests
// on a number that is not a bound and the accounting is unsound.
func TestPromptUpperBoundIsABound(t *testing.T) {
	// A BPE vocabulary contains single-byte tokens, so one token per byte
	// is the worst case and the bound has to allow for it.
	for _, body := range []string{"", "a", strings.Repeat("a", 1000), "你好"} {
		if got := promptUpperBound([]byte(body)); got < len([]byte(body)) {
			t.Errorf("promptUpperBound(%d bytes) = %d, below one token per byte", len(body), got)
		}
	}
}

func TestBodyOfRestoresTheBody(t *testing.T) {
	const body = `{"messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))

	raw, err := bodyOf(r)
	if err != nil {
		t.Fatalf("bodyOf: %v", err)
	}
	if string(raw) != body {
		t.Errorf("bodyOf returned %q, want %q", raw, body)
	}
	again, err := readAll(r)
	if err != nil {
		t.Fatalf("re-read: %v", err)
	}
	if string(again) != body {
		t.Errorf("the restored body is %q, want %q", again, body)
	}
	if r.ContentLength != int64(len(body)) {
		t.Errorf("ContentLength = %d, want %d", r.ContentLength, len(body))
	}
}

func TestBodyOfHandlesNoBody(t *testing.T) {
	r := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	r.Body = nil
	raw, err := bodyOf(r)
	if err != nil || raw != nil {
		t.Errorf("bodyOf(no body) = (%q, %v), want (nil, nil)", raw, err)
	}
}

// TestCountPromptChatTakesTwoHops locks the shape of the chat path: the
// template has to be applied before the count, because /tokenize has no
// notion of a message list and would silently count the wrong thing if
// handed one.
func TestCountPromptChatTakesTwoHops(t *testing.T) {
	var templateBody, tokenizeBody string
	mux := http.NewServeMux()
	mux.HandleFunc("/apply-template", func(w http.ResponseWriter, r *http.Request) {
		raw, _ := readAll(r)
		templateBody = string(raw)
		writeJSON(w, map[string]any{"prompt": "<|user|>hi<|assistant|>"})
	})
	mux.HandleFunc("/tokenize", func(w http.ResponseWriter, r *http.Request) {
		raw, _ := readAll(r)
		tokenizeBody = string(raw)
		writeJSON(w, map[string]any{"tokens": []int{1, 2, 3, 4, 5}})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	tk := &Tokenizer{BaseURL: srv.URL}
	var shape requestShape
	if err := json.Unmarshal([]byte(`{"messages":[{"role":"user","content":"hi"}]}`), &shape); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	n, err := tk.CountPrompt(context.Background(), shape)
	if err != nil {
		t.Fatalf("CountPrompt: %v", err)
	}
	if n != 5 {
		t.Errorf("CountPrompt = %d, want 5", n)
	}
	if !strings.Contains(templateBody, `"role":"user"`) {
		t.Errorf("/apply-template got %q, want the message list unchanged", templateBody)
	}
	var sent struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal([]byte(tokenizeBody), &sent); err != nil {
		t.Fatalf("decode /tokenize body %q: %v", tokenizeBody, err)
	}
	if sent.Content != "<|user|>hi<|assistant|>" {
		t.Errorf("/tokenize content = %q, want the templated prompt", sent.Content)
	}
}

// TestCountPromptCompletionsTakesOneHop: a prompt string is already what
// /tokenize wants, so paying for the template hop would be paying for a
// transformation of nothing.
func TestCountPromptCompletionsTakesOneHop(t *testing.T) {
	templateHits := 0
	mux := http.NewServeMux()
	mux.HandleFunc("/apply-template", func(w http.ResponseWriter, r *http.Request) {
		templateHits++
		writeJSON(w, map[string]any{"prompt": ""})
	})
	mux.HandleFunc("/tokenize", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{"tokens": []int{1, 2, 3}})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	tk := &Tokenizer{BaseURL: srv.URL}
	var shape requestShape
	if err := json.Unmarshal([]byte(`{"prompt":"once upon a time"}`), &shape); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	n, err := tk.CountPrompt(context.Background(), shape)
	if err != nil {
		t.Fatalf("CountPrompt: %v", err)
	}
	if n != 3 {
		t.Errorf("CountPrompt = %d, want 3", n)
	}
	if templateHits != 0 {
		t.Errorf("/apply-template was called %d times for a plain prompt", templateHits)
	}
}

// TestCountPromptSkipsWhatItCannotCount: a batched prompt array and a
// token-id prompt are both real /v1/completions bodies this gate does not
// model, and reporting zero sends the caller to the byte bound rather than to
// a count of the wrong thing.
func TestCountPromptSkipsWhatItCannotCount(t *testing.T) {
	hits := 0
	mux := http.NewServeMux()
	mux.HandleFunc("/tokenize", func(w http.ResponseWriter, r *http.Request) {
		hits++
		writeJSON(w, map[string]any{"tokens": []int{1}})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	tk := &Tokenizer{BaseURL: srv.URL}
	for _, body := range []string{
		`{"prompt":["a","b"]}`,
		`{"prompt":[1,2,3]}`,
		`{}`,
	} {
		var shape requestShape
		if err := json.Unmarshal([]byte(body), &shape); err != nil {
			t.Fatalf("unmarshal %s: %v", body, err)
		}
		n, err := tk.CountPrompt(context.Background(), shape)
		if err != nil {
			t.Errorf("CountPrompt(%s) errored: %v", body, err)
		}
		if n != 0 {
			t.Errorf("CountPrompt(%s) = %d, want 0", body, n)
		}
	}
	if hits != 0 {
		t.Errorf("/tokenize was called %d times for bodies with nothing countable", hits)
	}
}

func TestCountArray(t *testing.T) {
	tests := []struct {
		name    string
		raw     string
		want    int
		wantErr bool
	}{
		{name: "empty array", raw: `[]`, want: 0},
		{name: "integers", raw: `[1,2,3,4]`, want: 4},
		{
			// --with-pieces answers with objects rather than bare ids, and
			// counting has to survive that without knowing the shape.
			name: "with_pieces objects",
			raw:  `[{"id":1,"piece":"he"},{"id":2,"piece":"llo"}]`,
			want: 2,
		},
		{name: "absent", raw: ``, want: 0},
		{name: "not an array", raw: `{"tokens":5}`, wantErr: true},
		{name: "truncated", raw: `[1,2`, wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := countArray(json.RawMessage(tc.raw))
			if tc.wantErr {
				if err == nil {
					t.Errorf("countArray(%s) = %d, want an error", tc.raw, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("countArray(%s): %v", tc.raw, err)
			}
			if got != tc.want {
				t.Errorf("countArray(%s) = %d, want %d", tc.raw, got, tc.want)
			}
		})
	}
}

func TestTokenizerWithoutURLFails(t *testing.T) {
	tk := &Tokenizer{}
	var shape requestShape
	_ = json.Unmarshal([]byte(`{"prompt":"hi"}`), &shape)
	if _, err := tk.CountPrompt(context.Background(), shape); err == nil {
		t.Error("CountPrompt with no BaseURL succeeded; it has nothing to ask")
	}
}
