package kvbudget

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Tokenizer counts a prompt with the engine's own tokenizer, which is the
// only thing that can count it correctly: the number depends on the model's
// vocabulary and on the chat template baked into its GGUF.
//
// Chat costs two round trips, and cannot be made to cost one. POST /tokenize
// accepts a `content` string and nothing else -- it has no idea what a
// message list is -- so the template has to be applied first, by POST
// /apply-template, which is the only place the model's own template lives.
// Completions carry a prompt string already and take the second hop alone.
//
// Both are loopback calls to a sibling container, so the cost is two local
// round trips against a prefill that is about to process the same tokens.
type Tokenizer struct {
	// BaseURL is the engine's root, e.g. http://localhost:8080.
	BaseURL string

	// HTTPClient defaults to a client with tokenizeTimeout.
	HTTPClient *http.Client
}

// tokenizeTimeout bounds one hop. Short because this is the engine tokenizing
// text on the same host, and because the fallback -- charging the byte bound
// -- is available immediately and costs only precision.
const tokenizeTimeout = 2 * time.Second

// tokenizeBodyLimit caps a reply. /tokenize returns one integer per token, so
// a 100k-token prompt answers with something near a megabyte.
const tokenizeBodyLimit = 16 << 20

// CountPrompt returns how many tokens the request's prompt occupies.
//
// A body with neither messages nor a prompt returns 0 with no error: there is
// nothing to count and nothing wrong. The caller reads that as "no
// measurement" and charges its bound.
func (t *Tokenizer) CountPrompt(ctx context.Context, shape requestShape) (int, error) {
	content, err := t.contentOf(ctx, shape)
	if err != nil {
		return 0, err
	}
	if content == "" {
		return 0, nil
	}
	return t.tokenize(ctx, content)
}

// contentOf reduces a request to the single string /tokenize can count,
// applying the model's chat template when the request is a message list.
func (t *Tokenizer) contentOf(ctx context.Context, shape requestShape) (string, error) {
	if len(shape.Messages) > 0 {
		return t.applyTemplate(ctx, shape.Messages)
	}
	if len(shape.Prompt) == 0 {
		return "", nil
	}
	// /v1/completions takes a string, an array of strings, or token ids.
	// Only the first is worth a tokenizer call: an array is a batch whose
	// per-item accounting this gate does not model, and ids are already
	// counted by the bound.
	var s string
	if err := json.Unmarshal(shape.Prompt, &s); err != nil {
		return "", nil
	}
	return s, nil
}

// applyTemplate turns a message list into the prompt string the model will
// actually see, including whatever the template adds around it -- which is
// not a rounding error on a short conversation.
func (t *Tokenizer) applyTemplate(ctx context.Context, messages json.RawMessage) (string, error) {
	body, err := json.Marshal(map[string]json.RawMessage{"messages": messages})
	if err != nil {
		return "", err
	}
	var out struct {
		Prompt string `json:"prompt"`
	}
	if err := t.post(ctx, "/apply-template", body, &out); err != nil {
		return "", err
	}
	return out.Prompt, nil
}

// tokenize asks for the token ids and counts them.
func (t *Tokenizer) tokenize(ctx context.Context, content string) (int, error) {
	body, err := json.Marshal(map[string]string{"content": content})
	if err != nil {
		return 0, err
	}
	var out struct {
		Tokens json.RawMessage `json:"tokens"`
	}
	if err := t.post(ctx, "/tokenize", body, &out); err != nil {
		return 0, err
	}
	n, err := countArray(out.Tokens)
	if err != nil {
		return 0, fmt.Errorf("POST /tokenize: %w", err)
	}
	return n, nil
}

// countArray counts a JSON array's elements without materialising them.
//
// Streamed rather than decoded into a slice because the only thing wanted
// from a megabyte of token ids is how many there are, and because it counts
// the `with_pieces` object form as readily as the bare integers.
func countArray(raw json.RawMessage) (int, error) {
	if len(raw) == 0 {
		return 0, nil
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil {
		return 0, err
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '[' {
		return 0, fmt.Errorf("tokens is not an array")
	}
	n := 0
	for dec.More() {
		// Skipping a whole value at a time is what makes an object element
		// cost the same as an integer one.
		var discard json.RawMessage
		if err := dec.Decode(&discard); err != nil {
			return 0, err
		}
		n++
	}
	// The closing bracket has to be there. Without this check a truncated
	// body counts as however many elements arrived, and a short count is
	// the one error direction this gate cannot absorb: it under-reserves,
	// which is the oversubscription the package exists to prevent.
	if _, err := dec.Token(); err != nil {
		return 0, fmt.Errorf("tokens array is truncated: %w", err)
	}
	return n, nil
}

func (t *Tokenizer) post(ctx context.Context, path string, body []byte, into any) error {
	if t.BaseURL == "" {
		return fmt.Errorf("POST %s: no engine URL", path)
	}
	ctx, cancel := context.WithTimeout(ctx, tokenizeTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		strings.TrimRight(t.BaseURL, "/")+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	hc := t.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: tokenizeTimeout}
	}
	resp, err := hc.Do(req)
	if err != nil {
		return fmt.Errorf("POST %s: %w", path, err)
	}
	defer func() { _ = resp.Body.Close() }()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, tokenizeBodyLimit))
	if err != nil {
		return fmt.Errorf("read POST %s: %w", path, err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("POST %s status %d", path, resp.StatusCode)
	}
	if err := json.Unmarshal(raw, into); err != nil {
		return fmt.Errorf("decode POST %s: %w", path, err)
	}
	return nil
}
