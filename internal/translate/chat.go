package translate

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
)

const (
	pathChatCompletions = "/v1/chat/completions"
	// finishReasonLength is OpenAI's spelling for a reply the token budget
	// cut off. The transcript translator branches on it: a truncated reply
	// is retried with double the room rather than taken, because half a
	// sentence reads as a finished translation forever.
	finishReasonLength = "length"
	// Neutral defaults for translation; models may ignore or override via
	// their own sampling. Not tuned for any single model family.
	defaultTemperature = 0.2
	defaultMaxTokens   = 4096
)

// Completer runs a non-streaming chat completion against the already-
// mounted OpenAI data-plane handler (so model rewrite / engine routing
// stay identical to /v1/chat/completions).
type Completer struct {
	Handler   http.Handler
	ModelName string

	// BuildPrompt, when non-nil, builds the user message. Nil uses
	// DefaultPrompt (model-agnostic).
	BuildPrompt PromptBuilder

	// Optional sampling overrides. Nil fields use package defaults
	// (Temperature/MaxTokens) or are omitted (TopP).
	Temperature *float64
	TopP        *float64
	MaxTokens   *int
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
	TopP        *float64      `json:"top_p,omitempty"`
	MaxTokens   int           `json:"max_tokens"`
	Stream      bool          `json:"stream"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage chatUsage `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    any    `json:"code"`
	} `json:"error"`
}

// chatUsage is what the engine reports it spent. Passed back up so a route can
// declare its token spend to whoever is metering — every route here does, since
// nothing downstream of this process can recover a token count from a
// translation.
type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (u chatUsage) add(other chatUsage) chatUsage {
	return chatUsage{
		PromptTokens:     u.PromptTokens + other.PromptTokens,
		CompletionTokens: u.CompletionTokens + other.CompletionTokens,
		TotalTokens:      u.TotalTokens + other.TotalTokens,
	}
}

// declareUsage is how a translate route tells whoever is metering what the
// request cost. The gateway reads these three headers and has nothing else to
// read: it forwards the answer without seeing the completion behind it, so a
// route that stays quiet is recorded as zero tokens.
//
// An engine that reported nothing writes no headers, which keeps "not reported"
// distinguishable from "reported as zero". Engines that report a total but no
// breakdown, or the reverse, are both accepted: the total is derived when it is
// missing rather than the row being dropped over it.
func declareUsage(w http.ResponseWriter, usage chatUsage) {
	if usage == (chatUsage{}) {
		return
	}
	total := usage.TotalTokens
	if total == 0 {
		total = usage.PromptTokens + usage.CompletionTokens
	}
	w.Header().Set("X-Model-Usage-Prompt-Tokens", strconv.Itoa(usage.PromptTokens))
	w.Header().Set("X-Model-Usage-Completion-Tokens", strconv.Itoa(usage.CompletionTokens))
	w.Header().Set("X-Model-Usage-Total-Tokens", strconv.Itoa(total))
}

// chatCall is one completion in the shape the transcript route needs: a
// system message the rules live in, and a budget sized from the turns being
// translated rather than fixed for the process.
type chatCall struct {
	System string
	User   string
	// MaxTokens of zero falls back to the Completer's own setting.
	MaxTokens int
}

// chatResult is an answer with what it cost.
type chatResult struct {
	Text  string
	Usage chatUsage
	// Truncated reports that the engine stopped on the token budget rather
	// than because the model was finished. Worth distinguishing from a
	// mangled reply: the caller can afford the same turns a second time
	// with more room, and cannot fix a model that will not follow the
	// numbering.
	Truncated bool
}

// Complete posts a single-turn user prompt and returns the assistant text with
// what the engine says it cost. The cost is not optional to the caller: it is
// the only place a token count for this request exists, and a route that drops
// it is metered as free by everything upstream.
func (c *Completer) Complete(ctx context.Context, prompt string) (string, chatUsage, error) {
	res, err := c.complete(ctx, chatCall{User: prompt})
	if err != nil {
		return "", chatUsage{}, err
	}
	return res.Text, res.Usage, nil
}

func (c *Completer) complete(ctx context.Context, call chatCall) (chatResult, error) {
	if c == nil || c.Handler == nil {
		return chatResult{}, fmt.Errorf("chat completer not configured")
	}
	temp := defaultTemperature
	if c.Temperature != nil {
		temp = *c.Temperature
	}
	maxTok := defaultMaxTokens
	if c.MaxTokens != nil {
		maxTok = *c.MaxTokens
	}
	if call.MaxTokens > 0 {
		maxTok = call.MaxTokens
	}

	messages := make([]chatMessage, 0, 2)
	if call.System != "" {
		messages = append(messages, chatMessage{Role: "system", Content: call.System})
	}
	messages = append(messages, chatMessage{Role: "user", Content: call.User})

	body, err := json.Marshal(chatRequest{
		Model:       c.ModelName,
		Messages:    messages,
		Temperature: temp,
		TopP:        c.TopP,
		MaxTokens:   maxTok,
		Stream:      false,
	})
	if err != nil {
		return chatResult{}, err
	}

	req := httptest.NewRequestWithContext(ctx, http.MethodPost, pathChatCompletions, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	c.Handler.ServeHTTP(rec, req)

	respBody, err := io.ReadAll(rec.Body)
	if err != nil {
		return chatResult{}, err
	}
	if rec.Code < 200 || rec.Code >= 300 {
		return chatResult{}, fmt.Errorf("chat completions status %d: %s", rec.Code, truncate(string(respBody), 512))
	}

	var parsed chatResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return chatResult{}, fmt.Errorf("decode chat response: %w", err)
	}
	if parsed.Error != nil && parsed.Error.Message != "" {
		return chatResult{}, fmt.Errorf("engine error: %s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return chatResult{}, fmt.Errorf("empty chat completions choices")
	}
	return chatResult{
		Text:      strings.TrimSpace(parsed.Choices[0].Message.Content),
		Usage:     parsed.Usage,
		Truncated: parsed.Choices[0].FinishReason == finishReasonLength,
	}, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
