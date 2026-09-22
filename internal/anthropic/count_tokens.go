package anthropic

import "encoding/json"

// CountTokensRequest is the inbound /v1/messages/count_tokens body.
// Anthropic's spec accepts a subset of MessagesRequest fields; we
// mirror that shape so callers can use one struct for both.
//
// Per https://docs.anthropic.com/en/api/messages-count-tokens the
// fields are: model, messages (required), system (optional), tools
// (optional), tool_choice (optional). max_tokens is NOT used (no
// generation).
type CountTokensRequest struct {
	Model      string          `json:"model"`
	Messages   []Message       `json:"messages"`
	System     SystemPrompt    `json:"system,omitempty"`
	Tools      []Tool          `json:"tools,omitempty"`
	ToolChoice json.RawMessage `json:"tool_choice,omitempty"`
}

// CountTokensResponse is the outbound body. Anthropic's response
// contains a single integer key.
type CountTokensResponse struct {
	InputTokens int64 `json:"input_tokens"`
}

// EstimateTokens returns a heuristic token count for an Anthropic
// count_tokens request. The estimator is intentionally rough — about
// the same accuracy as OpenAI's tiktoken on English text — and is
// shipped behind the `X-Llm-Init-Token-Estimate: heuristic` response
// header so clients can opt out or compensate.
//
// Heuristic: chars / 4, rounded up. The 4-chars-per-token rule of
// thumb is documented by Anthropic itself
// (https://docs.anthropic.com/en/docs/build-with-claude/tokens) and
// matches OpenAI's tiktoken to within ~10% on prose. Tool schemas
// and system prompts are summed alongside message content. Image
// blocks contribute a flat 1568 tokens per image (Anthropic's own
// published image-token budget for medium-resolution images).
//
// Returns 0 for an empty request (no messages, no system, no tools).
// Adapters call this when upstream tokenize integration is
// unavailable (Stage 5 ships with no upstream tokenize support).
func EstimateTokens(req CountTokensRequest) int64 {
	var chars int64
	if t := req.System.Text; t != "" {
		chars += int64(len(t))
	}
	const imageTokenBudget = 1568
	var imageTokens int64
	for _, m := range req.Messages {
		for _, blk := range m.Content.Blocks() {
			switch blk.Type {
			case BlockText:
				chars += int64(len(blk.Text))
			case BlockToolUse:
				if blk.Name != "" {
					chars += int64(len(blk.Name))
				}
				chars += int64(len(blk.Input))
			case BlockToolResult:
				if blk.Content != nil {
					chars += int64(len(blk.Content.Text()))
				}
			case BlockImage:
				imageTokens += imageTokenBudget
			}
		}
	}
	for _, tool := range req.Tools {
		chars += int64(len(tool.Name))
		chars += int64(len(tool.Description))
		chars += int64(len(tool.InputSchema))
	}
	chars += int64(len(req.ToolChoice))
	if chars == 0 && imageTokens == 0 {
		return 0
	}
	// chars / 4 rounded up.
	return (chars+3)/4 + imageTokens
}
