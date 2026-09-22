package anthropic

import (
	"encoding/json"
)

// OpenAI-shape constants. These appear across the Anthropic-to-OpenAI
// translation paths and also in adapter/proxy and adapter/ollama via
// ToolsToOpenAI / ToolChoiceToOpenAI. Named constants keep goconst
// quiet on the cross-file repetition.
const (
	openaiTypeFunction  = "function"
	openaiToolChoiceReq = "required"
	openaiToolChoiceAny = "any"
	openaiKeyName       = "name"
	openaiKeyType       = "type"
	openaiKeyFunction   = "function"

	// Anthropic tool_choice.type values. Used by ToolChoiceToOpenAI
	// when translating to OpenAI shapes.
	toolChoiceAuto = "auto"
	toolChoiceNone = "none"
)

// ToolsToOpenAI converts Anthropic Tool[] to OpenAI tools[]. The two
// schemas differ in three places:
//
//	Anthropic                            OpenAI
//	---------                            ------
//	name                                 function.name
//	description                          function.description
//	input_schema                         function.parameters
//	(implicit type: function)            type: "function"
//
// Both backends (Ollama 0.1.45+ via /api/chat, vLLM/SGLang/llama.cpp
// via /v1/chat/completions) consume the OpenAI shape, so the same
// converter feeds both adapters.
//
// Empty / nil input returns nil so the caller can decide whether to
// omit the field entirely.
func ToolsToOpenAI(tools []Tool) []map[string]any {
	if len(tools) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		fn := map[string]any{openaiKeyName: t.Name}
		if t.Description != "" {
			fn["description"] = t.Description
		}
		if len(t.InputSchema) > 0 {
			// input_schema is opaque JSON; preserve byte-for-byte
			// to avoid losing custom JSON Schema extensions.
			var schema any
			if err := json.Unmarshal(t.InputSchema, &schema); err == nil {
				fn["parameters"] = schema
			}
		}
		out = append(out, map[string]any{
			openaiKeyType:     openaiTypeFunction,
			openaiKeyFunction: fn,
		})
	}
	return out
}

// ToolChoiceToOpenAI translates Anthropic tool_choice (a JSON object
// or a bare string in legacy clients) into OpenAI's tool_choice
// (mostly identical except for the "tool" -> "function" wrapper).
//
// Anthropic shapes:
//
//	{"type":"auto"}            → "auto"
//	{"type":"any"}             → "required"   (OpenAI's any-tool name)
//	{"type":"tool","name":X}   → {"type":"function","function":{"name":X}}
//	"<string>"                 → passthrough (a few legacy clients)
//
// Returns nil for empty input so the caller can skip emitting the
// field. Returns the raw bytes (as a generic any) when the input is
// unrecognised so we don't silently swallow novel modes.
func ToolChoiceToOpenAI(raw json.RawMessage) any {
	if len(raw) == 0 {
		return nil
	}
	trimmed := trimSpaceBytes(raw)
	if len(trimmed) == 0 {
		return nil
	}
	if trimmed[0] == '"' {
		var s string
		if json.Unmarshal(trimmed, &s) == nil {
			return s
		}
		return string(raw)
	}
	var obj map[string]any
	if err := json.Unmarshal(trimmed, &obj); err != nil {
		return string(raw)
	}
	t, _ := obj[openaiKeyType].(string)
	switch t {
	case toolChoiceAuto, toolChoiceNone:
		return t
	case openaiToolChoiceAny:
		return openaiToolChoiceReq
	case "tool":
		name, _ := obj[openaiKeyName].(string)
		if name == "" {
			return openaiToolChoiceReq
		}
		return map[string]any{
			openaiKeyType:     openaiTypeFunction,
			openaiKeyFunction: map[string]any{openaiKeyName: name},
		}
	default:
		// Unknown type — forward the original object so the engine
		// can decide (some engines accept custom modes).
		return obj
	}
}

// OpenAIToolCall is the wire shape of one entry in OpenAI's
// choices[].message.tool_calls. Defined here so the adapters share a
// single decoder.
type OpenAIToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// ToolCallsToBlocks converts a list of OpenAI tool_calls into
// Anthropic tool_use content blocks. The OpenAI "arguments" string is
// a serialised JSON object; we re-parse it into raw JSON so the
// Anthropic content_block.input field carries the same object shape
// the model emitted (some clients pin on input being a JSON object
// rather than a string).
func ToolCallsToBlocks(calls []OpenAIToolCall) []ContentBlock {
	if len(calls) == 0 {
		return nil
	}
	out := make([]ContentBlock, 0, len(calls))
	for _, c := range calls {
		input := json.RawMessage(c.Function.Arguments)
		// Validate that arguments parses; otherwise emit an empty
		// object so Anthropic SDKs don't choke on invalid JSON
		// (anthropic-sdk-python type-checks input as a dict).
		var probe any
		if err := json.Unmarshal(input, &probe); err != nil || len(input) == 0 {
			input = json.RawMessage("{}")
		}
		out = append(out, ContentBlock{
			Type:  BlockToolUse,
			ID:    c.ID,
			Name:  c.Function.Name,
			Input: input,
		})
	}
	return out
}

// ToolResultsFromMessage extracts tool_result content blocks from an
// Anthropic Message and returns the corresponding OpenAI-shape
// messages (one per tool_result block; OpenAI requires one message
// per tool call response).
//
// Each tool_result block becomes:
//
//	{"role":"tool", "tool_call_id":"<id>", "content":"<flattened text>"}
//
// is_error=true blocks have the prefix "[error] " prepended to the
// content so OpenAI engines (which don't have an is_error
// equivalent) surface the error condition to the model.
//
// Returns (toolMessages, remainingBlocks): toolMessages is the OpenAI
// messages to emit immediately; remainingBlocks is the residual
// content (text / image blocks) that should still go through as a
// user-role message. Adapters fold both into their messages array.
func ToolResultsFromMessage(m Message) (toolMsgs []map[string]any, residual []ContentBlock) {
	for _, blk := range m.Content.Blocks() {
		if blk.Type != BlockToolResult {
			residual = append(residual, blk)
			continue
		}
		text := ""
		if blk.Content != nil {
			text = blk.Content.Text()
		}
		if blk.IsError && text != "" {
			text = "[error] " + text
		}
		toolMsgs = append(toolMsgs, map[string]any{
			"role":         RoleTool,
			"tool_call_id": blk.ToolUseID,
			"content":      text,
		})
	}
	return toolMsgs, residual
}

// AssistantToolUseToOpenAI extracts tool_use blocks from an Anthropic
// assistant Message and returns an OpenAI-shape `tool_calls` array
// ready to attach to the assistant's message. Text blocks become the
// message's content field; tool_use blocks become tool_calls entries.
//
// This is the inverse of ToolCallsToBlocks: it lets adapters replay a
// multi-turn conversation where the assistant previously called a
// tool and the next user turn carries a tool_result.
func AssistantToolUseToOpenAI(m Message) (content string, toolCalls []map[string]any) {
	for _, blk := range m.Content.Blocks() {
		switch blk.Type {
		case BlockText:
			content += blk.Text
		case BlockToolUse:
			args := string(blk.Input)
			if args == "" {
				args = "{}"
			}
			toolCalls = append(toolCalls, map[string]any{
				"id":          blk.ID,
				openaiKeyType: openaiTypeFunction,
				openaiKeyFunction: map[string]any{
					openaiKeyName: blk.Name,
					"arguments":   args,
				},
			})
		}
	}
	return content, toolCalls
}

func trimSpaceBytes(b []byte) []byte {
	start, end := 0, len(b)
	for start < end {
		c := b[start]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			start++
			continue
		}
		break
	}
	for end > start {
		c := b[end-1]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			end--
			continue
		}
		break
	}
	return b[start:end]
}
