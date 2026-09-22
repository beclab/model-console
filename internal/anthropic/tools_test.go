package anthropic

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestToolsToOpenAI(t *testing.T) {
	t.Parallel()
	tools := []Tool{
		{
			Name:        "get_weather",
			Description: "Get the weather",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
		},
		{Name: "noop"}, // no description, no schema
	}
	out := ToolsToOpenAI(tools)
	if len(out) != 2 {
		t.Fatalf("got %d tools, want 2", len(out))
	}
	if out[0]["type"] != "function" {
		t.Errorf("type=%v", out[0]["type"])
	}
	fn, _ := out[0]["function"].(map[string]any)
	if fn["name"] != "get_weather" || fn["description"] != "Get the weather" {
		t.Errorf("function=%+v", fn)
	}
	params, _ := fn["parameters"].(map[string]any)
	if params["type"] != "object" {
		t.Errorf("parameters not preserved: %+v", params)
	}
	fn2, _ := out[1]["function"].(map[string]any)
	if _, hasDesc := fn2["description"]; hasDesc {
		t.Errorf("noop tool should omit description")
	}
}

func TestToolsToOpenAI_Empty(t *testing.T) {
	t.Parallel()
	if ToolsToOpenAI(nil) != nil {
		t.Errorf("nil input should return nil")
	}
	if ToolsToOpenAI([]Tool{}) != nil {
		t.Errorf("empty slice should return nil")
	}
}

func TestToolChoiceToOpenAI(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		in   string
		want any
	}{
		{"auto", `{"type":"auto"}`, "auto"},
		{"none", `{"type":"none"}`, "none"},
		{"any", `{"type":"any"}`, "required"},
		{
			"specific tool",
			`{"type":"tool","name":"get_weather"}`,
			map[string]any{
				"type":     "function",
				"function": map[string]any{"name": "get_weather"},
			},
		},
		{"bare string", `"auto"`, "auto"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ToolChoiceToOpenAI(json.RawMessage(tc.in))
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got %#v want %#v", got, tc.want)
			}
		})
	}
}

func TestToolCallsToBlocks(t *testing.T) {
	t.Parallel()
	calls := []OpenAIToolCall{
		{
			ID:   "call_1",
			Type: "function",
			Function: struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			}{Name: "get_weather", Arguments: `{"location":"SF"}`},
		},
		{
			ID:   "call_2",
			Type: "function",
			Function: struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			}{Name: "bad", Arguments: "not-json"},
		},
	}
	blocks := ToolCallsToBlocks(calls)
	if len(blocks) != 2 {
		t.Fatalf("got %d blocks", len(blocks))
	}
	if blocks[0].Type != BlockToolUse || blocks[0].ID != "call_1" || blocks[0].Name != "get_weather" {
		t.Errorf("blocks[0]=%+v", blocks[0])
	}
	if string(blocks[0].Input) != `{"location":"SF"}` {
		t.Errorf("blocks[0].Input=%s", blocks[0].Input)
	}
	if string(blocks[1].Input) != "{}" {
		t.Errorf("malformed args should fall back to {}; got %s", blocks[1].Input)
	}
}

func TestToolResultsFromMessage(t *testing.T) {
	t.Parallel()
	m := Message{
		Role: RoleUser,
		Content: Content{blocks: []ContentBlock{
			{Type: BlockToolResult, ToolUseID: "call_1", Content: &Content{Plain: "72F"}},
			{Type: BlockToolResult, ToolUseID: "call_2", Content: &Content{Plain: "boom"}, IsError: true},
			{Type: BlockText, Text: "extra context"},
		}},
	}
	toolMsgs, residual := ToolResultsFromMessage(m)
	if len(toolMsgs) != 2 {
		t.Fatalf("got %d tool msgs", len(toolMsgs))
	}
	if toolMsgs[0]["tool_call_id"] != "call_1" || toolMsgs[0]["content"] != "72F" {
		t.Errorf("toolMsgs[0]=%+v", toolMsgs[0])
	}
	if toolMsgs[1]["content"] != "[error] boom" {
		t.Errorf("is_error prefix missing: %+v", toolMsgs[1])
	}
	if len(residual) != 1 || residual[0].Type != BlockText {
		t.Errorf("residual=%+v", residual)
	}
}

func TestAssistantToolUseToOpenAI(t *testing.T) {
	t.Parallel()
	m := Message{
		Role: RoleAssistant,
		Content: Content{blocks: []ContentBlock{
			{Type: BlockText, Text: "I'll check the weather."},
			{Type: BlockToolUse, ID: "call_1", Name: "get_weather", Input: json.RawMessage(`{"location":"SF"}`)},
		}},
	}
	content, calls := AssistantToolUseToOpenAI(m)
	if content != "I'll check the weather." {
		t.Errorf("content=%q", content)
	}
	if len(calls) != 1 {
		t.Fatalf("got %d calls", len(calls))
	}
	if calls[0]["id"] != "call_1" {
		t.Errorf("id=%v", calls[0]["id"])
	}
	fn, _ := calls[0]["function"].(map[string]any)
	if fn["name"] != "get_weather" || fn["arguments"] != `{"location":"SF"}` {
		t.Errorf("function=%+v", fn)
	}
}
