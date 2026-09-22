package ollama

import (
	"encoding/json"
	"testing"
)

func TestToolCallArgumentsString(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		raw  json.RawMessage
		want string
	}{
		{"empty", nil, "{}"},
		{"object", json.RawMessage(`{"city":"北京"}`), `{"city":"北京"}`},
		{"string", func() json.RawMessage {
			b, _ := json.Marshal(`{"city":"北京"}`)
			return b
		}(), `{"city":"北京"}`},
		{"empty_string", json.RawMessage(`""`), "{}"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := toolCallArgumentsString(tc.raw); got != tc.want {
				t.Errorf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestToolCallArgumentsObject(t *testing.T) {
	t.Parallel()
	obj := toolCallArgumentsObject(map[string]any{"city": "北京"})
	if obj["city"] != "北京" {
		t.Fatalf("object passthrough=%v", obj)
	}
	obj = toolCallArgumentsObject(`{"city":"北京"}`)
	if obj["city"] != "北京" {
		t.Fatalf("string parse=%v", obj)
	}
	obj = toolCallArgumentsObject("")
	if len(obj) != 0 {
		t.Fatalf("empty=%v", obj)
	}
	obj = toolCallArgumentsObject("not-json")
	if len(obj) != 0 {
		t.Fatalf("invalid=%v", obj)
	}
}

func TestNormalizeToolCallsForOllama(t *testing.T) {
	t.Parallel()
	out := normalizeToolCallsForOllama([]map[string]any{{
		"id": "call_1", "type": "function",
		"function": map[string]any{
			"name":      "get_weather",
			"arguments": `{"city":"北京"}`,
		},
	}})
	if len(out) != 1 {
		t.Fatalf("len=%d", len(out))
	}
	fn, _ := out[0][openaiKeyFunction].(map[string]any)
	args, ok := fn[openaiKeyArguments].(map[string]any)
	if !ok {
		t.Fatalf("arguments type=%T", fn[openaiKeyArguments])
	}
	if args["city"] != "北京" {
		t.Errorf("city=%v", args["city"])
	}
}

func TestLegacyFunctionCallToToolCalls(t *testing.T) {
	t.Parallel()
	out := legacyFunctionCallToToolCalls(map[string]any{
		"name":      "get_weather",
		"arguments": `{"city":"北京"}`,
	})
	fn, _ := out[0][openaiKeyFunction].(map[string]any)
	args, _ := fn[openaiKeyArguments].(map[string]any)
	if args["city"] != "北京" {
		t.Errorf("city=%v", args["city"])
	}
}
