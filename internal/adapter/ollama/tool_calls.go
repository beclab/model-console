package ollama

import "encoding/json"

const (
	openaiKeyFunction     = "function"
	openaiKeyArguments    = "arguments"
	openaiKeyToolCalls    = "tool_calls"
	openaiKeyFunctionCall = "function_call"
	openaiKeyToolCallID   = "tool_call_id"
	openaiKeyToolName     = "tool_name"
	openaiTypeFunction    = "function"
)

// normalizeToolCallsForOllama rewrites OpenAI tool_calls on inbound
// /v1/chat/completions messages into the shape Ollama /api/chat expects:
// function.arguments is a JSON object, not a serialized string.
func normalizeToolCallsForOllama(raw any) []map[string]any {
	if raw == nil {
		return nil
	}
	b, err := json.Marshal(raw)
	if err != nil {
		return nil
	}
	var arr []any
	if err := json.Unmarshal(b, &arr); err != nil {
		return nil
	}
	out := make([]map[string]any, 0, len(arr))
	for _, item := range arr {
		tc, ok := item.(map[string]any)
		if !ok {
			continue
		}
		out = append(out, normalizeOneToolCallForOllama(tc))
	}
	return out
}

func normalizeOneToolCallForOllama(tc map[string]any) map[string]any {
	out := map[string]any{}
	for _, k := range []string{"id", openaiKeyType, "index"} {
		if v, ok := tc[k]; ok {
			out[k] = v
		}
	}
	fn, _ := tc[openaiKeyFunction].(map[string]any)
	if fn == nil {
		return out
	}
	normFn := map[string]any{}
	if name, ok := fn["name"]; ok {
		normFn["name"] = name
	}
	normFn[openaiKeyArguments] = toolCallArgumentsObject(fn[openaiKeyArguments])
	out[openaiKeyFunction] = normFn
	return out
}

func legacyFunctionCallToToolCalls(fc map[string]any) []map[string]any {
	fn := map[string]any{}
	if name, ok := fc["name"]; ok {
		fn["name"] = name
	}
	fn[openaiKeyArguments] = toolCallArgumentsObject(fc[openaiKeyArguments])
	return []map[string]any{{
		openaiKeyType:     openaiTypeFunction,
		openaiKeyFunction: fn,
	}}
}

// toolCallArgumentsObject converts OpenAI function.arguments (string or
// object) into the JSON object Ollama expects on /api/chat requests.
func toolCallArgumentsObject(raw any) map[string]any {
	empty := map[string]any{}
	if raw == nil {
		return empty
	}
	switch v := raw.(type) {
	case map[string]any:
		return v
	case string:
		return parseToolCallArgumentsJSON(v)
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return empty
		}
		if len(b) > 0 && b[0] == '"' {
			var s string
			if err := json.Unmarshal(b, &s); err == nil {
				return parseToolCallArgumentsJSON(s)
			}
		}
		var obj map[string]any
		if err := json.Unmarshal(b, &obj); err == nil && obj != nil {
			return obj
		}
		return empty
	}
}

func parseToolCallArgumentsJSON(s string) map[string]any {
	empty := map[string]any{}
	if s == "" {
		return empty
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(s), &obj); err != nil || obj == nil {
		return empty
	}
	return obj
}

// toolCallArgumentsString normalizes Ollama function.arguments into the
// OpenAI wire form: a string containing serialized JSON.
func toolCallArgumentsString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "{}"
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err == nil {
			if s == "" {
				return "{}"
			}
			return s
		}
	}
	return string(raw)
}

func openAIWireToolCalls(from []ollamaToolCall) []openAIWireToolCall {
	if len(from) == 0 {
		return nil
	}
	out := make([]openAIWireToolCall, 0, len(from))
	for _, tc := range from {
		w := openAIWireToolCall{
			ID:   tc.ID,
			Type: tc.Type,
		}
		w.Function.Name = tc.Function.Name
		w.Function.Arguments = toolCallArgumentsString(tc.Function.Arguments)
		out = append(out, w)
	}
	return out
}
