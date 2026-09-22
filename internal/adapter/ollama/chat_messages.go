package ollama

import (
	"log/slog"
	"strings"
)

// OpenAI multimodal content block types used when flattening chat
// messages for Ollama /api/chat. Named constants keep goconst quiet
// across chat_messages*.go and translate_chat_test.go.
const (
	openaiBlockText      = "text"
	openaiBlockInputText = "input_text"
	openaiBlockImageURL  = "image_url"
	openaiBlockImage     = "image"
	openaiKeyImageURL    = "image_url"
	openaiKeyURL         = "url"
	openaiKeyText        = "text"
	openaiKeySource      = "source"
	openaiKeyType        = "type"
	openaiKeyData        = "data"
	openaiSourceBase64   = "base64"
	openaiSourceURL      = "url"
)

// normalizeOpenAIChatMessages rewrites OpenAI /v1/chat/completions
// messages into Ollama /api/chat shape. Ollama's ChatRequest.messages[].content
// is a string; clients such as OpenClaw send block arrays instead.
func normalizeOpenAIChatMessages(raw any) []map[string]any {
	msgs, ok := raw.([]any)
	if !ok {
		return nil
	}
	out := make([]map[string]any, 0, len(msgs))
	for _, item := range msgs {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		out = append(out, normalizeOneOpenAIChatMessage(m))
	}
	return out
}

func normalizeOneOpenAIChatMessage(m map[string]any) map[string]any {
	out := map[string]any{}
	if role, ok := m[keyRole].(string); ok {
		out[keyRole] = role
	}
	text, images, droppedURLs := flattenOpenAIChatContent(m[keyContent])
	out[keyContent] = text
	if len(images) > 0 {
		out["images"] = images
	}
	if droppedURLs > 0 {
		slog.Warn("ollama: dropped url-source image blocks in chat message (Ollama needs base64)",
			slog.Int("dropped_count", droppedURLs))
	}
	if v, ok := m[openaiKeyToolCalls]; ok {
		out[openaiKeyToolCalls] = normalizeToolCallsForOllama(v)
	} else if fc, ok := m[openaiKeyFunctionCall].(map[string]any); ok {
		out[openaiKeyToolCalls] = legacyFunctionCallToToolCalls(fc)
	}
	if v, ok := m[openaiKeyToolCallID]; ok {
		out[openaiKeyToolCallID] = v
	}
	if v, ok := m[openaiKeyToolName]; ok {
		out[openaiKeyToolName] = v
	}
	if v, ok := m["name"]; ok {
		out["name"] = v
	}
	return out
}

func flattenOpenAIChatContent(content any) (text string, images []string, droppedURLs int) {
	switch c := content.(type) {
	case string:
		return c, nil, 0
	case nil:
		return "", nil, 0
	case []any:
		var b strings.Builder
		for _, part := range c {
			t, imgs, dropped := flattenOpenAIChatContentPart(part)
			b.WriteString(t)
			images = append(images, imgs...)
			droppedURLs += dropped
		}
		return b.String(), images, droppedURLs
	default:
		return "", nil, 0
	}
}

func flattenOpenAIChatContentPart(part any) (text string, images []string, droppedURLs int) {
	m, ok := part.(map[string]any)
	if !ok {
		return "", nil, 0
	}
	typ, _ := m[openaiKeyType].(string)
	switch typ {
	case openaiBlockText, openaiBlockInputText:
		if s, ok := m[openaiKeyText].(string); ok {
			return s, nil, 0
		}
	case openaiBlockImageURL:
		iu, _ := m[openaiKeyImageURL].(map[string]any)
		url, _ := iu[openaiKeyURL].(string)
		if b64, ok := base64FromDataURL(url); ok {
			return "", []string{b64}, 0
		}
		if url != "" {
			return "", nil, 1
		}
	case openaiBlockImage:
		src, _ := m[openaiKeySource].(map[string]any)
		switch src[openaiKeyType] {
		case openaiSourceBase64:
			if data, ok := src[openaiKeyData].(string); ok {
				return "", []string{data}, 0
			}
		case openaiSourceURL:
			return "", nil, 1
		}
	}
	return "", nil, 0
}

func base64FromDataURL(url string) (string, bool) {
	if !strings.HasPrefix(url, "data:") {
		return "", false
	}
	if idx := strings.Index(url, ","); idx >= 0 {
		return url[idx+1:], true
	}
	return "", false
}
