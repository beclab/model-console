package ollama

import "testing"

func TestFlattenOpenAIChatContent(t *testing.T) {
	t.Parallel()
	text, imgs, dropped := flattenOpenAIChatContent([]any{
		map[string]any{openaiKeyType: openaiBlockText, openaiKeyText: "hello "},
		map[string]any{openaiKeyType: openaiBlockInputText, openaiKeyText: "world"},
		map[string]any{openaiKeyType: openaiBlockImageURL, openaiKeyImageURL: map[string]any{
			openaiKeyURL: "data:image/jpeg;base64,abc123",
		}},
	})
	if text != "hello world" {
		t.Errorf("text=%q", text)
	}
	if len(imgs) != 1 || imgs[0] != "abc123" {
		t.Errorf("images=%v", imgs)
	}
	if dropped != 0 {
		t.Errorf("dropped=%d", dropped)
	}

	_, _, dropped = flattenOpenAIChatContent([]any{
		map[string]any{openaiKeyType: openaiBlockImageURL, openaiKeyImageURL: map[string]any{
			openaiKeyURL: "https://example.com/a.jpg",
		}},
	})
	if dropped != 1 {
		t.Errorf("dropped=%d want 1", dropped)
	}
}

func TestFlattenOpenAIChatContent_StringPassthrough(t *testing.T) {
	t.Parallel()
	text, imgs, dropped := flattenOpenAIChatContent("plain")
	if text != "plain" || len(imgs) != 0 || dropped != 0 {
		t.Errorf("text=%q imgs=%v dropped=%d", text, imgs, dropped)
	}
}
