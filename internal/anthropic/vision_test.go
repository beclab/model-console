package anthropic

import (
	"reflect"
	"strings"
	"testing"
)

func TestOpenAIContentFromBlocks_TextOnly(t *testing.T) {
	t.Parallel()
	blocks := []ContentBlock{
		{Type: BlockText, Text: "hello "},
		{Type: BlockText, Text: "world"},
	}
	got := OpenAIContentFromBlocks(blocks)
	if got != "hello world" {
		t.Errorf("got %#v want %q", got, "hello world")
	}
}

func TestOpenAIContentFromBlocks_WithBase64Image(t *testing.T) {
	t.Parallel()
	blocks := []ContentBlock{
		{Type: BlockText, Text: "look at this:"},
		{Type: BlockImage, Source: &ImageSource{
			Type: "base64", MediaType: "image/png", Data: "iVBOR",
		}},
	}
	got := OpenAIContentFromBlocks(blocks)
	arr, ok := got.([]map[string]any)
	if !ok {
		t.Fatalf("expected array form, got %T: %#v", got, got)
	}
	if len(arr) != 2 {
		t.Fatalf("len=%d", len(arr))
	}
	if arr[0]["type"] != "text" || arr[0]["text"] != "look at this:" {
		t.Errorf("text block=%+v", arr[0])
	}
	if arr[1]["type"] != "image_url" {
		t.Errorf("image type=%v", arr[1]["type"])
	}
	url, _ := arr[1]["image_url"].(map[string]any)
	if want := "data:image/png;base64,iVBOR"; url["url"] != want {
		t.Errorf("url=%v want %q", url["url"], want)
	}
}

func TestOpenAIContentFromBlocks_WithURL(t *testing.T) {
	t.Parallel()
	blocks := []ContentBlock{
		{Type: BlockImage, Source: &ImageSource{Type: "url", URL: "https://example.com/img.jpg"}},
	}
	got := OpenAIContentFromBlocks(blocks)
	arr, ok := got.([]map[string]any)
	if !ok || len(arr) != 1 {
		t.Fatalf("expected single-image array, got %#v", got)
	}
	url, _ := arr[0]["image_url"].(map[string]any)
	if url["url"] != "https://example.com/img.jpg" {
		t.Errorf("url=%v", url["url"])
	}
}

func TestOpenAIContentFromBlocks_DefaultMediaType(t *testing.T) {
	t.Parallel()
	blocks := []ContentBlock{
		{Type: BlockImage, Source: &ImageSource{Type: "base64", Data: "X"}},
	}
	got := OpenAIContentFromBlocks(blocks)
	arr, _ := got.([]map[string]any)
	url, _ := arr[0]["image_url"].(map[string]any)
	if !strings.HasPrefix(url["url"].(string), "data:image/png;base64,") {
		t.Errorf("missing default image/png: %v", url["url"])
	}
}

func TestOpenAIContentFromBlocks_DropMalformed(t *testing.T) {
	t.Parallel()
	blocks := []ContentBlock{
		{Type: BlockImage, Source: nil},                          // no source
		{Type: BlockImage, Source: &ImageSource{Type: "base64"}}, // no data
		{Type: BlockImage, Source: &ImageSource{Type: "url"}},    // no url
		{Type: BlockText, Text: "fallback"},
		{Type: BlockImage, Source: &ImageSource{Type: "base64", Data: "ok"}}, // one good
	}
	got := OpenAIContentFromBlocks(blocks)
	arr, _ := got.([]map[string]any)
	// Only the text + the one good image survive.
	if len(arr) != 2 {
		t.Errorf("got %d entries want 2: %#v", len(arr), arr)
	}
}

func TestOllamaImagesFromBlocks(t *testing.T) {
	t.Parallel()
	blocks := []ContentBlock{
		{Type: BlockText, Text: "describe"},
		{Type: BlockImage, Source: &ImageSource{Type: "base64", Data: "AAA"}},
		{Type: BlockImage, Source: &ImageSource{Type: "base64", Data: "BBB"}},
		{Type: BlockImage, Source: &ImageSource{Type: "url", URL: "https://x"}},
	}
	imgs, dropped := OllamaImagesFromBlocks(blocks)
	if !reflect.DeepEqual(imgs, []string{"AAA", "BBB"}) {
		t.Errorf("imgs=%v", imgs)
	}
	if dropped != 1 {
		t.Errorf("dropped=%d want 1", dropped)
	}
}

func TestOllamaImagesFromBlocks_None(t *testing.T) {
	t.Parallel()
	blocks := []ContentBlock{{Type: BlockText, Text: "hi"}}
	imgs, dropped := OllamaImagesFromBlocks(blocks)
	if imgs != nil {
		t.Errorf("imgs should be nil when no image blocks, got %v", imgs)
	}
	if dropped != 0 {
		t.Errorf("dropped=%d", dropped)
	}
}

func TestValidateBase64(t *testing.T) {
	t.Parallel()
	if err := ValidateBase64(""); err != nil {
		t.Errorf("empty: %v", err)
	}
	if err := ValidateBase64("aGVsbG8="); err != nil {
		t.Errorf("padded: %v", err)
	}
	if err := ValidateBase64("aGVsbG8"); err != nil {
		t.Errorf("unpadded: %v", err)
	}
	if err := ValidateBase64("not base64!"); err == nil {
		t.Errorf("expected error for invalid input")
	}
}

func TestOpenAIContentFromBlocks_Empty(t *testing.T) {
	t.Parallel()
	if got := OpenAIContentFromBlocks(nil); got != nil {
		t.Errorf("nil input should return nil, got %#v", got)
	}
}
