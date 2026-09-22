package anthropic

import (
	"encoding/base64"
	"fmt"
	"strings"
)

// OpenAI multimodal content type identifiers + image source vocab.
// Named so the translator and tests refer to one canonical spelling.
const (
	openaiContentTypeText     = "text"
	openaiContentTypeImageURL = "image_url"
	openaiKeyURL              = "url"

	// Anthropic ImageSource.type values.
	imageSourceBase64 = "base64"
	imageSourceURL    = "url"
	// Default media type when the client omits it (defensive; every
	// real SDK supplies one). RFC 6838 image/png is universally
	// supported.
	defaultImageMediaType = "image/png"
)

// OpenAIContentFromBlocks shapes an Anthropic content block array into
// the OpenAI multimodal content envelope. Behaviour:
//
//   - Text-only blocks: returns a plain string (concatenation of all
//     text blocks). Simpler than the array form and accepted by every
//     OpenAI-compatible engine.
//   - One or more image blocks: returns the typed array form
//     ([]map[string]any) with text segments interleaved as
//     {type:text, text:"..."} and images as {type:image_url,
//     image_url:{url:"data:<media_type>;base64,<data>"}} or
//     {type:image_url, image_url:{url:"https://..."}}.
//
// tool_use / tool_result blocks are skipped here — those are handled
// at the message-building layer (they become tool_calls or
// role:"tool" messages, not content within a user/assistant message).
//
// Returns nil for empty input so the caller can omit the field
// entirely. Returns a string "" (not nil) when ONLY whitespace text
// blocks are present so the wire shape stays consistent.
func OpenAIContentFromBlocks(blocks []ContentBlock) any {
	if len(blocks) == 0 {
		return nil
	}
	hasImage := false
	for _, b := range blocks {
		if b.Type == BlockImage {
			hasImage = true
			break
		}
	}
	if !hasImage {
		var text strings.Builder
		for _, b := range blocks {
			if b.Type == BlockText {
				text.WriteString(b.Text)
			}
		}
		return text.String()
	}

	out := make([]map[string]any, 0, len(blocks))
	for _, b := range blocks {
		switch b.Type {
		case BlockText:
			if b.Text != "" {
				out = append(out, map[string]any{
					openaiKeyType:         openaiContentTypeText,
					openaiContentTypeText: b.Text,
				})
			}
		case BlockImage:
			url, ok := imageBlockToDataURL(b)
			if !ok {
				// Skip malformed image blocks rather than fail the
				// whole request — Anthropic clients sometimes ship
				// empty image source fields during development.
				continue
			}
			out = append(out, map[string]any{
				openaiKeyType:             openaiContentTypeImageURL,
				openaiContentTypeImageURL: map[string]any{openaiKeyURL: url},
			})
		}
	}
	if len(out) == 0 {
		return ""
	}
	return out
}

// imageBlockToDataURL converts an Anthropic image content block into
// the URL form OpenAI's image_url field expects:
//
//   - base64 source: "data:<media_type>;base64,<data>" (RFC 2397).
//     Media type defaults to image/png when missing — every Anthropic
//     SDK in the wild sets media_type, but defensive parsing keeps a
//     misformatted client from breaking the request outright.
//   - URL source: passthrough.
//
// Returns ("", false) for unrecognised source types so the caller can
// drop the block.
func imageBlockToDataURL(b ContentBlock) (string, bool) {
	if b.Source == nil {
		return "", false
	}
	switch b.Source.Type {
	case imageSourceBase64:
		if b.Source.Data == "" {
			return "", false
		}
		media := b.Source.MediaType
		if media == "" {
			media = defaultImageMediaType
		}
		return fmt.Sprintf("data:%s;base64,%s", media, b.Source.Data), true
	case imageSourceURL:
		if b.Source.URL == "" {
			return "", false
		}
		return b.Source.URL, true
	default:
		return "", false
	}
}

// OllamaImagesFromBlocks extracts the base64 image strings from an
// Anthropic content block array in the shape Ollama's /api/chat
// accepts (`images: ["<base64>", "<base64>", ...]`).
//
// Ollama does NOT support URL image sources — it requires base64. URL
// sources are skipped here; the adapter logs a warning via the
// returned `droppedURLCount` so operators see misconfigurations
// without flooding logs on every chat turn.
//
// Returns (imagesArray, droppedURLCount). imagesArray is nil when
// blocks contain no image blocks at all; the caller should omit the
// field entirely in that case.
func OllamaImagesFromBlocks(blocks []ContentBlock) (images []string, droppedURLCount int) {
	for _, b := range blocks {
		if b.Type != BlockImage || b.Source == nil {
			continue
		}
		switch b.Source.Type {
		case imageSourceBase64:
			if b.Source.Data != "" {
				images = append(images, b.Source.Data)
			}
		case imageSourceURL:
			droppedURLCount++
		}
	}
	return images, droppedURLCount
}

// ValidateBase64 returns nil iff s is decodable as standard or
// padded base64. Adapters use this as an early-rejection guard so
// the upstream doesn't get a parse error after llm-init has already
// committed to the request. Empty input is accepted (some clients
// ship empty image strings as placeholders).
func ValidateBase64(s string) error {
	if s == "" {
		return nil
	}
	_, err := base64.StdEncoding.DecodeString(s)
	if err == nil {
		return nil
	}
	if _, err2 := base64.RawStdEncoding.DecodeString(s); err2 == nil {
		return nil
	}
	return err
}
