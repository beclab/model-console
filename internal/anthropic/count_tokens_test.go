package anthropic

import (
	"encoding/json"
	"testing"
)

func TestEstimateTokens_TextOnly(t *testing.T) {
	t.Parallel()
	req := CountTokensRequest{
		Messages: []Message{
			{Role: RoleUser, Content: Content{Plain: "hello world"}}, // 11 chars
		},
	}
	got := EstimateTokens(req)
	// 11 chars -> ceil(11/4) = 3
	if got != 3 {
		t.Errorf("got %d want 3", got)
	}
}

func TestEstimateTokens_WithSystem(t *testing.T) {
	t.Parallel()
	req := CountTokensRequest{
		System: SystemPrompt{Text: "you are helpful"}, // 15
		Messages: []Message{
			{Role: RoleUser, Content: Content{Plain: "hi"}}, // 2
		},
	}
	got := EstimateTokens(req)
	// 17 chars -> ceil(17/4) = 5
	if got != 5 {
		t.Errorf("got %d want 5", got)
	}
}

func TestEstimateTokens_Image(t *testing.T) {
	t.Parallel()
	req := CountTokensRequest{
		Messages: []Message{{
			Role: RoleUser,
			Content: Content{blocks: []ContentBlock{
				{Type: BlockText, Text: "look:"}, // 5
				{Type: BlockImage, Source: &ImageSource{Type: "base64", Data: "..."}},
			}},
		}},
	}
	got := EstimateTokens(req)
	// 5 chars -> ceil(5/4) = 2; + 1568 image budget
	if got != 2+1568 {
		t.Errorf("got %d want %d", got, 2+1568)
	}
}

func TestEstimateTokens_Tools(t *testing.T) {
	t.Parallel()
	req := CountTokensRequest{
		Messages: []Message{
			{Role: RoleUser, Content: Content{Plain: "hi"}}, // 2
		},
		Tools: []Tool{{
			Name:        "get_weather",         // 11
			Description: "Gets the weather",    // 16
			InputSchema: json.RawMessage(`{}`), // 2
		}},
	}
	got := EstimateTokens(req)
	// 2+11+16+2 = 31 -> ceil(31/4) = 8
	if got != 8 {
		t.Errorf("got %d want 8", got)
	}
}

func TestEstimateTokens_Empty(t *testing.T) {
	t.Parallel()
	if got := EstimateTokens(CountTokensRequest{}); got != 0 {
		t.Errorf("empty req should be 0, got %d", got)
	}
}
