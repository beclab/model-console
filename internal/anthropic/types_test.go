package anthropic

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestContent_UnmarshalString(t *testing.T) {
	t.Parallel()
	var c Content
	if err := json.Unmarshal([]byte(`"hello"`), &c); err != nil {
		t.Fatalf("unmarshal string: %v", err)
	}
	if c.Plain != "hello" {
		t.Errorf("Plain=%q want hello", c.Plain)
	}
	blocks := c.Blocks()
	if len(blocks) != 1 || blocks[0].Type != BlockText || blocks[0].Text != "hello" {
		t.Errorf("Blocks()=%+v want one text block", blocks)
	}
	if got := c.Text(); got != "hello" {
		t.Errorf("Text()=%q want hello", got)
	}
}

func TestContent_UnmarshalBlockArray(t *testing.T) {
	t.Parallel()
	var c Content
	in := `[{"type":"text","text":"a"},{"type":"text","text":"b"}]`
	if err := json.Unmarshal([]byte(in), &c); err != nil {
		t.Fatalf("unmarshal blocks: %v", err)
	}
	if c.Plain != "" {
		t.Errorf("Plain=%q want empty when blocks were sent", c.Plain)
	}
	if got := c.Text(); got != "ab" {
		t.Errorf("Text()=%q want ab", got)
	}
}

func TestContent_UnmarshalNull(t *testing.T) {
	t.Parallel()
	var c Content
	if err := json.Unmarshal([]byte("null"), &c); err != nil {
		t.Fatalf("unmarshal null: %v", err)
	}
	if c.Plain != "" || len(c.Blocks()) != 0 {
		t.Errorf("expected zero content for null, got Plain=%q blocks=%v", c.Plain, c.Blocks())
	}
}

func TestContent_MarshalRoundTrip(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"plain string", `"hi"`, `"hi"`},
		{"block array", `[{"type":"text","text":"hi"}]`, `[{"type":"text","text":"hi"}]`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var c Content
			if err := json.Unmarshal([]byte(tt.in), &c); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			out, err := json.Marshal(c)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(out) != tt.want {
				t.Errorf("round-trip: got %s want %s", out, tt.want)
			}
		})
	}
}

func TestSystemPrompt_UnmarshalString(t *testing.T) {
	t.Parallel()
	var s SystemPrompt
	if err := json.Unmarshal([]byte(`"you are helpful"`), &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if s.Text != "you are helpful" {
		t.Errorf("Text=%q", s.Text)
	}
}

func TestSystemPrompt_UnmarshalBlocks(t *testing.T) {
	t.Parallel()
	var s SystemPrompt
	in := `[{"type":"text","text":"line1"},{"type":"text","text":" line2"}]`
	if err := json.Unmarshal([]byte(in), &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if s.Text != "line1 line2" {
		t.Errorf("Text=%q want concat", s.Text)
	}
}

func TestMapFinishReason(t *testing.T) {
	t.Parallel()
	cases := map[string]string{
		"stop":          StopReasonEndTurn,
		"":              StopReasonEndTurn,
		"length":        StopReasonMaxTokens,
		"tool_calls":    StopReasonToolUse,
		"function_call": StopReasonToolUse,
		"weird":         StopReasonEndTurn,
	}
	for in, want := range cases {
		if got := MapFinishReason(in); got != want {
			t.Errorf("MapFinishReason(%q) = %q want %q", in, got, want)
		}
	}
}

func TestNewMessageID(t *testing.T) {
	t.Parallel()
	id := NewMessageID(1)
	if !strings.HasPrefix(id, "msg_01") {
		t.Errorf("ID prefix wrong: %q", id)
	}
	if len(id) != len("msg_01")+16 {
		t.Errorf("ID length wrong: %q (%d)", id, len(id))
	}
	// Distinct seeds produce distinct ids.
	if NewMessageID(2) == id {
		t.Errorf("expected different ids for different seeds")
	}
}

func TestMessagesRequest_Decode_MinimalText(t *testing.T) {
	t.Parallel()
	body := `{
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 64,
        "messages": [{"role":"user","content":"hi"}]
    }`
	var req MessagesRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if req.MaxTokens != 64 {
		t.Errorf("MaxTokens=%d", req.MaxTokens)
	}
	if len(req.Messages) != 1 || req.Messages[0].Content.Text() != "hi" {
		t.Errorf("messages decode wrong: %+v", req.Messages)
	}
}

func TestMessagesRequest_Decode_BlockContent(t *testing.T) {
	t.Parallel()
	body := `{
        "model":"m","max_tokens":1,
        "system":"sys",
        "messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]
    }`
	var req MessagesRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if req.System.Text != "sys" {
		t.Errorf("system=%q", req.System.Text)
	}
	if req.Messages[0].Content.Text() != "ab" {
		t.Errorf("flatten failed: %q", req.Messages[0].Content.Text())
	}
}

func TestContent_SetBlocks(t *testing.T) {
	t.Parallel()
	var c Content
	c.SetBlocks([]ContentBlock{{Type: BlockText, Text: "x"}})
	out, _ := json.Marshal(c)
	if string(out) != `[{"type":"text","text":"x"}]` {
		t.Errorf("SetBlocks marshal: %s", out)
	}
}
