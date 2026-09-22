package ollama

import (
	"encoding/json"
	"testing"
)

func TestRewriteJSONModelField(t *testing.T) {
	in := []byte(`{"model":"client-alias","messages":[],"stream":true}`)
	out, ok := RewriteJSONModelField(in, "qwen2.5:7b")
	if !ok {
		t.Fatal("expected rewrite ok")
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if m["model"] != "qwen2.5:7b" {
		t.Errorf("model = %v, want qwen2.5:7b", m["model"])
	}
}

func TestRewriteJSONModelField_NonObjectPassthrough(t *testing.T) {
	in := []byte(`not-json`)
	out, ok := RewriteJSONModelField(in, "x")
	if ok || string(out) != string(in) {
		t.Errorf("got ok=%v out=%q, want passthrough", ok, out)
	}
}
