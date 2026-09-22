package hfwrap

import "testing"

func TestMatchPatterns(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		rel     string
		include []string
		exclude []string
		want    bool
	}{
		{name: "whole repo", rel: "onnx/model.onnx", want: true},
		{name: "exact basename", rel: "weights/q4.gguf", include: []string{"q4.gguf"}, want: true},
		{name: "exact path", rel: "weights/q4.gguf", include: []string{"weights/q4.gguf"}, want: true},
		{name: "glob crosses slash", rel: "onnx/model.gguf", include: []string{"*.gguf"}, want: true},
		{name: "glob miss", rel: "README.md", include: []string{"*.gguf"}, want: false},
		{name: "exclude wins", rel: "model.gguf", include: []string{"*.gguf"}, exclude: []string{"*.gguf"}, want: false},
		{name: "exclude other", rel: "model.gguf", exclude: []string{"*.md"}, want: true},
		{name: "empty include pattern ignored", rel: "a.bin", include: []string{"", "a.bin"}, want: true},
		{name: "nested glob star", rel: "onnx/foo/model.gguf", include: []string{"*.gguf"}, want: true},
		{name: "path glob one segment", rel: "onnx/model.onnx", include: []string{"onnx/*"}, want: true},
		{name: "path glob does not eat extra slash", rel: "onnx/foo/model.onnx", include: []string{"onnx/*"}, want: false},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := MatchPatterns(tc.rel, tc.include, tc.exclude); got != tc.want {
				t.Fatalf("MatchPatterns(%q) = %v, want %v", tc.rel, got, tc.want)
			}
		})
	}
}

func TestUnderSubdir(t *testing.T) {
	t.Parallel()
	if !UnderSubdir("onnx/model.onnx", "onnx") {
		t.Fatal("child of subdir must match")
	}
	if !UnderSubdir("onnx", "onnx") {
		t.Fatal("subdir itself must match")
	}
	if UnderSubdir("tokenizer.json", "onnx") {
		t.Fatal("sibling of subdir must not match")
	}
	if !UnderSubdir("anywhere/file", "") {
		t.Fatal("empty subdir accepts every path")
	}
}
