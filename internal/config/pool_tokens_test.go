package config

import "testing"

func TestDerivePoolTokens(t *testing.T) {
	tests := []struct {
		name  string
		kind  EngineKind
		args  string
		want  int
		wantK bool
	}{
		{
			name: "llamacpp -c is the pool, not the window",
			kind: EngineLlamaCpp, args: "-c 12288 -np 2 -no-kvu",
			want: 12288, wantK: true,
		},
		{
			name: "llamacpp pool is padded like the engine pads it",
			kind: EngineLlamaCpp, args: "-c 12000",
			want: 12032, wantK: true,
		},
		{
			// -c 0 and no -c both mean "take the training context from
			// the GGUF", which is not readable from here.
			name: "llamacpp -c 0 defers to the model file",
			kind: EngineLlamaCpp, args: "-c 0",
			want: 0, wantK: false,
		},
		{
			name: "llamacpp no -c defers to the model file",
			kind: EngineLlamaCpp, args: "-np 2",
			want: 0, wantK: false,
		},
		{
			// server.cpp sizes the pool as n_parallel * per_slot when -c
			// is absent, which is the one case an absent -c is knowable.
			name: "llamacpp per-slot ceiling sizes the pool without -c",
			kind: EngineLlamaCpp, args: "-np 3 --kv-unified-per-slot 4096",
			want: 12288, wantK: true,
		},
		{
			name: "sglang --max-total-tokens",
			kind: EngineSGLang, args: "--max-total-tokens 65536",
			want: 65536, wantK: true,
		},
		{
			// The usual case: the pool comes from the memory fraction and
			// the hardware, so only the running engine knows it.
			name: "sglang mem fraction is not a token count",
			kind: EngineSGLang, args: "--mem-fraction-static 0.9 --context-length 32768",
			want: 0, wantK: false,
		},
		{
			name: "ollama pool is the product the runner is launched with",
			kind: EngineOllama, args: "",
			want: 0, wantK: false,
		},
		{
			// vLLM decides its block count by profiling the model, so no
			// flag states the pool. Absent from the table entirely.
			name: "vllm has no flag for the pool",
			kind: EngineVLLM, args: "--max-model-len 32768 --max-num-seqs 64",
			want: 0, wantK: false,
		},
		{
			name: "engines without a KV cache have no pool",
			kind: EngineEmbed, args: "",
			want: 0, wantK: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			args, err := ParseEngineArgs(tc.kind, tc.args)
			if err != nil {
				t.Fatalf("ParseEngineArgs(%q): %v", tc.args, err)
			}
			got, ok := DerivePoolTokens(tc.kind, args)
			if got != tc.want || ok != tc.wantK {
				t.Errorf("DerivePoolTokens = (%d, %v), want (%d, %v)",
					got, ok, tc.want, tc.wantK)
			}
		})
	}
}

// Ollama's two numbers arrive as env vars rather than flags, so they are
// exercised through the Known table the env parser writes.
func TestDerivePoolTokens_Ollama(t *testing.T) {
	tests := []struct {
		name  string
		known map[string]string
		want  int
		wantK bool
	}{
		{
			name:  "both pinned",
			known: map[string]string{"num_ctx": "8192", "num_parallel": "4"},
			want:  32768, wantK: true,
		},
		{
			// Ollama's own default for num_parallel is four or one
			// depending on free memory, so half a product is no product.
			name:  "width unpinned leaves the pool unknown",
			known: map[string]string{"num_ctx": "8192"},
			want:  0, wantK: false,
		},
		{
			name:  "window unpinned leaves the pool unknown",
			known: map[string]string{"num_parallel": "4"},
			want:  0, wantK: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := DerivePoolTokens(EngineOllama, EngineArgs{Known: tc.known})
			if got != tc.want || ok != tc.wantK {
				t.Errorf("DerivePoolTokens = (%d, %v), want (%d, %v)",
					got, ok, tc.want, tc.wantK)
			}
		})
	}
}
