package config

import (
	"reflect"
	"strings"
	"testing"
)

func TestParseOneSource_HF(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		raw     string
		want    ModelSource
		wantErr string
	}{
		{
			name: "bare repo",
			raw:  "hf://Qwen/Qwen2.5-7B-Instruct",
			want: ModelSource{
				Index: 1, Kind: KindHF, Role: RoleMain,
				HFRepo:         "Qwen/Qwen2.5-7B-Instruct",
				RawValue:       "hf://Qwen/Qwen2.5-7B-Instruct",
				RedactedSource: "hf://Qwen/Qwen2.5-7B-Instruct",
			},
		},
		{
			name: "single include + revision",
			raw:  "hf://owner/repo --include q4.gguf --revision abcdef",
			want: ModelSource{
				Index: 1, Kind: KindHF, Role: RoleMain,
				HFRepo:         "owner/repo",
				HFInclude:      []string{"q4.gguf"},
				HFRevision:     "abcdef",
				RawValue:       "hf://owner/repo --include q4.gguf --revision abcdef",
				RedactedSource: "hf://owner/repo --include q4.gguf --revision abcdef",
			},
		},
		{
			name: "multiple include + exclude with = form",
			raw:  "hf://o/r --include=a.gguf --include=b.gguf --exclude=*.bin",
			want: ModelSource{
				Index: 1, Kind: KindHF, Role: RoleMain,
				HFRepo:         "o/r",
				HFInclude:      []string{"a.gguf", "b.gguf"},
				HFExclude:      []string{"*.bin"},
				RawValue:       "hf://o/r --include=a.gguf --include=b.gguf --exclude=*.bin",
				RedactedSource: "hf://o/r --include=a.gguf --include=b.gguf --exclude=*.bin",
			},
		},
		{
			name: "subdir for unified repo",
			raw:  "hf://beclab/embeddinggemma-300m --include onnx/* --include onnx/**/* --subdir onnx",
			want: ModelSource{
				Index: 1, Kind: KindHF, Role: RoleMain,
				HFRepo:         "beclab/embeddinggemma-300m",
				HFInclude:      []string{"onnx/*", "onnx/**/*"},
				HFSubdir:       "onnx",
				RawValue:       "hf://beclab/embeddinggemma-300m --include onnx/* --include onnx/**/* --subdir onnx",
				RedactedSource: "hf://beclab/embeddinggemma-300m --include onnx/* --include onnx/**/* --subdir onnx",
			},
		},
		{
			name: "subdir equals form",
			raw:  "hf://o/r --subdir=openvino",
			want: ModelSource{
				Index: 1, Kind: KindHF, Role: RoleMain,
				HFRepo:         "o/r",
				HFSubdir:       "openvino",
				RawValue:       "hf://o/r --subdir=openvino",
				RedactedSource: "hf://o/r --subdir=openvino",
			},
		},
		{
			name:    "subdir path traversal rejected",
			raw:     "hf://o/r --subdir ../etc",
			wantErr: "not allowed",
		},
		{
			name:    "subdir absolute rejected",
			raw:     "hf://o/r --subdir /onnx",
			wantErr: "relative to the snapshot root",
		},
		{
			name:    "subdir duplicate rejected",
			raw:     "hf://o/r --subdir onnx --subdir openvino",
			wantErr: "more than once",
		},
		{
			name:    "subdir without value",
			raw:     "hf://o/r --subdir",
			wantErr: "--subdir needs a value",
		},
		{
			name:    "endpoint flag rejected",
			raw:     "hf://o/r --endpoint https://hf-mirror.com",
			wantErr: "--endpoint is not allowed",
		},
		{
			name:    "endpoint flag = form rejected",
			raw:     "hf://o/r --endpoint=https://x",
			wantErr: "--endpoint is not allowed",
		},
		{
			name:    "include without value",
			raw:     "hf://o/r --include",
			wantErr: "--include needs a value",
		},
		{
			name:    "non-owner-repo body",
			raw:     "hf://justone",
			wantErr: "owner/repo",
		},
		{
			name:    "unsupported flag",
			raw:     "hf://o/r --weird",
			wantErr: "unsupported hf://",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := ParseOneSource(1, tc.raw, "", "")
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want substring %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestParseOneSource_OllamaHeuristic(t *testing.T) {
	t.Parallel()
	cases := []struct {
		raw      string
		wantKind SourceKind
		wantTag  string
		wantURL  string
	}{
		// Library tag: no dot+slash, no scheme.
		{"ollama://qwen3:0.6b", KindOllama, "qwen3:0.6b", ""},
		{"ollama://llama3", KindOllama, "llama3", ""},
		// "library/llama3" has '/' but no '.', so heuristic falls
		// through to Library tag form; see model-source.html#ollama-overload.
		{"ollama://library/llama3", KindOllama, "library/llama3", ""},
		// Explicit URL via scheme.
		{"ollama://https://huggingface.co/owner/repo/resolve/main/q4.gguf", KindOllamaURL, "",
			"https://huggingface.co/owner/repo/resolve/main/q4.gguf"},
		{"ollama://http://example.com/x.gguf", KindOllamaURL, "",
			"http://example.com/x.gguf"},
		// Heuristic: dot+slash without scheme -> URL form (auto-prepended https://).
		{"ollama://hf-mirror.com/Qwen/repo/q4.gguf", KindOllamaURL, "",
			"https://hf-mirror.com/Qwen/repo/q4.gguf"},
	}
	for _, tc := range cases {
		t.Run(tc.raw, func(t *testing.T) {
			t.Parallel()
			ms, err := ParseOneSource(1, tc.raw, "", "")
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if ms.Kind != tc.wantKind {
				t.Errorf("kind = %q, want %q", ms.Kind, tc.wantKind)
			}
			if ms.OllamaTag != tc.wantTag {
				t.Errorf("tag = %q, want %q", ms.OllamaTag, tc.wantTag)
			}
			if ms.URL != tc.wantURL {
				t.Errorf("url = %q, want %q", ms.URL, tc.wantURL)
			}
		})
	}
}

func TestParseOneSource_OllamaRejectsFlags(t *testing.T) {
	t.Parallel()
	if _, err := ParseOneSource(1, "ollama://qwen3:0.6b --include foo", "", ""); err == nil {
		t.Fatal("expected error: ollama:// does not accept inline flags")
	}
}

func TestParseOneSource_URL(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name     string
		raw      string
		local    string
		wantSha  string
		wantPath string
		wantErr  string
	}{
		{
			name:     "https with sha256 fragment + LOCAL",
			raw:      "https://example.com/q4.gguf#sha256=" + strings.Repeat("a", 64),
			local:    "/cache/manual/q4.gguf",
			wantSha:  strings.Repeat("a", 64),
			wantPath: "/cache/manual/q4.gguf",
		},
		{
			name:    "https without LOCAL",
			raw:     "https://example.com/q4.gguf",
			local:   "",
			wantErr: "MODEL_SOURCE_LOCAL: required",
		},
		{
			name:    "fragment not sha256",
			raw:     "https://example.com/q4.gguf#md5=deadbeef",
			local:   "/cache/x.gguf",
			wantErr: "only #sha256= URL fragment is supported",
		},
		{
			name:    "sha256 wrong length",
			raw:     "https://example.com/q4.gguf#sha256=tooshort",
			local:   "/cache/x.gguf",
			wantErr: "sha256 must be 64",
		},
		{
			name:    "scheme not http(s)",
			raw:     "ftp://example.com/x.gguf",
			local:   "/cache/x.gguf",
			wantErr: "unsupported scheme",
		},
		{
			name:    "URL inline flag rejected",
			raw:     "https://example.com/x.gguf --revision foo",
			local:   "/cache/x.gguf",
			wantErr: "https?:// does not accept inline flags",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := ParseOneSource(1, tc.raw, tc.local, "")
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want substring %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if got.Kind != KindURL {
				t.Errorf("kind = %q, want %q", got.Kind, KindURL)
			}
			if got.URLSha256 != tc.wantSha {
				t.Errorf("sha = %q, want %q", got.URLSha256, tc.wantSha)
			}
			if got.LocalPath != tc.wantPath {
				t.Errorf("local = %q, want %q", got.LocalPath, tc.wantPath)
			}
		})
	}
}

func TestParseOneSource_Role(t *testing.T) {
	t.Parallel()
	ms, err := ParseOneSource(2, "hf://o/r", "", "")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if ms.Role != RoleMain {
		t.Errorf("Role = %q, want %q", ms.Role, RoleMain)
	}
	ms2, err := ParseOneSource(3, "hf://o/r", "", RoleExtra)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if ms2.Role != RoleExtra {
		t.Errorf("Role = %q, want %q", ms2.Role, RoleExtra)
	}

	if _, err := ParseOneSource(4, "hf://o/r", "", "custom"); err == nil ||
		!strings.Contains(err.Error(), "role") {
		t.Fatalf("custom role err = %v, want role validation error", err)
	}
}

// --role mmproj is the comma-form replacement for the retired
// MODEL_SOURCE_<i>_ROLE=mmproj, so it must survive parsing on an hf://
// segment and stay out of the flags handed to `hf download`.
func TestParseOneSource_InlineRoleMmproj(t *testing.T) {
	t.Parallel()
	for _, raw := range []string{
		"hf://o/r --include mmproj.gguf --role mmproj",
		"hf://o/r --include mmproj.gguf --role=mmproj",
	} {
		ms, err := ParseOneSource(2, raw, "", RoleExtra)
		if err != nil {
			t.Fatalf("%s: unexpected err: %v", raw, err)
		}
		if ms.Role != RoleMmproj {
			t.Errorf("%s: Role = %q, want %q", raw, ms.Role, RoleMmproj)
		}
		if len(ms.HFInclude) != 1 || ms.HFInclude[0] != "mmproj.gguf" {
			t.Errorf("%s: HFInclude = %v, want [mmproj.gguf]", raw, ms.HFInclude)
		}
	}
}

func TestParseOneSource_InlineRoleRejections(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		raw     string
		role    string
		wantErr string
	}{
		{
			name:    "value other than mmproj",
			raw:     "hf://o/r --role extra",
			role:    RoleExtra,
			wantErr: "--role",
		},
		{
			name:    "missing value",
			raw:     "hf://o/r --role",
			role:    RoleExtra,
			wantErr: "--role needs a value",
		},
		{
			name:    "twice",
			raw:     "hf://o/r --role mmproj --role mmproj",
			role:    RoleExtra,
			wantErr: "more than once",
		},
		{
			name:    "on ollama segment",
			raw:     "ollama://qwen3:0.6b --role mmproj",
			role:    RoleExtra,
			wantErr: "does not accept inline flags",
		},
		{
			name:    "on url segment",
			raw:     "https://example.com/mmproj.gguf --role mmproj",
			role:    RoleExtra,
			wantErr: "does not accept inline flags",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := ParseOneSource(2, tc.raw, "/cache/mmproj.gguf", tc.role)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("err = %v, want substring %q", err, tc.wantErr)
			}
		})
	}
}

func TestLoadModelSources_InlineRoleMmproj(t *testing.T) {
	t.Parallel()
	got, err := LoadModelSources(mapEnv(map[string]string{
		"MODEL_SOURCE": "hf://o/vlm --include model.gguf," +
			"hf://o/vlm --include mmproj.gguf --role mmproj," +
			"hf://o/layout --subdir onnx",
	}))
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	wantRoles := []string{RoleMain, RoleMmproj, RoleExtra}
	if len(got) != len(wantRoles) {
		t.Fatalf("len = %d, want %d", len(got), len(wantRoles))
	}
	for i, want := range wantRoles {
		if got[i].Role != want {
			t.Errorf("entry %d: Role = %q, want %q", i, got[i].Role, want)
		}
	}
}

func TestLoadModelSources_InlineRoleMmprojListRules(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		source  string
		wantErr string
	}{
		{
			name:    "on the first segment",
			source:  "hf://o/vlm --include mmproj.gguf --role mmproj,hf://o/vlm --include model.gguf",
			wantErr: "first source is the main source",
		},
		{
			name: "more than one",
			source: "hf://o/vlm --include model.gguf," +
				"hf://o/vlm --include mmproj.gguf --role mmproj," +
				"hf://o/vlm --include mmproj2.gguf --role mmproj",
			wantErr: "at most one source",
		},
		{
			name:    "without an include",
			source:  "hf://o/vlm --include model.gguf,hf://o/vlm --role mmproj",
			wantErr: "exactly one --include",
		},
		{
			name: "with two includes",
			source: "hf://o/vlm --include model.gguf," +
				"hf://o/vlm --include mmproj.gguf --include mmproj2.gguf --role mmproj",
			wantErr: "exactly one --include",
		},
		{
			name:    "main source is not hf",
			source:  "ollama://qwen3:0.6b,hf://o/vlm --include mmproj.gguf --role mmproj",
			wantErr: "needs an hf:// main source",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := LoadModelSources(mapEnv(map[string]string{"MODEL_SOURCE": tc.source}))
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("err = %v, want substring %q", err, tc.wantErr)
			}
		})
	}
}

func TestParseOneSource_BadShlex(t *testing.T) {
	t.Parallel()
	_, err := ParseOneSource(1, `hf://o/r --include "unbalanced`, "", "")
	if err == nil || !strings.Contains(err.Error(), "unbalanced quote") {
		t.Fatalf("err = %v, want unbalanced quote", err)
	}
}

func TestParseOneSource_ViaOllama(t *testing.T) {
	t.Parallel()
	ms, _ := ParseOneSource(1, "ollama://qwen3:0.6b", "", "")
	if !ms.ViaOllama() {
		t.Error("ollama Library tag should be ViaOllama=true")
	}
	ms, _ = ParseOneSource(1, "ollama://https://example.com/x.gguf", "", "")
	if !ms.ViaOllama() {
		t.Error("ollama-url overload should be ViaOllama=true")
	}
	ms, _ = ParseOneSource(1, "hf://o/r", "", "")
	if ms.ViaOllama() {
		t.Error("hf source should be ViaOllama=false")
	}
}

func TestRedactRaw_URLUserinfo(t *testing.T) {
	t.Parallel()
	cases := []struct {
		raw  string
		want string
	}{
		{"https://user:pass@host/path", "https://REDACTED@host/path"},
		{"https://example.com/x.gguf", "https://example.com/x.gguf"},
		{"hf://owner/repo --revision abc", "hf://owner/repo --revision abc"},
		{"ollama://https://user:pass@host/x.gguf", "ollama://https://REDACTED@host/x.gguf"},
	}
	for _, tc := range cases {
		t.Run(tc.raw, func(t *testing.T) {
			t.Parallel()
			got := redactRaw(tc.raw)
			if got != tc.want {
				t.Errorf("redactRaw(%q) = %q, want %q", tc.raw, got, tc.want)
			}
		})
	}
}

func TestSplitModelSourceList(t *testing.T) {
	t.Parallel()
	cases := []struct {
		raw  string
		want []string
	}{
		{
			raw:  "hf://Qwen/A",
			want: []string{"hf://Qwen/A"},
		},
		{
			raw: "ollama://qwen3:0.6b,ollama://llama3:8b",
			want: []string{
				"ollama://qwen3:0.6b",
				"ollama://llama3:8b",
			},
		},
		{
			raw: "hf://o/a --include a.gguf,hf://o/b --include b.gguf",
			want: []string{
				"hf://o/a --include a.gguf",
				"hf://o/b --include b.gguf",
			},
		},
		{
			raw: "hf://o/a --include a.gguf,not-a-source, https://example.com/b.gguf",
			want: []string{
				"hf://o/a --include a.gguf,not-a-source",
				"https://example.com/b.gguf",
			},
		},
		{
			raw: "hf://o/a,\t\n https://example.com/b.gguf",
			want: []string{
				"hf://o/a",
				"https://example.com/b.gguf",
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.raw, func(t *testing.T) {
			t.Parallel()
			got := splitModelSourceList(tc.raw)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestLoadModelSources_CommaSeparated(t *testing.T) {
	t.Parallel()
	sha := strings.Repeat("a", 64)
	g := mapEnv(map[string]string{
		"MODEL_SOURCE": "ollama://qwen3:0.6b,ollama://llama3:8b",
	})
	got, err := LoadModelSources(g)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2", len(got))
	}
	if got[0].Role != RoleMain || got[0].OllamaTag != "qwen3:0.6b" || got[0].Index != 1 {
		t.Errorf("entry 0: %+v", got[0])
	}
	if got[1].Role != RoleExtra || got[1].OllamaTag != "llama3:8b" || got[1].Index != 2 {
		t.Errorf("entry 1: %+v", got[1])
	}

	g2 := mapEnv(map[string]string{
		"MODEL_SOURCE":       "https://example.com/a.gguf#sha256=" + sha + ",https://example.com/b.gguf#sha256=" + sha,
		"MODEL_SOURCE_LOCAL": "/cache/a.gguf,/cache/b.gguf",
	})
	got2, err := LoadModelSources(g2)
	if err != nil {
		t.Fatalf("dual url err: %v", err)
	}
	if len(got2) != 2 || got2[1].LocalPath != "/cache/b.gguf" {
		t.Fatalf("dual url got %+v", got2)
	}

	g3 := mapEnv(map[string]string{
		"MODEL_SOURCE":       "hf://owner/main, https://example.com/aux.gguf#sha256=" + sha,
		"MODEL_SOURCE_LOCAL": ",/cache/aux.gguf",
	})
	got3, err := LoadModelSources(g3)
	if err != nil {
		t.Fatalf("mixed source err: %v", err)
	}
	if len(got3) != 2 || got3[0].LocalPath != "" || got3[1].LocalPath != "/cache/aux.gguf" {
		t.Fatalf("mixed source locals got %+v", got3)
	}
}

func TestLoadModelSources_CommaLOCALMismatch(t *testing.T) {
	t.Parallel()
	_, err := LoadModelSources(mapEnv(map[string]string{
		"MODEL_SOURCE":       "ollama://a,ollama://b",
		"MODEL_SOURCE_LOCAL": "/only/one",
	}))
	if err == nil || !strings.Contains(err.Error(), "MODEL_SOURCE_LOCAL") {
		t.Fatalf("err = %v, want LOCAL count mismatch", err)
	}
}

func TestLoadModelSources_Single(t *testing.T) {
	t.Parallel()
	g := mapEnv(map[string]string{
		"MODEL_SOURCE": "hf://Qwen/Qwen2.5-7B",
	})
	got, err := LoadModelSources(g)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(got) != 1 || got[0].Kind != KindHF || got[0].HFRepo != "Qwen/Qwen2.5-7B" {
		t.Fatalf("got %+v", got)
	}
	if got[0].Index != 1 {
		t.Errorf("index = %d, want 1", got[0].Index)
	}
}

func TestLoadModelSources_RejectsRemovedModelSourceNum(t *testing.T) {
	t.Parallel()
	_, err := LoadModelSources(mapEnv(map[string]string{
		"MODEL_SOURCE":           "hf://InternVL/InternVL3-8B",
		removedModelSourceNumEnv: "2",
		"MODEL_SOURCE_1":         "hf://InternVL/InternVL3-8B",
		"MODEL_SOURCE_1_ROLE":    "main",
		"MODEL_SOURCE_2":         "https://example.com/mmproj.gguf#sha256=" + strings.Repeat("b", 64),
		"MODEL_SOURCE_2_LOCAL":   "/cache/internvl/mmproj.gguf",
		"MODEL_SOURCE_2_ROLE":    "mmproj",
	}))
	if err == nil {
		t.Fatal("expected MODEL_SOURCE_NUM removal error")
	}
	for _, want := range []string{removedModelSourceNumEnv, "removed", "comma-separated MODEL_SOURCE"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("err = %q, want substring %q", err, want)
		}
	}
}

func TestLoadModelSources_FailureModes(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		env     map[string]string
		wantErr string
	}{
		{
			name:    "neither env set",
			env:     map[string]string{},
			wantErr: "MODEL_SOURCE: required",
		},
		{
			name: "url without local",
			env: map[string]string{
				"MODEL_SOURCE": "https://example.com/x.gguf",
			},
			wantErr: "MODEL_SOURCE_LOCAL: required",
		},
		{
			name: "unsupported scheme",
			env: map[string]string{
				"MODEL_SOURCE": "ftp://x/y",
			},
			wantErr: "unsupported scheme",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := LoadModelSources(mapEnv(tc.env))
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("err = %v, want substring %q", err, tc.wantErr)
			}
		})
	}
}
