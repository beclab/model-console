package verify

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/fetch"
)

// realVerifier is the production Verifier: the same implementation the
// downloader uses, so these tests exercise the real size and digest
// comparison rather than a stub that agrees with them.
func realVerifier() Verifier { return fetch.New(nil) }

func writeFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
}

// "hello" hashed with sha256, used throughout so the fixtures stay
// readable.
const (
	helloBody = "hello"
	helloSHA  = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
)

// A record written and read back must survive verbatim, since every
// later decision keys off its fields.
func TestWriteThenReadRoundTrips(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "sub", "model.bin.llm-init.json")
	want := Sidecar{
		Source: Source{
			Kind:     KindHF,
			Repo:     "Qwen/Qwen3-27B",
			Revision: "main",
			Commit:   "0123456789abcdef0123456789abcdef01234567",
			Endpoint: "https://huggingface.co",
			Include:  []string{"*.gguf"},
			Exclude:  []string{"*.pth"},
			Subdir:   "onnx",
		},
		Files: []File{{Path: "a/b.gguf", Size: 5, SHA256: helloSHA, ETag: `"abc"`}},
	}
	if err := Write(path, want); err != nil {
		t.Fatalf("Write: %v", err)
	}
	got, err := Read(path)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if got.Version != SchemaVersion {
		t.Errorf("Version = %d, want %d", got.Version, SchemaVersion)
	}
	if !got.Source.Equal(want.Source) || got.Source.Commit != want.Source.Commit {
		t.Errorf("Source = %+v, want %+v", got.Source, want.Source)
	}
	if len(got.Files) != 1 || got.Files[0] != want.Files[0] {
		t.Errorf("Files = %+v, want %+v", got.Files, want.Files)
	}
	if got.CompletedAt.IsZero() {
		t.Error("CompletedAt was not stamped")
	}
}

// Write creates the parent directory: the HF sidecar lives in a
// .llm-init/ dir that nothing else makes.
func TestWriteCreatesParentDir(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), ".llm-init", "deadbeef.json")
	if err := Write(path, Sidecar{Files: []File{{Path: "x", Size: 1}}}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("Stat: %v", err)
	}
}

// A missing sidecar is the first-boot case and callers branch on it, so
// it has to be distinguishable from a read error.
func TestReadMissingIsNotExist(t *testing.T) {
	t.Parallel()
	_, err := Read(filepath.Join(t.TempDir(), "nope.json"))
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("err = %v, want os.ErrNotExist", err)
	}
}

// Garbage on disk must not read as a valid record. Treating it as a miss
// costs a re-download; trusting it serves unchecked bytes.
func TestReadMalformedFails(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "s.json")
	writeFile(t, path, "{not json")
	if _, err := Read(path); err == nil {
		t.Fatal("Read: want error for malformed JSON")
	}
}

// A record from a future (or older) schema is not migrated in place.
func TestReadForeignVersionFails(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "s.json")
	writeFile(t, path, `{"version":99,"files":[{"path":"x","size":1}]}`)
	_, err := Read(path)
	if !errors.Is(err, ErrVersionMismatch) {
		t.Fatalf("err = %v, want ErrVersionMismatch", err)
	}
}

func TestRemoveIsIdempotent(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "s.json")
	writeFile(t, path, "{}")
	if err := Remove(path); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if err := Remove(path); err != nil {
		t.Fatalf("Remove on missing file: %v", err)
	}
}

// The size level must not read file contents, and must catch the
// failure it exists for: a download that stopped early.
func TestCheckLocalSizeLevel(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "model.bin"), helloBody)
	s := Sidecar{Files: []File{{Path: "model.bin", Size: 5, SHA256: helloSHA}}}

	rep, err := s.CheckLocal(context.Background(), realVerifier(), root, LevelSize)
	if err != nil {
		t.Fatalf("CheckLocal: %v", err)
	}
	if rep.Digest != 0 || rep.SizeOnly != 1 {
		t.Errorf("report = %+v, want everything size-only at LevelSize", rep)
	}

	writeFile(t, filepath.Join(root, "model.bin"), "hel")
	if _, err := s.CheckLocal(context.Background(), realVerifier(), root, LevelSize); err == nil {
		t.Fatal("CheckLocal: want error for a truncated file")
	}
}

// A file whose length is right but whose bytes are wrong is exactly what
// the digest level exists to catch, and exactly what the size level
// cannot see.
func TestCheckLocalDigestCatchesSameSizeCorruption(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "model.bin"), "HELLO")
	s := Sidecar{Files: []File{{Path: "model.bin", Size: 5, SHA256: helloSHA}}}

	if _, err := s.CheckLocal(context.Background(), realVerifier(), root, LevelSize); err != nil {
		t.Fatalf("LevelSize should not read contents: %v", err)
	}
	_, err := s.CheckLocal(context.Background(), realVerifier(), root, LevelSHA256)
	if err == nil {
		t.Fatal("LevelSHA256: want a digest mismatch")
	}
	if !strings.Contains(err.Error(), "model.bin") {
		t.Errorf("error %q should name the file", err)
	}
}

// An HF snapshot mixes LFS blobs, whose digest the cache gives us, with
// small files stored under a git object id that have none. Asking for
// the digest level must verify what it can and report honestly how much
// that was, rather than failing the whole snapshot or claiming every
// file was hashed.
func TestCheckLocalReportsPartialDigestCoverage(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "weights.gguf"), helloBody)
	writeFile(t, filepath.Join(root, "config.json"), "{}")
	s := Sidecar{Files: []File{
		{Path: "weights.gguf", Size: 5, SHA256: helloSHA},
		{Path: "config.json", Size: 2},
	}}

	rep, err := s.CheckLocal(context.Background(), realVerifier(), root, LevelSHA256)
	if err != nil {
		t.Fatalf("CheckLocal: %v", err)
	}
	if rep.Files != 2 || rep.Digest != 1 || rep.SizeOnly != 1 {
		t.Errorf("report = %+v, want 2 files with 1 digest-checked and 1 size-only", rep)
	}
}

// A record listing nothing is not a record of a successful download.
func TestCheckLocalEmptyIsAnError(t *testing.T) {
	t.Parallel()
	_, err := Sidecar{}.CheckLocal(context.Background(), realVerifier(), t.TempDir(), LevelSize)
	if !errors.Is(err, ErrEmpty) {
		t.Fatalf("err = %v, want ErrEmpty", err)
	}
}

// A file the sidecar lists but disk does not have is a miss, not a pass.
func TestCheckLocalMissingFile(t *testing.T) {
	t.Parallel()
	s := Sidecar{Files: []File{{Path: "gone.bin", Size: 1}}}
	if _, err := s.CheckLocal(context.Background(), realVerifier(), t.TempDir(), LevelSize); err == nil {
		t.Fatal("CheckLocal: want error for a missing file")
	}
}

// Every field that describes what was asked for participates in
// identity, so changing any of them stops the cached bytes being reused.
func TestSourceEqualDiscriminates(t *testing.T) {
	t.Parallel()
	base := Source{
		Kind: KindHF, Repo: "o/r", Revision: "main",
		Endpoint: "https://huggingface.co", Subdir: "onnx",
		Include: []string{"a"}, Exclude: []string{"b"},
	}
	same := Source{
		Kind: KindHF, Repo: "o/r", Revision: "main",
		Endpoint: "https://huggingface.co", Subdir: "onnx",
		Include: []string{"a"}, Exclude: []string{"b"},
	}
	if !base.Equal(same) {
		t.Fatal("two identically-configured sources must be equal")
	}
	for name, mut := range map[string]func(Source) Source{
		"kind":     func(s Source) Source { s.Kind = KindURL; return s },
		"repo":     func(s Source) Source { s.Repo = "o/other"; return s },
		"revision": func(s Source) Source { s.Revision = "dev"; return s },
		"endpoint": func(s Source) Source { s.Endpoint = "https://hf-mirror.com"; return s },
		"subdir":   func(s Source) Source { s.Subdir = "openvino"; return s },
		"include":  func(s Source) Source { s.Include = []string{"a", "c"}; return s },
		"exclude":  func(s Source) Source { s.Exclude = nil; return s },
	} {
		if base.Equal(mut(base)) {
			t.Errorf("%s: sources differing in %s must not be equal", name, name)
		}
	}
}

// Fingerprint files a record under the request it describes, so it has
// to discriminate on exactly what Equal does. The two are separate
// pieces of code over the same field list, and the way they drift is
// that somebody adds a field to one of them: a Fingerprint that misses
// a field files two different requests under one name, and each
// overwrites the other's record on every pass.
//
// Every field is walked by reflection rather than listed, so adding one
// to the struct and to neither function fails here too. Commit and
// URLRedacted are the documented exceptions -- see Source.Equal.
func TestFingerprintAgreesWithEqual(t *testing.T) {
	t.Parallel()
	base := Source{
		Kind: KindHF, Repo: "o/r", Revision: "main", Commit: "aaa",
		Endpoint: "https://huggingface.co", Subdir: "onnx",
		Include: []string{"a"}, Exclude: []string{"b"},
		URLSHA256:      URLIdentity("https://example.com/m.gguf"),
		URLRedacted:    "https://example.com/m.gguf",
		DeclaredSHA256: strings.Repeat("f", 64),
	}
	notIdentity := map[string]bool{"Commit": true, "URLRedacted": true}

	rt := reflect.TypeOf(base)
	for i := range rt.NumField() {
		name := rt.Field(i).Name
		mutated := base
		switch f := reflect.ValueOf(&mutated).Elem().Field(i); f.Kind() {
		case reflect.String:
			f.SetString(f.String() + "-changed")
		case reflect.Slice:
			f.Set(reflect.ValueOf([]string{"changed"}))
		default:
			t.Fatalf("%s: unhandled field kind %s; extend this test", name, f.Kind())
		}

		equal := base.Equal(mutated)
		sameFingerprint := base.Fingerprint() == mutated.Fingerprint()
		if notIdentity[name] {
			if !equal || !sameFingerprint {
				t.Errorf("%s is documented as outside the source identity, but changing it made Equal=%v sameFingerprint=%v",
					name, equal, sameFingerprint)
			}
			continue
		}
		if equal {
			t.Errorf("Equal ignores %s", name)
		}
		if sameFingerprint {
			t.Errorf("Fingerprint ignores %s", name)
		}
	}
}

// Two records under one snapshot are told apart by the fingerprint
// alone, so it has to be short enough to sit in a filename and long
// enough that two requests do not collide.
func TestFingerprintIsAShortStableName(t *testing.T) {
	t.Parallel()
	s := Source{Kind: KindHF, Repo: "o/r", Revision: "main", Include: []string{"model.gguf"}}
	first := s.Fingerprint()
	if len(first) != 12 {
		t.Errorf("Fingerprint = %q, want 12 characters", first)
	}
	if second := s.Fingerprint(); second != first {
		t.Errorf("Fingerprint is not stable: %q then %q", first, second)
	}
	// The separator matters: without it ["a","bc"] and ["ab","c"] hash
	// the same, and a main GGUF could be served a projector's record.
	a := Source{Kind: KindHF, Include: []string{"a", "bc"}}
	b := Source{Kind: KindHF, Include: []string{"ab", "c"}}
	if a.Fingerprint() == b.Fingerprint() {
		t.Error("adjacent list entries must not run together")
	}
}

// Commit records which bytes landed, not what was asked for. A branch
// that has moved upstream still matches the configured source; catching
// that move is the remote check's job.
func TestSourceEqualIgnoresCommit(t *testing.T) {
	t.Parallel()
	a := Source{Kind: KindHF, Repo: "o/r", Revision: "main", Commit: "aaa"}
	b := a
	b.Commit = "bbb"
	if !a.Equal(b) {
		t.Error("a moved branch must still match the same configured source")
	}
}

// A URL source is identified by the digest of its URL, and the URL's
// credentials must not reach the file.
func TestURLIdentity(t *testing.T) {
	t.Parallel()
	const u = "https://example.com/model.gguf"
	if URLIdentity(u) == URLIdentity(u+"?v=2") {
		t.Error("different URLs must have different identities")
	}
	if got := URLIdentity(u); len(got) != 64 {
		t.Errorf("URLIdentity = %q, want 64 hex chars", got)
	}

	path := filepath.Join(t.TempDir(), "s.json")
	secret := "https://user:hunter2@example.com/model.gguf?X-Amz-Signature=deadbeef"
	if err := Write(path, Sidecar{
		Source: Source{
			Kind:        KindURL,
			URLSHA256:   URLIdentity(secret),
			URLRedacted: "https://REDACTED@example.com/model.gguf",
		},
		Files: []File{{Path: "model.gguf", Size: 1}},
	}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	for _, leak := range []string{"hunter2", "X-Amz-Signature", "deadbeef"} {
		if strings.Contains(string(raw), leak) {
			t.Errorf("sidecar leaked %q onto a shared volume:\n%s", leak, raw)
		}
	}
}
