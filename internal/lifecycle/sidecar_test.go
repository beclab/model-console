package lifecycle

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/verify"
)

// urlEnsureFixture is one URL-source manager plus the downloader behind
// it, so a test can run repeated passes and count who actually fetched.
type urlEnsureFixture struct {
	mgr  *Manager
	dl   *fakeDownloader
	o    Options
	dest string
}

func newURLFixture(t *testing.T, url, sha string, body []byte) *urlEnsureFixture {
	t.Helper()
	dest := filepath.Join(t.TempDir(), "m.bin")
	o := baseOptions(t, []config.ModelSource{urlSource(url, sha, dest)})
	dl := &fakeDownloader{files: map[string][]byte{url: body}, etag: `"v1"`}
	o.Downloader = dl
	o.Adapter = &fakeAdapter{kind: config.EngineLlamaCpp}
	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return &urlEnsureFixture{mgr: mgr, dl: dl, o: o, dest: dest}
}

func (f *urlEnsureFixture) ensure(t *testing.T) {
	t.Helper()
	if err := f.mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}
}

const modelBody = "model bytes"

// The point of the record: a second boot with the same configuration
// reuses what is already on disk and never calls the downloader, so a
// machine that cannot reach the upstream still starts.
func TestEnsureURL_SecondPassNeverTouchesTheNetwork(t *testing.T) {
	t.Parallel()
	f := newURLFixture(t, "https://example.com/m.bin", "", []byte(modelBody))
	f.ensure(t)
	if got := f.dl.hits.Load(); got != 1 {
		t.Fatalf("first pass: downloader calls = %d, want 1", got)
	}

	// Anything reaching the upstream from here on is a failure, and the
	// downloader now says so out loud rather than quietly succeeding.
	f.dl.downloadErr = errors.New("upstream is unreachable")
	f.ensure(t)
	if got := f.dl.hits.Load(); got != 1 {
		t.Errorf("second pass: downloader calls = %d, want the first pass's 1", got)
	}
	if data, err := os.ReadFile(f.dest); err != nil || string(data) != modelBody {
		t.Errorf("model bytes = %q (%v), want them untouched", data, err)
	}
}

// A record is written next to the bytes, not somewhere central, so that
// deleting the model directory takes the record with it.
func TestEnsureURL_RecordSitsBesideTheBytes(t *testing.T) {
	t.Parallel()
	f := newURLFixture(t, "https://example.com/m.bin", "", []byte(modelBody))
	f.ensure(t)

	sc, err := verify.Read(f.dest + urlSidecarSuffix)
	if err != nil {
		t.Fatalf("Read sidecar: %v", err)
	}
	if len(sc.Files) != 1 {
		t.Fatalf("Files = %+v, want exactly the one downloaded file", sc.Files)
	}
	if sc.Files[0].Path != filepath.Base(f.dest) {
		t.Errorf("Path = %q, want %q", sc.Files[0].Path, filepath.Base(f.dest))
	}
	if sc.Files[0].Size != int64(len(modelBody)) {
		t.Errorf("Size = %d, want %d", sc.Files[0].Size, len(modelBody))
	}
	if sc.Files[0].ETag != `"v1"` {
		t.Errorf("ETag = %q, want the one the upstream advertised", sc.Files[0].ETag)
	}
	if sc.Source.Kind != verify.KindURL {
		t.Errorf("Kind = %q, want %q", sc.Source.Kind, verify.KindURL)
	}
}

// The record sits on a volume other applications mount, and a presigned
// URL carries a credential in its query. What lands in the file has to
// be enough to recognise the source and nothing more.
func TestEnsureURL_RecordKeepsNoCredential(t *testing.T) {
	t.Parallel()
	const signed = "https://user:hunter2@bucket.example.com/m.bin" +
		"?X-Amz-Credential=AKIAEXAMPLE&X-Amz-Signature=deadbeefcafe"
	f := newURLFixture(t, signed, "", []byte(modelBody))
	f.ensure(t)

	raw, err := os.ReadFile(f.dest + urlSidecarSuffix)
	if err != nil {
		t.Fatalf("read record: %v", err)
	}
	for _, secret := range []string{"hunter2", "X-Amz-Signature", "deadbeefcafe", "AKIAEXAMPLE"} {
		if strings.Contains(string(raw), secret) {
			t.Errorf("record contains %q:\n%s", secret, raw)
		}
	}

	sc, err := verify.Read(f.dest + urlSidecarSuffix)
	if err != nil {
		t.Fatalf("Read sidecar: %v", err)
	}
	if want := "https://bucket.example.com/m.bin"; sc.Source.URLRedacted != want {
		t.Errorf("URLRedacted = %q, want %q", sc.Source.URLRedacted, want)
	}

	// Stripping the query for display must not weaken the identity: the
	// same path served with a different signature is still the same
	// object, and a re-signed URL is a different one.
	if !f.mgr.reuseURLDownload(context.Background(), f.mgr.opts.Config, urlSource(signed, "", f.dest), f.dest, verify.LevelSize) {
		t.Error("the pass that wrote the record could not recognise it")
	}
}

// Pointing MODEL_SOURCE at a different URL must not serve the previous
// model just because a file of the right size is sitting there.
func TestEnsureURL_ChangedSourceRedownloads(t *testing.T) {
	t.Parallel()
	f := newURLFixture(t, "https://example.com/m.bin", "", []byte(modelBody))
	f.ensure(t)

	const other = "https://example.com/other.bin"
	f.mgr.opts.Config.Sources = []config.ModelSource{urlSource(other, "", f.dest)}
	f.dl.files[other] = []byte("different model")
	f.ensure(t)

	if got := f.dl.hits.Load(); got != 2 {
		t.Errorf("downloader calls = %d, want a second fetch for the new URL", got)
	}
	if data, _ := os.ReadFile(f.dest); string(data) != "different model" {
		t.Errorf("model bytes = %q, want the new source's", data)
	}
}

// A truncated file is the failure the size level exists for, and the
// pass has to repair it rather than hand the engine a partial model.
func TestEnsureURL_TruncatedBytesAreRepaired(t *testing.T) {
	t.Parallel()
	f := newURLFixture(t, "https://example.com/m.bin", "", []byte(modelBody))
	f.ensure(t)
	if err := os.WriteFile(f.dest, []byte("mod"), 0o644); err != nil {
		t.Fatal(err)
	}

	f.ensure(t)
	if got := f.dl.hits.Load(); got != 2 {
		t.Errorf("downloader calls = %d, want the damaged file re-fetched", got)
	}
	if data, _ := os.ReadFile(f.dest); string(data) != modelBody {
		t.Errorf("model bytes = %q, want them restored", data)
	}
}

// Corruption that preserves the length is invisible to the size level
// and is exactly what asking for the digest level buys.
func TestEnsureURL_SameSizeCorruptionNeedsTheDigestLevel(t *testing.T) {
	t.Parallel()
	f := newURLFixture(t, "https://example.com/m.bin", "", []byte(modelBody))
	f.mgr.opts.Config.Runtime.VerifyLevel = "sha256"
	f.ensure(t)

	corrupt := strings.ToUpper(modelBody)
	if err := os.WriteFile(f.dest, []byte(corrupt), 0o644); err != nil {
		t.Fatal(err)
	}

	f.mgr.opts.Config.Runtime.VerifyLevel = "size"
	f.ensure(t)
	if got := f.dl.hits.Load(); got != 1 {
		t.Errorf("size level: downloader calls = %d, want the same-size file left alone", got)
	}

	f.mgr.opts.Config.Runtime.VerifyLevel = "sha256"
	f.ensure(t)
	if got := f.dl.hits.Load(); got != 2 {
		t.Errorf("sha256 level: downloader calls = %d, want the corruption caught", got)
	}
	if data, _ := os.ReadFile(f.dest); string(data) != modelBody {
		t.Errorf("model bytes = %q, want them restored", data)
	}
}

// force=true is how an operator says "I do not trust what is there",
// so it must not consult the record at all.
func TestEnsureURL_ForceIgnoresTheRecord(t *testing.T) {
	t.Parallel()
	f := newURLFixture(t, "https://example.com/m.bin", "", []byte(modelBody))
	f.ensure(t)

	f.mgr.RetryWith(RetryOptions{Force: true})
	f.ensure(t)
	if got := f.dl.hits.Load(); got != 2 {
		t.Errorf("downloader calls = %d, want force to re-fetch regardless of the record", got)
	}
}

// A digest declared in MODEL_SOURCE has already been checked against the
// bytes, so recording it costs nothing -- and it means the digest level
// works on a later pass without the size level ever having read the
// file.
func TestEnsureURL_DeclaredDigestIsRecordedForFree(t *testing.T) {
	t.Parallel()
	f := newURLFixture(t, "https://example.com/x.gguf", shaOfOllamaBlob, []byte(ollamaBlobBody))
	f.ensure(t)

	sc, err := verify.Read(f.dest + urlSidecarSuffix)
	if err != nil {
		t.Fatalf("Read sidecar: %v", err)
	}
	if sc.Files[0].SHA256 != shaOfOllamaBlob {
		t.Errorf("SHA256 = %q, want the declared digest recorded", sc.Files[0].SHA256)
	}
	if sc.Source.DeclaredSHA256 != shaOfOllamaBlob {
		t.Errorf("DeclaredSHA256 = %q, want it part of the source identity", sc.Source.DeclaredSHA256)
	}
}

// At the size level a download with no declared digest is not hashed:
// reading back a multi-gigabyte model to record a number nobody will
// check is a full read for nothing.
func TestEnsureURL_SizeLevelDoesNotHash(t *testing.T) {
	t.Parallel()
	f := newURLFixture(t, "https://example.com/m.bin", "", []byte(modelBody))
	f.ensure(t)
	sc, _ := verify.Read(f.dest + urlSidecarSuffix)
	if sc.Files[0].SHA256 != "" {
		t.Errorf("SHA256 = %q, want no digest computed at the size level", sc.Files[0].SHA256)
	}

	g := newURLFixture(t, "https://example.com/m.bin", "", []byte(modelBody))
	g.mgr.opts.Config.Runtime.VerifyLevel = "sha256"
	g.ensure(t)
	sc, _ = verify.Read(g.dest + urlSidecarSuffix)
	if sc.Files[0].SHA256 == "" {
		t.Error("SHA256 is empty, want the digest level to record one")
	}
}

// The record lands on a volume other applications mount, so a URL's
// credentials must not travel with it.
func TestEnsureURL_RecordKeepsCredentialsOffTheVolume(t *testing.T) {
	t.Parallel()
	const secret = "https://user:hunter2@example.com/m.bin?X-Amz-Signature=deadbeef"
	f := newURLFixture(t, secret, "", []byte(modelBody))
	f.mgr.opts.Config.Sources[0].RedactedSource = "https://REDACTED@example.com/m.bin"
	f.ensure(t)

	raw, err := os.ReadFile(f.dest + urlSidecarSuffix)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	for _, leak := range []string{"hunter2", "deadbeef", "X-Amz-Signature"} {
		if strings.Contains(string(raw), leak) {
			t.Errorf("record leaked %q:\n%s", leak, raw)
		}
	}
}

// A record that survived the bytes it describes would vouch for a file
// that is no longer there.
func TestEnsureURL_DeletedBytesAreRefetched(t *testing.T) {
	t.Parallel()
	f := newURLFixture(t, "https://example.com/m.bin", "", []byte(modelBody))
	f.ensure(t)
	if err := os.Remove(f.dest); err != nil {
		t.Fatal(err)
	}

	f.ensure(t)
	if got := f.dl.hits.Load(); got != 2 {
		t.Errorf("downloader calls = %d, want the missing file re-fetched", got)
	}
}

// VERIFY_LEVEL is the standing depth; ?level= raises it for one pass and
// is the only route to remote, which reaches the network.
func TestPassLevel(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name       string
		configured string
		requested  string
		want       verify.Level
	}{
		{"default is the cheap check", "", "", verify.LevelSize},
		{"configured level applies", "sha256", "", verify.LevelSHA256},
		{"request overrides upward", "size", "sha256", verify.LevelSHA256},
		{"request overrides downward", "sha256", "size", verify.LevelSize},
		{"remote is reachable only per pass", "size", "remote", verify.LevelRemote},
		{"garbage falls back to the configured level", "sha256", "nonsense", verify.LevelSHA256},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := config.Config{Runtime: config.Runtime{VerifyLevel: tc.configured}}
			if got := passLevel(cfg, RetryOptions{Level: tc.requested}); got != tc.want {
				t.Errorf("passLevel = %q, want %q", got, tc.want)
			}
		})
	}
}
