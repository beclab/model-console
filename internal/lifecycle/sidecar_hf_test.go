package lifecycle

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/llm-init/llm-init/internal/adapter/hfwrap"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/progress"
	"github.com/llm-init/llm-init/internal/verify"
)

// hfFixture is one HF-source manager over a temporary cache, plus a
// stand-in for the `hf` CLI that writes huggingface_hub's own layout so
// the record-reading code is exercised against the real shape rather
// than one invented for the test.
type hfFixture struct {
	mgr    *Manager
	root   string
	commit string
	spawns *atomic.Int32
	fail   error
}

const (
	hfRepo      = "owner/repo"
	hfCommit    = "0f1e2d3c4b5a69788796a5b4c3d2e1f001234567"
	hfModelBody = "gguf bytes"
)

// newHFFixture wires a manager whose `hf` stub materialises files in the
// cache on each call. Callers name the files one download produces.
func newHFFixture(t *testing.T, ms config.ModelSource, files map[string]string) *hfFixture {
	t.Helper()
	root := t.TempDir()
	o := baseOptions(t, []config.ModelSource{ms})
	o.HFCacheRoot = root
	o.Adapter = &fakeAdapter{kind: config.EngineLlamaCpp}

	f := &hfFixture{root: root, commit: hfCommit, spawns: &atomic.Int32{}}
	o.HFRun = func(_ context.Context, c hfwrap.Config, _ progress.Sink) (hfwrap.Result, error) {
		f.spawns.Add(1)
		if f.fail != nil {
			return hfwrap.Result{}, f.fail
		}
		seedHFSnapshot(t, root, c.Repo, f.commit, files)
		return hfwrap.Result{
			Status: "ok",
			Commit: f.commit,
			Path:   hfResultPath(root, c, f.commit),
		}, nil
	}

	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	f.mgr = mgr
	return f
}

func (f *hfFixture) ensure(t *testing.T) {
	t.Helper()
	if err := f.mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}
}

func (f *hfFixture) snapshot() string {
	return hfwrap.SnapshotPath(f.root, hfRepo, f.commit)
}

// registered returns the files handed to the adapter on the last pass.
func (f *hfFixture) registered() []string {
	ad := f.mgr.opts.Adapter.(*fakeAdapter)
	ad.regMu.Lock()
	defer ad.regMu.Unlock()
	if len(ad.regFiles) == 0 {
		return nil
	}
	return ad.regFiles[len(ad.regFiles)-1]
}

// hfResultPath mirrors what hfwrap resolves: the file itself for a
// single-file download, the snapshot directory otherwise.
func hfResultPath(root string, c hfwrap.Config, commit string) string {
	snap := hfwrap.SnapshotPath(root, c.Repo, commit)
	if c.HFFile != "" {
		return filepath.Join(snap, c.HFFile)
	}
	return snap
}

// seedHFSnapshot writes huggingface_hub's cache layout: content under
// blobs/ named after the ETag, snapshot entries as relative symlinks
// into it, and the resolved commit in refs/main. The digest-for-free
// path depends on all three, so a test that faked it with plain files
// would prove nothing.
func seedHFSnapshot(t *testing.T, root, repo, commit string, files map[string]string) {
	t.Helper()
	repoDir := hfwrap.HFRepoCacheDir(root, repo)
	snap := hfwrap.SnapshotPath(root, repo, commit)
	for _, d := range []string{filepath.Join(repoDir, "blobs"), filepath.Join(repoDir, "refs"), snap} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(repoDir, "refs", "main"), []byte(commit), 0o644); err != nil {
		t.Fatal(err)
	}
	for name, body := range files {
		sum := sha256.Sum256([]byte(body))
		etag := hex.EncodeToString(sum[:])
		if err := os.WriteFile(filepath.Join(repoDir, "blobs", etag), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		link := filepath.Join(snap, name)
		_ = os.Remove(link)
		if err := os.Symlink(filepath.Join("..", "..", "blobs", etag), link); err != nil {
			t.Fatal(err)
		}
	}
}

// The reason the record exists on this channel: `hf download` reaches
// the Hub even when the snapshot is complete, so without a record a
// machine that cannot resolve huggingface.co fails to boot a model it
// already has.
func TestEnsureHF_SecondPassNeverSpawnsTheCLI(t *testing.T) {
	t.Parallel()
	f := newHFFixture(t, hfSource(hfRepo, "main", "model.gguf"),
		map[string]string{"model.gguf": hfModelBody})
	f.ensure(t)
	if got := f.spawns.Load(); got != 1 {
		t.Fatalf("first pass: hf spawns = %d, want 1", got)
	}

	f.fail = errors.New("huggingface.co is unreachable")
	f.ensure(t)
	if got := f.spawns.Load(); got != 1 {
		t.Errorf("second pass: hf spawns = %d, want the first pass's 1", got)
	}
}

// A snapshot entry is a symlink whose name is the ETag, and for an LFS
// file that ETag is the content's sha256. Recording it turns the digest
// of a multi-gigabyte model into one readlink.
func TestEnsureHF_DigestComesFromTheBlobName(t *testing.T) {
	t.Parallel()
	f := newHFFixture(t, hfSource(hfRepo, "main", "model.gguf"),
		map[string]string{"model.gguf": hfModelBody})
	f.ensure(t)

	sc := readOnlyHFRecord(t, f)
	if len(sc.Files) != 1 || sc.Files[0].Path != "model.gguf" {
		t.Fatalf("Files = %+v, want the one snapshot entry", sc.Files)
	}
	sum := sha256.Sum256([]byte(hfModelBody))
	if want := hex.EncodeToString(sum[:]); sc.Files[0].SHA256 != want {
		t.Errorf("SHA256 = %q, want %q taken from the blob name", sc.Files[0].SHA256, want)
	}
	if sc.Source.Kind != verify.KindHF || sc.Source.Repo != hfRepo {
		t.Errorf("Source = %+v, want the hf request that produced it", sc.Source)
	}
}

// The record lands inside the repo's own cache directory, so deleting
// the repo from the cache takes its records with it.
func TestEnsureHF_RecordLivesUnderTheRepoCacheDir(t *testing.T) {
	t.Parallel()
	f := newHFFixture(t, hfSource(hfRepo, "main", "model.gguf"),
		map[string]string{"model.gguf": hfModelBody})
	f.ensure(t)

	repoDir := hfwrap.HFRepoCacheDir(f.root, hfRepo)
	entries, err := os.ReadDir(filepath.Join(repoDir, hfSidecarDir))
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("records = %d, want exactly one", len(entries))
	}
	if !strings.HasPrefix(entries[0].Name(), f.commit+"-") {
		t.Errorf("record name = %q, want it to name the commit it describes", entries[0].Name())
	}
}

// A main GGUF and its projector are two requests against one snapshot,
// differing only in --include. Each needs its own record; one name for
// both would have each overwrite the other and re-download every boot.
func TestEnsureHF_SiblingRequestsKeepSeparateRecords(t *testing.T) {
	t.Parallel()
	main := hfSource(hfRepo, "main", "model.gguf")
	mmproj := config.ModelSource{
		Index:      2,
		Kind:       config.KindHF,
		Role:       config.RoleMmproj,
		HFRepo:     hfRepo,
		HFRevision: "main",
		HFInclude:  []string{"mmproj.gguf"},
	}
	f := newHFFixture(t, main, map[string]string{
		"model.gguf":  hfModelBody,
		"mmproj.gguf": "projector bytes",
	})
	f.mgr.opts.Config.Sources = []config.ModelSource{main, mmproj}

	f.ensure(t)
	if got := f.spawns.Load(); got != 2 {
		t.Fatalf("first pass: hf spawns = %d, want one per request", got)
	}

	entries, err := os.ReadDir(filepath.Join(hfwrap.HFRepoCacheDir(f.root, hfRepo), hfSidecarDir))
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(entries) != 2 {
		t.Errorf("records = %d, want one per request: %v", len(entries), names(entries))
	}

	f.fail = errors.New("huggingface.co is unreachable")
	f.ensure(t)
	if got := f.spawns.Load(); got != 2 {
		t.Errorf("second pass: hf spawns = %d, want both requests served from their records", got)
	}
}

// One snapshot holds whatever every request against that revision
// fetched. A record that listed all of it would put the main GGUF in the
// projector's record too, and VERIFY_LEVEL=sha256 would then re-hash the
// weights once per source on every boot.
func TestEnsureHF_RecordCoversOnlyItsOwnRequest(t *testing.T) {
	t.Parallel()
	main := hfSource(hfRepo, "main", "model.gguf")
	mmproj := config.ModelSource{
		Index:      2,
		Kind:       config.KindHF,
		Role:       config.RoleMmproj,
		HFRepo:     hfRepo,
		HFRevision: "main",
		HFInclude:  []string{"mmproj.gguf"},
	}
	f := newHFFixture(t, main, map[string]string{
		"model.gguf":  hfModelBody,
		"mmproj.gguf": "projector bytes",
	})
	f.mgr.opts.Config.Sources = []config.ModelSource{main, mmproj}
	f.ensure(t)

	dir := filepath.Join(hfwrap.HFRepoCacheDir(f.root, hfRepo), hfSidecarDir)
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	seen := map[string][]string{}
	for _, e := range entries {
		sc, rerr := verify.Read(filepath.Join(dir, e.Name()))
		if rerr != nil {
			t.Fatalf("Read %s: %v", e.Name(), rerr)
		}
		var paths []string
		for _, file := range sc.Files {
			paths = append(paths, file.Path)
		}
		seen[strings.Join(sc.Source.Include, ",")] = paths
	}

	if got := seen["model.gguf"]; len(got) != 1 || got[0] != "model.gguf" {
		t.Errorf("main record lists %v, want only its own file", got)
	}
	if got := seen["mmproj.gguf"]; len(got) != 1 || got[0] != "mmproj.gguf" {
		t.Errorf("projector record lists %v, want only its own file", got)
	}
}

// Narrowing or widening --include is a different request against the
// same repo and commit, and must not be served by the old record.
func TestEnsureHF_ChangedIncludeRedownloads(t *testing.T) {
	t.Parallel()
	f := newHFFixture(t, hfSource(hfRepo, "main", "model.gguf"),
		map[string]string{"model.gguf": hfModelBody, "other.gguf": "other bytes"})
	f.ensure(t)

	f.mgr.opts.Config.Sources = []config.ModelSource{hfSource(hfRepo, "main", "other.gguf")}
	f.ensure(t)
	if got := f.spawns.Load(); got != 2 {
		t.Errorf("hf spawns = %d, want the new --include fetched", got)
	}
}

// Same-length corruption is invisible to the size level and is exactly
// what the digest level buys -- for free here, since the digest was
// recorded from the blob name.
func TestEnsureHF_SameSizeCorruptionNeedsTheDigestLevel(t *testing.T) {
	t.Parallel()
	f := newHFFixture(t, hfSource(hfRepo, "main", "model.gguf"),
		map[string]string{"model.gguf": hfModelBody})
	f.ensure(t)

	// Overwrite through the symlink, as a damaged blob would be.
	corrupt := strings.ToUpper(hfModelBody)
	if err := os.WriteFile(filepath.Join(f.snapshot(), "model.gguf"), []byte(corrupt), 0o644); err != nil {
		t.Fatal(err)
	}

	f.ensure(t)
	if got := f.spawns.Load(); got != 1 {
		t.Errorf("size level: hf spawns = %d, want the same-size file left alone", got)
	}

	f.mgr.opts.Config.Runtime.VerifyLevel = "sha256"
	f.ensure(t)
	if got := f.spawns.Load(); got != 2 {
		t.Errorf("sha256 level: hf spawns = %d, want the corruption caught", got)
	}
}

// A record that outlived the bytes it describes would vouch for a file
// that is no longer there.
func TestEnsureHF_DeletedSnapshotFileIsRefetched(t *testing.T) {
	t.Parallel()
	f := newHFFixture(t, hfSource(hfRepo, "main", "model.gguf"),
		map[string]string{"model.gguf": hfModelBody})
	f.ensure(t)
	if err := os.Remove(filepath.Join(f.snapshot(), "model.gguf")); err != nil {
		t.Fatal(err)
	}

	f.ensure(t)
	if got := f.spawns.Load(); got != 2 {
		t.Errorf("hf spawns = %d, want the missing file re-fetched", got)
	}
}

// force=true is an operator saying they do not trust what is on disk,
// so the record must not be consulted at all.
func TestEnsureHF_ForceIgnoresTheRecord(t *testing.T) {
	t.Parallel()
	f := newHFFixture(t, hfSource(hfRepo, "main", "model.gguf"),
		map[string]string{"model.gguf": hfModelBody})
	f.ensure(t)

	f.mgr.RetryWith(RetryOptions{Force: true})
	f.ensure(t)
	if got := f.spawns.Load(); got != 2 {
		t.Errorf("hf spawns = %d, want force to re-download regardless of the record", got)
	}
}

// Reuse has to hand the engine the same path the download did. For a
// single --include that is the file inside the snapshot, not the
// snapshot itself, and an engine handed the directory would not start.
func TestEnsureHF_ReuseReturnsTheSameEnginePath(t *testing.T) {
	t.Parallel()
	f := newHFFixture(t, hfSource(hfRepo, "main", "model.gguf"),
		map[string]string{"model.gguf": hfModelBody})
	f.ensure(t)
	downloaded := f.registered()

	f.fail = errors.New("huggingface.co is unreachable")
	f.ensure(t)
	reused := f.registered()

	want := filepath.Join(f.snapshot(), "model.gguf")
	if len(reused) != 1 || reused[0] != want {
		t.Fatalf("reused Register files = %v, want [%q]", reused, want)
	}
	if len(downloaded) != 1 || downloaded[0] != reused[0] {
		t.Errorf("Register files changed between passes: %v then %v", downloaded, reused)
	}
}

// A whole-repo download resolves to the snapshot directory, which the
// record stores as the empty engine path rather than as an absolute one
// that would break if the cache moved.
func TestEnsureHF_WholeRepoRecordsNoEnginePath(t *testing.T) {
	t.Parallel()
	f := newHFFixture(t, hfSource(hfRepo, "main"),
		map[string]string{"config.json": "{}", "model.safetensors": hfModelBody})
	f.ensure(t)

	if sc := readOnlyHFRecord(t, f); sc.EnginePath != "" {
		t.Errorf("EnginePath = %q, want empty for a whole-snapshot download", sc.EnginePath)
	}

	f.fail = errors.New("huggingface.co is unreachable")
	f.ensure(t)
	if got := f.registered(); len(got) != 1 || got[0] != f.snapshot() {
		t.Errorf("Register files = %v, want the snapshot directory %q", got, f.snapshot())
	}
}

// Not every cache entry is a symlink into blobs/ -- an operator can
// populate a snapshot by hand, and huggingface_hub can be told to copy
// rather than link. Those files are recorded by size alone, which still
// catches a truncated file.
func TestEnsureHF_PlainFileRecordsSizeOnly(t *testing.T) {
	t.Parallel()
	f := newHFFixture(t, hfSource(hfRepo, "main", "model.gguf"),
		map[string]string{"model.gguf": hfModelBody})
	f.ensure(t)

	// Replace the symlink with a real file and re-record.
	path := filepath.Join(f.snapshot(), "model.gguf")
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(hfModelBody), 0o644); err != nil {
		t.Fatal(err)
	}
	f.mgr.writeHFSidecar(f.mgr.opts.Config, f.mgr.opts.Config.Sources[0],
		hfwrap.Result{Commit: f.commit, Path: path})

	sc := readOnlyHFRecord(t, f)
	if sc.Files[0].SHA256 != "" {
		t.Errorf("SHA256 = %q, want none inferred from a plain file", sc.Files[0].SHA256)
	}
	if sc.Files[0].Size != int64(len(hfModelBody)) {
		t.Errorf("Size = %d, want %d", sc.Files[0].Size, len(hfModelBody))
	}

	f.mgr.opts.Config.Runtime.VerifyLevel = "sha256"
	f.fail = errors.New("huggingface.co is unreachable")
	f.ensure(t)
	if got := f.spawns.Load(); got != 1 {
		t.Errorf("hf spawns = %d, want a size-only entry to still satisfy the digest level", got)
	}
}

// readOnlyHFRecord returns the single record in the repo's cache
// directory, failing if there is not exactly one.
func readOnlyHFRecord(t *testing.T, f *hfFixture) verify.Sidecar {
	t.Helper()
	dir := filepath.Join(hfwrap.HFRepoCacheDir(f.root, hfRepo), hfSidecarDir)
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("records = %v, want exactly one", names(entries))
	}
	sc, err := verify.Read(filepath.Join(dir, entries[0].Name()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	return sc
}

func names(entries []os.DirEntry) []string {
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		out = append(out, e.Name())
	}
	return out
}
