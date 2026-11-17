package lifecycle

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/adapter/hfwrap"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/progress"
)

func TestEnterDownloadPhase_SeedsWholeRepo(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/owner/full/tree/main" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `[
			{"path":"a.bin","size":100,"type":"file"},
			{"path":"b.bin","size":200,"type":"file"}
		]`)
	}))
	t.Cleanup(server.Close)

	pm := progress.New(time.Now())
	m := newBudgetManager(t, pm, server.URL, "")
	main := config.ModelSource{Kind: config.KindHF, HFRepo: "owner/full"}
	m.pass = &passState{
		cfg:  config.Config{HFEndpoint: server.URL},
		main: &main,
	}
	m.enterDownloadPhase(context.Background())

	st := pm.Snapshot()
	if st.Phase != progress.PhaseDownload {
		t.Errorf("phase=%q, want download", st.Phase)
	}
	if st.BytesTotal != 300 || st.FilesTotal != 2 {
		t.Errorf("seed bytes=%d files=%d, want 300/2", st.BytesTotal, st.FilesTotal)
	}
	if !m.pass.entered {
		t.Error("pass.entered must be true")
	}

	m.enterDownloadPhase(context.Background())
	if pm.Snapshot().BytesTotal != 300 {
		t.Error("second enter must be a no-op")
	}
}

func TestEnterDownloadPhase_TreeFailureLeavesDynamic(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "nope", http.StatusBadGateway)
	}))
	t.Cleanup(server.Close)

	pm := progress.New(time.Now())
	m := newBudgetManager(t, pm, server.URL, "")
	main := config.ModelSource{Kind: config.KindHF, HFRepo: "owner/full"}
	m.pass = &passState{
		cfg:  config.Config{HFEndpoint: server.URL},
		main: &main,
	}
	m.enterDownloadPhase(context.Background())

	st := pm.Snapshot()
	if st.BytesTotal != 0 || st.FilesTotal != 0 {
		t.Errorf("failed tree must not pin: bytes=%d files=%d", st.BytesTotal, st.FilesTotal)
	}
}

func TestEnterDownloadPhase_ResumeSeedsCachedSnapshot(t *testing.T) {
	t.Parallel()
	cache, repo := writeHFBlobs(t, "owner/full", map[string]int{
		blobA: 100,
	})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/owner/full/tree/main" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprintf(w, `[
			{"path":"a.bin","size":100,"type":"file","oid":"%s","lfs":{"oid":"%s"}},
			{"path":"b.bin","size":200,"type":"file","oid":"%s","lfs":{"oid":"%s"}}
		]`, gitOID1, blobA, gitOID2, blobB)
	}))
	t.Cleanup(server.Close)

	pm := progress.New(time.Now())
	m := newBudgetManager(t, pm, server.URL, cache)
	main := config.ModelSource{Kind: config.KindHF, HFRepo: repo, HFRevision: "main"}
	m.pass = &passState{
		cfg:  config.Config{HFEndpoint: server.URL},
		main: &main,
	}
	m.enterDownloadPhase(context.Background())

	st := pm.Snapshot()
	if st.BytesTotal != 300 || st.BytesCompleted != 100 {
		t.Errorf("bytes %d/%d, want 100/300 cached", st.BytesCompleted, st.BytesTotal)
	}
	if st.FilesTotal != 2 || st.FilesCompleted != 1 {
		t.Errorf("files %d/%d, want 1/2 cached", st.FilesCompleted, st.FilesTotal)
	}
	if len(st.Files) != 2 {
		t.Fatalf("file list len=%d, want 2", len(st.Files))
	}
	if st.Files[0].Path != "a.bin" || st.Files[0].Status != progress.FileDone {
		t.Errorf("files[0]=%+v, want a.bin done", st.Files[0])
	}
	if st.Files[1].Path != "b.bin" || st.Files[1].Status != progress.FileWaiting {
		t.Errorf("files[1]=%+v, want b.bin waiting", st.Files[1])
	}
}

func TestEnterDownloadPhase_SeedsWaitingFileList(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/owner/full/tree/main" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `[
			{"path":"mmproj-F16.gguf","size":50,"type":"file"},
			{"path":"model.gguf","size":250,"type":"file"}
		]`)
	}))
	t.Cleanup(server.Close)

	pm := progress.New(time.Now())
	m := newBudgetManager(t, pm, server.URL, "")
	main := config.ModelSource{Kind: config.KindHF, HFRepo: "owner/full"}
	m.pass = &passState{
		cfg:  config.Config{HFEndpoint: server.URL},
		main: &main,
	}
	m.enterDownloadPhase(context.Background())

	st := pm.Snapshot()
	if len(st.Files) != 2 {
		t.Fatalf("file list len=%d, want 2", len(st.Files))
	}
	if st.Files[0].Path != "mmproj-F16.gguf" || st.Files[0].Status != progress.FileWaiting {
		t.Errorf("files[0]=%+v, want mmproj waiting", st.Files[0])
	}
	if st.Files[1].Path != "model.gguf" || st.Files[1].BytesTotal != 250 {
		t.Errorf("files[1]=%+v, want model.gguf 250", st.Files[1])
	}
}

func TestEnterDownloadPhase_ForceDropsCachedSeed(t *testing.T) {
	t.Parallel()
	cache, repo := writeHFBlobs(t, "owner/full", map[string]int{
		blobA: 100,
	})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprintf(w, `[{"path":"a.bin","size":100,"type":"file","oid":"%s","lfs":{"oid":"%s"}}]`, gitOID1, blobA)
	}))
	t.Cleanup(server.Close)

	pm := progress.New(time.Now())
	m := newBudgetManager(t, pm, server.URL, cache)
	main := config.ModelSource{Kind: config.KindHF, HFRepo: repo, HFRevision: "main"}
	m.pass = &passState{
		cfg:   config.Config{HFEndpoint: server.URL},
		main:  &main,
		force: true,
	}
	m.enterDownloadPhase(context.Background())

	st := pm.Snapshot()
	if st.BytesTotal != 100 || st.BytesCompleted != 0 {
		t.Errorf("force: bytes %d/%d, want 0/100", st.BytesCompleted, st.BytesTotal)
	}
	if st.FilesCompleted != 0 {
		t.Errorf("force: FilesCompleted=%d, want 0", st.FilesCompleted)
	}
	if len(st.Files) != 1 || st.Files[0].Status != progress.FileWaiting || st.Files[0].BytesCompleted != 0 {
		t.Errorf("force: files=%+v, want one waiting row at 0", st.Files)
	}
}

// An image ships the old model with refs/main pointing at it; main has
// since moved and every file kept its name. None of the old bytes are
// the new model, so the pass starts from zero rather than full.
func TestEnterDownloadPhase_MovedBranchDoesNotSeedOldSnapshot(t *testing.T) {
	t.Parallel()
	cache, repo := writeHFSnapshot(t, "unsloth/Qwen-GGUF", "main", map[string]int{
		"model.gguf":  250,
		"mmproj.gguf": 50,
	})
	writeBlobsInto(t, cache, repo, map[string]int{blobA: 250, blobB: 50})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `[
			{"path":"model.gguf","size":260,"type":"file","oid":"%s","lfs":{"oid":"%s"}},
			{"path":"mmproj.gguf","size":50,"type":"file","oid":"%s","lfs":{"oid":"%s"}}
		]`, gitOID1, blobC, gitOID2, blobD)
	}))
	t.Cleanup(server.Close)

	pm := progress.New(time.Now())
	m := newBudgetManager(t, pm, server.URL, cache)
	main := config.ModelSource{Kind: config.KindHF, HFRepo: repo, HFRevision: "main", HFInclude: []string{"model.gguf"}}
	mmproj := config.ModelSource{Kind: config.KindHF, Role: config.RoleMmproj, HFRepo: repo, HFRevision: "main", HFInclude: []string{"mmproj.gguf"}}
	m.pass = &passState{cfg: config.Config{HFEndpoint: server.URL}, main: &main, mmproj: &mmproj}
	m.enterDownloadPhase(context.Background())

	st := pm.Snapshot()
	if st.BytesTotal != 310 || st.BytesCompleted != 0 {
		t.Errorf("bytes %d/%d, want 0/310", st.BytesCompleted, st.BytesTotal)
	}
	if st.FilesTotal != 2 || st.FilesCompleted != 0 {
		t.Errorf("files %d/%d, want 0/2", st.FilesCompleted, st.FilesTotal)
	}
	for _, f := range st.Files {
		if f.Status != progress.FileWaiting || f.BytesCompleted != 0 {
			t.Errorf("row %+v, want waiting at 0", f)
		}
	}
}

// Only the projector moved: the weights' blob is the one main still
// lists, so it is seeded done and only the projector is left to count.
func TestEnterDownloadPhase_SeedsOnlyUnchangedBlobs(t *testing.T) {
	t.Parallel()
	cache, repo := writeHFBlobs(t, "owner/gguf", map[string]int{
		blobA:                    250,
		blobB:                    50,
		blobD + ".incomplete":    20,
		"not-a-blob-id-whatever": 50,
	})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `[
			{"path":"model.gguf","size":250,"type":"file","oid":"%s","lfs":{"oid":"%s"}},
			{"path":"mmproj.gguf","size":50,"type":"file","oid":"%s","lfs":{"oid":"%s"}}
		]`, gitOID1, blobA, gitOID2, blobD)
	}))
	t.Cleanup(server.Close)

	pm := progress.New(time.Now())
	m := newBudgetManager(t, pm, server.URL, cache)
	main := config.ModelSource{Kind: config.KindHF, HFRepo: repo, HFInclude: []string{"model.gguf"}}
	mmproj := config.ModelSource{Kind: config.KindHF, Role: config.RoleMmproj, HFRepo: repo, HFInclude: []string{"mmproj.gguf"}}
	m.pass = &passState{cfg: config.Config{HFEndpoint: server.URL}, main: &main, mmproj: &mmproj}
	m.enterDownloadPhase(context.Background())

	st := pm.Snapshot()
	if st.BytesTotal != 300 || st.BytesCompleted != 250 {
		t.Errorf("bytes %d/%d, want 250/300", st.BytesCompleted, st.BytesTotal)
	}
	if st.FilesCompleted != 1 {
		t.Errorf("FilesCompleted=%d, want 1", st.FilesCompleted)
	}
	want := map[string]progress.FileStatus{"model.gguf": progress.FileDone, "mmproj.gguf": progress.FileWaiting}
	for _, f := range st.Files {
		if f.Status != want[f.Path] {
			t.Errorf("%s status=%q, want %q", f.Path, f.Status, want[f.Path])
		}
	}
}

// The Olares mirror omits X-Linked-ETag, so hf names blobs by the
// mirror's ETag and no Hub id finds them. The target commit's snapshot
// still does; a same-named file in an older snapshot does not count.
func TestEnterDownloadPhase_SeedsTargetSnapshotWithMirrorBlobNames(t *testing.T) {
	t.Parallel()
	const (
		oldCommit = "0a9186325ff86b0b5eada4af2182542f96b6ac80"
		newCommit = "50968a4468ef4233ed78cd7c3de230dd1d61a56b"
	)
	cache := t.TempDir()
	repo := "unsloth/Qwen-GGUF"
	writeBlobsInto(t, cache, repo, map[string]int{"96a2783ccf0e573991f4f85fb18e210f-10": 250, "2dcaa90923175f6d8a9146132c660bfc-10": 50})
	writeSnapshotAt(t, cache, repo, newCommit, map[string]int{"model.gguf": 250})
	writeSnapshotAt(t, cache, repo, oldCommit, map[string]int{"model.gguf": 250, "mmproj.gguf": 50})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/" + repo + "/revision/main":
			fmt.Fprintf(w, `{"sha":%q}`, newCommit)
		case "/api/models/" + repo + "/tree/" + newCommit:
			fmt.Fprintf(w, `[
				{"path":"model.gguf","size":250,"type":"file","oid":"%s","lfs":{"oid":"%s"}},
				{"path":"mmproj.gguf","size":50,"type":"file","oid":"%s","lfs":{"oid":"%s"}}
			]`, gitOID1, blobC, gitOID2, blobD)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)

	pm := progress.New(time.Now())
	m := newBudgetManager(t, pm, server.URL, cache)
	main := config.ModelSource{Kind: config.KindHF, HFRepo: repo, HFInclude: []string{"model.gguf"}}
	mmproj := config.ModelSource{Kind: config.KindHF, Role: config.RoleMmproj, HFRepo: repo, HFInclude: []string{"mmproj.gguf"}}
	m.pass = &passState{cfg: config.Config{HFEndpoint: server.URL}, main: &main, mmproj: &mmproj}
	m.enterDownloadPhase(context.Background())

	st := pm.Snapshot()
	if st.BytesTotal != 300 || st.BytesCompleted != 250 {
		t.Errorf("bytes %d/%d, want 250/300", st.BytesCompleted, st.BytesTotal)
	}
	want := map[string]progress.FileStatus{"model.gguf": progress.FileDone, "mmproj.gguf": progress.FileWaiting}
	for _, f := range st.Files {
		if f.Status != want[f.Path] {
			t.Errorf("%s status=%q, want %q", f.Path, f.Status, want[f.Path])
		}
	}
}

// main moves between the estimate and the download. hf must fetch the
// commit the progress was seeded from, and refs/main must name it once
// it lands so the next boot finds the download record through it.
func TestEnsure_DownloadsAtTheEstimatedCommit(t *testing.T) {
	t.Parallel()
	const (
		estimated = "50968a4468ef4233ed78cd7c3de230dd1d61a56b"
		movedTo   = "1111111111111111111111111111111111111111"
		repo      = "owner/moving"
	)
	var resolves atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/" + repo + "/revision/main":
			sha := estimated
			if resolves.Add(1) > 1 {
				sha = movedTo
			}
			fmt.Fprintf(w, `{"sha":%q}`, sha)
		case "/api/models/" + repo + "/tree/" + estimated:
			fmt.Fprintf(w, `[{"path":"model.gguf","size":4,"type":"file","oid":"%s","lfs":{"oid":"%s"}}]`, gitOID1, blobA)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)

	cache := t.TempDir()
	o := baseOptions(t, []config.ModelSource{hfSource(repo, "", "model.gguf")})
	o.HFCacheRoot = cache
	o.Config.HFEndpoint = server.URL
	var got hfwrap.Config
	o.HFRun = func(_ context.Context, c hfwrap.Config, _ progress.Sink) (hfwrap.Result, error) {
		got = c
		writeSnapshotAt(t, cache, repo, c.Revision, map[string]int{"model.gguf": 4})
		return hfwrap.Result{Status: "ok", Commit: c.Revision,
			Path: filepath.Join(hfwrap.SnapshotPath(cache, repo, c.Revision), "model.gguf")}, nil
	}
	mgr, err := New(o)
	if err != nil {
		t.Fatal(err)
	}
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	if got.Revision != estimated {
		t.Errorf("hf ran at %q, want the estimated commit %s", got.Revision, estimated)
	}
	if n := resolves.Load(); n != 1 {
		t.Errorf("revision resolved %d times, want once", n)
	}
	if sha, _ := hfwrap.ReadRefCommit(cache, repo, ""); sha != estimated {
		t.Errorf("refs/main = %q, want %s", sha, estimated)
	}
	if _, ok := mgr.reuseHFDownload(context.Background(), o.Config, o.Config.Sources[0], "size"); !ok {
		t.Error("the next pass must reuse the record through refs/main")
	}
}

func TestResolveHFCommit_TriesThreeTimes(t *testing.T) {
	t.Parallel()
	const commit = "4af51b3542f775b0a8e4b310bfb3f17e01c306c9"
	var flaky, down atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/o/flaky/revision/main":
			if flaky.Add(1) < 3 {
				http.Error(w, "gateway timeout", http.StatusGatewayTimeout)
				return
			}
			fmt.Fprintf(w, `{"sha":%q}`, commit)
		case "/api/models/o/down/revision/main":
			down.Add(1)
			http.Error(w, "gateway timeout", http.StatusGatewayTimeout)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)

	m := &Manager{opts: Options{HFResolveBackoff: time.Millisecond}}
	cfg := config.Config{HFEndpoint: server.URL}
	if got := m.resolveHFTarget(context.Background(), cfg, config.ModelSource{Kind: config.KindHF, HFRepo: "o/flaky"}); got.commit != commit {
		t.Errorf("third attempt answered: got %q, want %s", got.commit, commit)
	}
	if got := m.resolveHFTarget(context.Background(), cfg, config.ModelSource{Kind: config.KindHF, HFRepo: "o/down"}); got.commit != "" {
		t.Errorf("unreachable: got %q, want none", got.commit)
	}
	if n := down.Load(); n != hfResolveAttempts {
		t.Errorf("unreachable remote asked %d times, want %d", n, hfResolveAttempts)
	}
}

// The HEAD and the revision API are asked at once. A HEAD that hangs
// costs the attempt hfHeadGrace once the API has answered, not the
// HEAD's own timeout on top of the API's.
func TestResolveHFTarget_DoesNotWaitOutAHungHead(t *testing.T) {
	t.Parallel()
	const commit = "4af51b3542f775b0a8e4b310bfb3f17e01c306c9"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/models/o/r/revision/main" {
			fmt.Fprintf(w, `{"sha":%q}`, commit)
			return
		}
		<-r.Context().Done()
	}))
	t.Cleanup(server.Close)

	m := &Manager{opts: Options{HFResolveBackoff: time.Millisecond}}
	src := config.ModelSource{Kind: config.KindHF, HFRepo: "o/r", HFInclude: []string{"model.gguf"}}
	start := time.Now()
	got := m.resolveHFTarget(context.Background(), config.Config{HFEndpoint: server.URL}, src)
	if got.commit != commit || got.entry != nil {
		t.Errorf("target %+v, want the API's commit without a size", got)
	}
	if took := time.Since(start); took > hfHeadGrace+2*time.Second {
		t.Errorf("took %s, want about hfHeadGrace (%s)", took, hfHeadGrace)
	}
}

// Everything the pass asks before downloading shares one deadline. A
// remote that never answers ends the estimate there, and the files are
// still listed.
func TestEnterDownloadPhase_StopsAtTheMetadataBudget(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	t.Cleanup(server.Close)

	pm := progress.New(time.Now())
	m := newBudgetManager(t, pm, server.URL, t.TempDir())
	m.opts.MetadataBudget = 300 * time.Millisecond
	m.opts.HFResolveBackoff = time.Millisecond
	main := config.ModelSource{Kind: config.KindHF, HFRepo: "o/r", HFInclude: []string{"model.gguf"}}
	mmproj := config.ModelSource{Kind: config.KindHF, Role: config.RoleMmproj, HFRepo: "o/r", HFInclude: []string{"mmproj.gguf"}}
	m.pass = &passState{cfg: config.Config{HFEndpoint: server.URL}, main: &main, mmproj: &mmproj}

	start := time.Now()
	m.enterDownloadPhase(context.Background())
	if took := time.Since(start); took > 3*time.Second {
		t.Errorf("estimate took %s, want it bounded by the 300ms budget", took)
	}
	st := pm.Snapshot()
	if st.BytesTotal != 0 {
		t.Errorf("bytes_total=%d, want no pinned total", st.BytesTotal)
	}
}

// The Olares mirror answered /api/ with a 401 after fifteen seconds
// while serving file HEADs. A main GGUF and its projector from one repo
// must still be listed and sized up front, from one revision, without
// the API.
func TestEnterDownloadPhase_SizesSingleFileSourcesByHead(t *testing.T) {
	t.Parallel()
	const (
		commit = "4af51b3542f775b0a8e4b310bfb3f17e01c306c9"
		repo   = "unsloth/vlm"
	)
	var api, heads atomic.Int32
	sizes := map[string]int64{"model.gguf": 1000, "mmproj.gguf": 100}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") {
			api.Add(1)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		heads.Add(1)
		var rev, file string
		if _, err := fmt.Sscanf(strings.ReplaceAll(strings.TrimPrefix(r.URL.Path, "/"+repo+"/resolve/"), "/", " "), "%s %s", &rev, &file); err != nil || r.Method != http.MethodHead || sizes[file] == 0 {
			http.NotFound(w, r)
			return
		}
		if rev != "main" && rev != commit {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("X-Repo-Commit", commit)
		w.Header().Set("ETag", `"0a6cc728b3879babffd375bdba8cafd0-10"`)
		w.Header().Set("Content-Length", fmt.Sprint(sizes[file]))
	}))
	t.Cleanup(server.Close)

	cache := t.TempDir()
	writeSnapshotAt(t, cache, repo, commit, map[string]int{"mmproj.gguf": 100})
	pm := progress.New(time.Now())
	m := newBudgetManager(t, pm, server.URL, cache)
	main := config.ModelSource{Index: 1, Kind: config.KindHF, HFRepo: repo, HFInclude: []string{"model.gguf"}}
	extra := config.ModelSource{Index: 2, Kind: config.KindHF, Role: config.RoleExtra, HFRepo: repo, HFInclude: []string{"mmproj.gguf"}}
	m.pass = &passState{cfg: config.Config{HFEndpoint: server.URL}, main: &main, extras: []config.ModelSource{extra}}
	m.enterDownloadPhase(context.Background())

	st := pm.Snapshot()
	if st.BytesTotal != 1100 || st.BytesCompleted != 100 {
		t.Errorf("bytes %d/%d, want 100/1100", st.BytesCompleted, st.BytesTotal)
	}
	want := map[string]progress.FileStatus{"model.gguf": progress.FileWaiting, "mmproj.gguf": progress.FileDone}
	if len(st.Files) != len(want) {
		t.Fatalf("files %+v, want both listed", st.Files)
	}
	for _, f := range st.Files {
		if f.Status != want[f.Path] {
			t.Errorf("%s status=%q, want %q", f.Path, f.Status, want[f.Path])
		}
	}
	if c := m.pass.hfCommit(main); c != commit {
		t.Errorf("main commit %q, want %s", c, commit)
	}
	if c := m.pass.hfCommit(extra); c != commit {
		t.Errorf("extra commit %q, want %s", c, commit)
	}
	if n := api.Load(); n > 1 {
		t.Errorf("/api/ asked %d times, want only the revision lookup beside the first HEAD", n)
	}
	if n := heads.Load(); n != 2 {
		t.Errorf("%d HEADs, want one per file", n)
	}
}

// With the remote unreachable, hf answers from refs/main and exits 0.
// The pass goes on with that snapshot, and says so. Not parallel: it
// swaps the default logger.
func TestEnsure_WarnsWhenHFServesTheCachedSnapshot(t *testing.T) {
	const (
		old  = "4ca720788d1e01f1bff70c033e0d0028fd02e502"
		repo = "owner/offline"
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "gateway timeout", http.StatusGatewayTimeout)
	}))
	t.Cleanup(server.Close)

	cache := t.TempDir()
	writeSnapshotAt(t, cache, repo, old, map[string]int{"model.gguf": 4})
	if err := hfwrap.WriteRefCommit(cache, repo, "", old); err != nil {
		t.Fatal(err)
	}

	var logs bytes.Buffer
	restore := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(restore) })

	o := baseOptions(t, []config.ModelSource{hfSource(repo, "", "model.gguf")})
	o.HFCacheRoot = cache
	o.Config.HFEndpoint = server.URL
	var got hfwrap.Config
	o.HFRun = func(_ context.Context, c hfwrap.Config, _ progress.Sink) (hfwrap.Result, error) {
		got = c
		return hfwrap.Result{Status: "ok", Commit: old,
			Path: filepath.Join(hfwrap.SnapshotPath(cache, repo, old), "model.gguf")}, nil
	}
	mgr, err := New(o)
	if err != nil {
		t.Fatal(err)
	}
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	if got.Revision != "" {
		t.Errorf("hf ran at %q, want the revision as given", got.Revision)
	}
	out := logs.String()
	for _, want := range []string{
		`"msg":"remote unreachable: hf revision not resolved; a cached snapshot may be used instead"`,
		`"attempts":3`,
		`"msg":"hf revision not confirmed before the download; hf returned the cached commit, which may be out of date"`,
		`"cached_commit":"` + old + `"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("logs lack %s:\n%s", want, out)
		}
	}
}

// hf reaching the remote on its own and landing a newer commit is not
// a fallback, and must not be reported as one.
func TestEnsure_NoCacheWarningWhenHFFetchesANewCommit(t *testing.T) {
	const (
		old   = "4ca720788d1e01f1bff70c033e0d0028fd02e502"
		fresh = "4af51b3542f775b0a8e4b310bfb3f17e01c306c9"
		repo  = "owner/recovered"
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "gateway timeout", http.StatusGatewayTimeout)
	}))
	t.Cleanup(server.Close)

	cache := t.TempDir()
	writeSnapshotAt(t, cache, repo, old, map[string]int{"model.gguf": 4})
	if err := hfwrap.WriteRefCommit(cache, repo, "", old); err != nil {
		t.Fatal(err)
	}

	var logs bytes.Buffer
	restore := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(restore) })

	o := baseOptions(t, []config.ModelSource{hfSource(repo, "", "model.gguf")})
	o.HFCacheRoot = cache
	o.Config.HFEndpoint = server.URL
	o.HFRun = func(_ context.Context, _ hfwrap.Config, _ progress.Sink) (hfwrap.Result, error) {
		writeSnapshotAt(t, cache, repo, fresh, map[string]int{"model.gguf": 5})
		return hfwrap.Result{Status: "ok", Commit: fresh,
			Path: filepath.Join(hfwrap.SnapshotPath(cache, repo, fresh), "model.gguf")}, nil
	}
	mgr, err := New(o)
	if err != nil {
		t.Fatal(err)
	}
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	if strings.Contains(logs.String(), "hf returned the cached commit") {
		t.Errorf("hf fetched %s, yet the pass reported the cache:\n%s", fresh, logs.String())
	}
}

func TestEstimateDownloadBudget_ExcludeAndMissingExact(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `[
			{"path":"README.md","size":10,"type":"file"},
			{"path":"model.gguf","size":90,"type":"file"}
		]`)
	}))
	t.Cleanup(server.Close)

	excl := config.ModelSource{
		Kind: config.KindHF, HFRepo: "owner/r", HFExclude: []string{"*.md"},
	}
	b := (&Manager{}).estimateDownloadBudget(
		context.Background(),
		config.Config{HFEndpoint: server.URL},
		&excl, nil, nil,
	)
	if !b.canPin || b.bytes != 90 || b.files != 1 {
		t.Fatalf("exclude: %+v, want bytes=90 files=1", b)
	}

	missing := config.ModelSource{
		Kind: config.KindHF, HFRepo: "owner/r",
		HFInclude: []string{"model.gguf", "absent.gguf"},
	}
	b = (&Manager{}).estimateDownloadBudget(
		context.Background(),
		config.Config{HFEndpoint: server.URL},
		&missing, nil, nil,
	)
	if b.canPin {
		t.Fatalf("missing exact include must not pin: %+v", b)
	}
}

func TestEstimateDownloadBudget_Subdir(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `[
			{"path":"tokenizer.json","size":7,"type":"file"},
			{"path":"onnx/model.onnx","size":40,"type":"file"},
			{"path":"onnx/config.json","size":5,"type":"file"}
		]`)
	}))
	t.Cleanup(server.Close)

	main := config.ModelSource{Kind: config.KindHF, HFRepo: "owner/emb", HFSubdir: "onnx"}
	b := (&Manager{}).estimateDownloadBudget(
		context.Background(),
		config.Config{HFEndpoint: server.URL},
		&main, nil, nil,
	)
	if !b.canPin || b.bytes != 45 || b.files != 2 {
		t.Fatalf("subdir: %+v, want bytes=45 files=2", b)
	}
}

func newBudgetManager(t *testing.T, pm progress.Manager, endpoint, cache string) *Manager {
	t.Helper()
	return &Manager{
		opts: Options{
			Manager:     pm,
			NowFunc:     time.Now,
			HFCacheRoot: cache,
			Config:      config.Config{HFEndpoint: endpoint},
		},
	}
}

func writeHFSnapshot(t *testing.T, repo, ref string, files map[string]int) (cache, repoID string) {
	t.Helper()
	cache = t.TempDir()
	sha := "cccccccccccccccccccccccccccccccccccccccc"
	refs := filepath.Join(hfwrap.HFRepoCacheDir(cache, repo), "refs")
	if err := os.MkdirAll(refs, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(refs, ref), []byte(sha+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	snap := hfwrap.SnapshotPath(cache, repo, sha)
	for rel, size := range files {
		p := filepath.Join(snap, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, make([]byte, size), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return cache, repo
}

func writeSnapshotAt(t *testing.T, cache, repo, commit string, files map[string]int) {
	t.Helper()
	snap := hfwrap.SnapshotPath(cache, repo, commit)
	for rel, size := range files {
		p := filepath.Join(snap, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, make([]byte, size), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

const (
	blobA   = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	blobB   = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	blobC   = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
	blobD   = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
	gitOID1 = "1111111111111111111111111111111111111111"
	gitOID2 = "2222222222222222222222222222222222222222"
)

// writeHFBlobs lays files out under a repo's blobs/ by name, the way the
// HF cache stores content.
func writeHFBlobs(t *testing.T, repo string, blobs map[string]int) (cache, repoID string) {
	t.Helper()
	cache = t.TempDir()
	writeBlobsInto(t, cache, repo, blobs)
	return cache, repo
}

func writeBlobsInto(t *testing.T, cache, repo string, blobs map[string]int) {
	t.Helper()
	dir := filepath.Join(hfwrap.HFRepoCacheDir(cache, repo), "blobs")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, size := range blobs {
		if err := os.WriteFile(filepath.Join(dir, name), make([]byte, size), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}
