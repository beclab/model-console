package lifecycle

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/adapter/hfwrap"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/progress"
	"github.com/llm-init/llm-init/internal/verify"
)

// urlUpstream is a stand-in for whatever serves a MODEL_SOURCE URL. Only
// its HEAD answer matters here: that is the whole of what a remote check
// on this channel can ask.
type urlUpstream struct {
	*httptest.Server
	etag  atomic.Value // string
	heads atomic.Int32
	body  string
}

func newURLUpstream(t *testing.T, body, etag string) *urlUpstream {
	t.Helper()
	up := &urlUpstream{body: body}
	up.etag.Store(etag)
	up.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if e, _ := up.etag.Load().(string); e != "" {
			w.Header().Set("ETag", e)
		}
		w.Header().Set("Content-Length", fmt.Sprint(len(up.body)))
		if r.Method == http.MethodHead {
			up.heads.Add(1)
			w.WriteHeader(http.StatusOK)
			return
		}
		_, _ = w.Write([]byte(up.body))
	}))
	t.Cleanup(up.Close)
	return up
}

// remoteURLFixture is a URL source that has already been downloaded
// once, ready for a `?level=remote` pass against a live upstream.
func remoteURLFixture(t *testing.T, up *urlUpstream) *urlEnsureFixture {
	t.Helper()
	f := newURLFixture(t, up.URL, "", []byte(up.body))
	f.dl.etag = mustETag(up)
	f.ensure(t)
	if got := f.dl.hits.Load(); got != 1 {
		t.Fatalf("setup: downloader calls = %d, want 1", got)
	}
	return f
}

func mustETag(up *urlUpstream) string {
	e, _ := up.etag.Load().(string)
	return e
}

func (f *urlEnsureFixture) ensureRemote(t *testing.T) {
	t.Helper()
	f.mgr.RetryWith(RetryOptions{Level: "remote"})
	f.ensure(t)
}

func (f *urlEnsureFixture) note() string {
	return f.mgr.opts.Manager.Snapshot().Note
}

// An upstream that still serves the same object is the answer the
// operator was hoping for, and it must not leave a warning behind.
func TestDriftURL_UnchangedUpstreamIsReused(t *testing.T) {
	t.Parallel()
	up := newURLUpstream(t, modelBody, `"v1"`)
	f := remoteURLFixture(t, up)

	f.ensureRemote(t)
	if got := f.dl.hits.Load(); got != 1 {
		t.Errorf("downloader calls = %d, want the local copy kept", got)
	}
	if up.heads.Load() == 0 {
		t.Error("the remote level never asked the upstream anything")
	}
	if n := f.note(); n != "" {
		t.Errorf("note = %q, want nothing said about an upstream that has not moved", n)
	}
}

// A CDN in front of an origin may mark the same ETag weak on one
// response and strong on the next. Reading that as a changed object
// would report drift on every check, and re-download the model on every
// check under follow.
func TestDriftURL_WeakAndStrongETagAreTheSameObject(t *testing.T) {
	t.Parallel()
	up := newURLUpstream(t, modelBody, `"v1"`)
	f := remoteURLFixture(t, up)
	f.mgr.opts.Config.Runtime.VerifyOnDrift = config.VerifyOnDriftFollow
	up.etag.Store(`W/"v1"`)

	f.ensureRemote(t)
	if got := f.dl.hits.Load(); got != 1 {
		t.Errorf("downloader calls = %d, want the same object recognised", got)
	}
	if n := f.note(); n != "" {
		t.Errorf("note = %q, want nothing said about a difference in spelling", n)
	}
}

// The default is to say the upstream moved, not to act on it: a model
// application is usually serving requests by the time anybody runs a
// remote check, and following would swap the weights under it.
func TestDriftURL_ReportKeepsTheLocalBytes(t *testing.T) {
	t.Parallel()
	up := newURLUpstream(t, modelBody, `"v1"`)
	f := remoteURLFixture(t, up)
	up.etag.Store(`"v2"`)

	f.ensureRemote(t)
	if got := f.dl.hits.Load(); got != 1 {
		t.Errorf("downloader calls = %d, want report to leave the model alone", got)
	}
	if n := f.note(); !strings.Contains(n, "VERIFY_ON_DRIFT") {
		t.Errorf("note = %q, want it to name the setting that would follow the change", n)
	}
}

// VERIFY_ON_DRIFT=follow is the operator asking for the opposite, and it
// only applies once a check has actually found a difference.
func TestDriftURL_FollowRedownloads(t *testing.T) {
	t.Parallel()
	up := newURLUpstream(t, modelBody, `"v1"`)
	f := remoteURLFixture(t, up)
	f.mgr.opts.Config.Runtime.VerifyOnDrift = config.VerifyOnDriftFollow
	up.etag.Store(`"v2"`)

	f.ensureRemote(t)
	if got := f.dl.hits.Load(); got != 2 {
		t.Errorf("downloader calls = %d, want follow to fetch the new object", got)
	}
}

// An unreachable upstream is not a clean bill of health. The bytes stay,
// because they verified locally, but the operator has to be told the
// question went unanswered rather than reading silence as "confirmed".
func TestDriftURL_UnreachableUpstreamSaysSo(t *testing.T) {
	t.Parallel()
	up := newURLUpstream(t, modelBody, `"v1"`)
	f := remoteURLFixture(t, up)
	up.Close()

	f.ensureRemote(t)
	if got := f.dl.hits.Load(); got != 1 {
		t.Errorf("downloader calls = %d, want the verified local bytes kept", got)
	}
	if n := f.note(); n == "" {
		t.Error("note is empty, want the unanswered check reported")
	}
}

// An upstream that sends no ETag can only be compared by length, and a
// matching length proves very little. Reporting that as confirmed would
// be the check's worst failure: it is exactly the case an operator ran
// it to rule out.
func TestDriftURL_NoETagIsNotAConfirmation(t *testing.T) {
	t.Parallel()
	up := newURLUpstream(t, modelBody, "")
	f := remoteURLFixture(t, up)

	f.ensureRemote(t)
	if got := f.dl.hits.Load(); got != 1 {
		t.Errorf("downloader calls = %d, want the local bytes kept", got)
	}
	if n := f.note(); n == "" {
		t.Error("note is empty, want an upstream that would not answer reported as such")
	}
}

// hfUpstream serves the Hub tree API for one repo.
type hfUpstream struct {
	*httptest.Server
	mu     atomic.Value // map[string]string: path -> body whose sha256 is the oid
	commit atomic.Value // string
	calls  atomic.Int32
}

func newHFUpstream(t *testing.T, commit string, files map[string]string) *hfUpstream {
	t.Helper()
	up := &hfUpstream{}
	up.mu.Store(files)
	up.commit.Store(commit)
	up.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "/tree/") {
			http.NotFound(w, r)
			return
		}
		up.calls.Add(1)
		type lfs struct {
			OID string `json:"oid"`
		}
		type entry struct {
			Path string `json:"path"`
			Size int64  `json:"size"`
			Type string `json:"type"`
			LFS  *lfs   `json:"lfs,omitempty"`
		}
		current, _ := up.mu.Load().(map[string]string)
		out := make([]entry, 0, len(current))
		for name, body := range current {
			sum := sha256.Sum256([]byte(body))
			out = append(out, entry{
				Path: name, Size: int64(len(body)), Type: "file",
				LFS: &lfs{OID: hex.EncodeToString(sum[:])},
			})
		}
		c, _ := up.commit.Load().(string)
		w.Header().Set("X-Repo-Commit", c)
		_ = json.NewEncoder(w).Encode(out)
	}))
	t.Cleanup(up.Close)
	return up
}

// remoteHFFixture is an HF source already downloaded once, pointed at a
// Hub stand-in.
func remoteHFFixture(t *testing.T, files map[string]string) (*hfFixture, *hfUpstream) {
	t.Helper()
	up := newHFUpstream(t, hfCommit, files)
	f := newHFFixture(t, hfSource(hfRepo, "main", "model.gguf"), files)
	f.mgr.opts.Config.HFEndpoint = up.URL
	f.ensure(t)
	if got := f.spawns.Load(); got != 1 {
		t.Fatalf("setup: hf spawns = %d, want 1", got)
	}
	return f, up
}

func (f *hfFixture) ensureRemote(t *testing.T) {
	t.Helper()
	f.mgr.RetryWith(RetryOptions{Level: "remote"})
	f.ensure(t)
}

func TestDriftHF_UnchangedTreeIsReused(t *testing.T) {
	t.Parallel()
	f, up := remoteHFFixture(t, map[string]string{"model.gguf": hfModelBody})

	f.ensureRemote(t)
	if got := f.spawns.Load(); got != 1 {
		t.Errorf("hf spawns = %d, want the cached snapshot kept", got)
	}
	if up.calls.Load() == 0 {
		t.Error("the remote level never asked the Hub anything")
	}
}

// The commit is the wrong test. A revision whose README moved resolves
// to a new commit while every weight file is byte-identical, and
// treating that as drift would have follow re-download gigabytes to
// land exactly what is already there.
func TestDriftHF_NewCommitWithUnchangedFilesIsNotDrift(t *testing.T) {
	t.Parallel()
	f, up := remoteHFFixture(t, map[string]string{"model.gguf": hfModelBody})
	f.mgr.opts.Config.Runtime.VerifyOnDrift = config.VerifyOnDriftFollow
	up.commit.Store("ffffffffffffffffffffffffffffffffffffffff")

	f.ensureRemote(t)
	if got := f.spawns.Load(); got != 1 {
		t.Errorf("hf spawns = %d, want a commit-only change ignored", got)
	}
}

func TestDriftHF_ChangedFileFollows(t *testing.T) {
	t.Parallel()
	f, up := remoteHFFixture(t, map[string]string{"model.gguf": hfModelBody})
	f.mgr.opts.Config.Runtime.VerifyOnDrift = config.VerifyOnDriftFollow
	up.mu.Store(map[string]string{"model.gguf": "retrained weights"})

	f.ensureRemote(t)
	if got := f.spawns.Load(); got != 2 {
		t.Errorf("hf spawns = %d, want the changed weights fetched", got)
	}
}

func TestDriftHF_ChangedFileIsOnlyReportedByDefault(t *testing.T) {
	t.Parallel()
	f, up := remoteHFFixture(t, map[string]string{"model.gguf": hfModelBody})
	up.mu.Store(map[string]string{"model.gguf": "retrained weights"})

	f.ensureRemote(t)
	if got := f.spawns.Load(); got != 1 {
		t.Errorf("hf spawns = %d, want report to leave the snapshot alone", got)
	}
	if n := f.mgr.opts.Manager.Snapshot().Note; !strings.Contains(n, "VERIFY_ON_DRIFT") {
		t.Errorf("note = %q, want it to name the setting that would follow the change", n)
	}
}

// A Hub that cannot be reached leaves the model running and says the
// check did not happen.
func TestDriftHF_UnreachableHubSaysSo(t *testing.T) {
	t.Parallel()
	f, up := remoteHFFixture(t, map[string]string{"model.gguf": hfModelBody})
	f.mgr.opts.Config.Runtime.VerifyOnDrift = config.VerifyOnDriftFollow
	up.Close()

	f.ensureRemote(t)
	if got := f.spawns.Load(); got != 1 {
		t.Errorf("hf spawns = %d, want the verified snapshot kept", got)
	}
	if n := f.mgr.opts.Manager.Snapshot().Note; n == "" {
		t.Error("note is empty, want the unanswered check reported")
	}
}

// RepoTree reads the recursive tree so a file in a subdirectory is
// compared rather than silently reported as gone upstream.
func TestRepoTree_ReadsNestedEntries(t *testing.T) {
	t.Parallel()
	var gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.RawQuery
		w.Header().Set("X-Repo-Commit", hfCommit)
		_, _ = w.Write([]byte(`[
			{"path":"onnx/model.onnx","size":12,"type":"file","lfs":{"oid":"abc"}},
			{"path":"onnx","size":0,"type":"directory"},
			{"path":"config.json","size":2,"type":"file"}
		]`))
	}))
	t.Cleanup(srv.Close)

	entries, commit, err := hfwrap.RepoTree(context.Background(),
		hfwrap.Config{Repo: hfRepo, Revision: "main", Endpoint: srv.URL})
	if err != nil {
		t.Fatalf("RepoTree: %v", err)
	}
	if !strings.Contains(gotQuery, "recursive=true") {
		t.Errorf("query = %q, want the recursive tree", gotQuery)
	}
	if commit != hfCommit {
		t.Errorf("commit = %q, want %q", commit, hfCommit)
	}
	if len(entries) != 2 {
		t.Fatalf("entries = %+v, want the two files without the directory", entries)
	}
	byPath := map[string]hfwrap.TreeEntry{}
	for _, e := range entries {
		byPath[e.Path] = e
	}
	if got := byPath["onnx/model.onnx"]; got.SHA256 != "abc" {
		t.Errorf("nested LFS entry = %+v, want its oid carried through", got)
	}
	if got := byPath["config.json"]; got.SHA256 != "" || got.Size != 2 {
		t.Errorf("non-LFS entry = %+v, want a size and no digest", got)
	}
}

// A model is several sources and the banner is one string. The verdict
// an operator asked for must survive a later source having nothing to
// say, and must disappear when that same source is checked again and is
// clean.
func TestDriftNote_OneSourcesSilenceKeepsAnothersFinding(t *testing.T) {
	t.Parallel()
	m := &Manager{opts: Options{Manager: progress.New(time.Now())}, pass: &passState{}}
	cfg := config.Config{}

	m.resolveDrift(cfg, "weights", driftFound, "ETag moved")
	m.resolveDrift(cfg, "projector", driftNone, "")

	got := m.opts.Manager.Snapshot().Note
	if !strings.Contains(got, "weights") {
		t.Errorf("note = %q, want the drifted source still reported", got)
	}
	if strings.Contains(got, "projector") {
		t.Errorf("note = %q, want nothing about the clean source", got)
	}

	m.resolveDrift(cfg, "weights", driftNone, "")
	if got := m.opts.Manager.Snapshot().Note; got != "" {
		t.Errorf("note = %q, want it retracted once every source is clean", got)
	}
}

// Retracting is not the same as clearing. The download path writes the
// banner too, and a check that finds nothing has no business removing
// somebody else's message.
func TestDriftNote_LeavesAnotherWritersBannerAlone(t *testing.T) {
	t.Parallel()
	m := &Manager{opts: Options{Manager: progress.New(time.Now())}, pass: &passState{}}

	m.resolveDrift(config.Config{}, "weights", driftFound, "ETag moved")
	m.note("下载中：xet 加速已启用")
	m.resolveDrift(config.Config{}, "weights", driftNone, "")

	if got := m.opts.Manager.Snapshot().Note; got != "下载中：xet 加速已启用" {
		t.Errorf("note = %q, want the other writer's message untouched", got)
	}
}

// The Hub caps a tree page and offers the rest behind a Link header.
// Reading only the first page would have the drift check report every
// file it could not see as deleted upstream, which is the one verdict
// that re-downloads a whole model under VERIFY_ON_DRIFT=follow.
func TestRepoTree_FollowsPagination(t *testing.T) {
	t.Parallel()
	var pages atomic.Int32
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Repo-Commit", hfCommit)
		if r.URL.Query().Get("cursor") == "" {
			pages.Add(1)
			w.Header().Set("Link", `<`+srv.URL+`/api/models/`+hfRepo+`/tree/main?recursive=true&cursor=p2>; rel="next"`)
			_, _ = w.Write([]byte(`[{"path":"a.gguf","size":1,"type":"file","lfs":{"oid":"aa"}}]`))
			return
		}
		pages.Add(1)
		_, _ = w.Write([]byte(`[{"path":"b.gguf","size":2,"type":"file","lfs":{"oid":"bb"}}]`))
	}))
	t.Cleanup(srv.Close)

	entries, commit, err := hfwrap.RepoTree(context.Background(),
		hfwrap.Config{Repo: hfRepo, Revision: "main", Endpoint: srv.URL})
	if err != nil {
		t.Fatalf("RepoTree: %v", err)
	}
	if got := pages.Load(); got != 2 {
		t.Errorf("pages fetched = %d, want both", got)
	}
	if commit != hfCommit {
		t.Errorf("commit = %q, want the first page's %q", commit, hfCommit)
	}
	if len(entries) != 2 {
		t.Fatalf("entries = %+v, want one from each page", entries)
	}
	if entries[0].Path != "a.gguf" || entries[1].Path != "b.gguf" {
		t.Errorf("entries = %+v, want them in page order", entries)
	}
}

// A record with no files is not "nothing to compare, so everything
// matches". Anything that cannot be compared has to read as unknown.
func TestCheckDrift_EmptyRecordIsUnknown(t *testing.T) {
	t.Parallel()
	var empty verify.Sidecar
	if v, _ := checkURLDrift(context.Background(), config.ModelSource{}, empty); v != driftUnknown {
		t.Errorf("url verdict = %v, want driftUnknown", v)
	}
	if v, _ := checkHFDrift(context.Background(), config.Config{}, config.ModelSource{}, empty); v != driftUnknown {
		t.Errorf("hf verdict = %v, want driftUnknown", v)
	}
}
