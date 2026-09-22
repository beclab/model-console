package lifecycle

import (
	"context"
	"errors"
	"log/slog"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/llm-init/llm-init/internal/adapter/hfwrap"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/fetch"
	"github.com/llm-init/llm-init/internal/verify"
)

// urlSidecarSuffix names the record beside a single downloaded file.
const urlSidecarSuffix = ".llm-init.json"

// passLevel resolves how hard this pass should check bytes already on
// disk. VERIFY_LEVEL is the standing default; `?level=` on /api/retry
// overrides it for one pass and is the only way to reach `remote`, since
// a standing default that touches the network would give up booting
// offline.
//
// An unrecognised `?level=` falls back to the configured default rather
// than failing the pass: the HTTP layer deliberately stopped rejecting
// unknown values so old dashboards keep working, and turning one into a
// failed ensure here would undo that.
func passLevel(cfg config.Config, opts RetryOptions) verify.Level {
	switch verify.Level(opts.Level) {
	case verify.LevelSize, verify.LevelSHA256, verify.LevelRemote:
		return verify.Level(opts.Level)
	}
	if l := verify.Level(cfg.Runtime.VerifyLevel); l == verify.LevelSHA256 {
		return l
	}
	return verify.LevelSize
}

// registerWithKnownDigests hands the installer dest along with the
// digest the download record already holds for it.
//
// Ollama's Register hashes every file it is given, which on this path is
// a second full read of a model whose digest is written down beside it:
// an ollama:// URL with a #sha256= fragment has had exactly those bytes
// verified by the downloader. On a multi-gigabyte GGUF that read is
// minutes of every boot, spent to arrive at a string already on disk.
//
// A missing record, or one without a digest, degrades to hashing.
func (m *Manager) registerWithKnownDigests(ctx context.Context, dest string) error {
	known := map[string]string{}
	if sc, err := verify.Read(dest + urlSidecarSuffix); err == nil {
		for _, f := range sc.Files {
			if f.SHA256 != "" && filepath.Base(f.Path) == filepath.Base(dest) {
				known[dest] = "sha256:" + f.SHA256
				break
			}
		}
	}
	return m.installer.Register(ctx, []string{dest}, known, m.opts.Manager.Sink())
}

// urlSourceIdentity describes a URL-backed source well enough to tell
// "the same download as last time" from "something else now".
func urlSourceIdentity(ms config.ModelSource) verify.Source {
	kind := verify.KindURL
	if ms.Kind == config.KindOllamaURL {
		kind = verify.KindOllamaURL
	}
	return verify.Source{
		Kind:           kind,
		URLSHA256:      verify.URLIdentity(ms.URL),
		URLRedacted:    recordableURL(ms),
		DeclaredSHA256: ms.URLSha256,
	}
}

// recordableURL is the most of a source URL that may be written to a
// file on a shared volume: scheme, host and path, with the userinfo,
// the query and the fragment dropped.
//
// config.RedactedSource is not that. It removes userinfo, which is what
// a log line needs, and keeps the query verbatim -- and the query is
// where a presigned URL carries its signature, which is a credential
// good until it expires. It reaches the operator's terminal; this
// reaches a JSON file that other applications mounting the same cache
// can read.
//
// A URL that will not parse records as nothing rather than as itself.
func recordableURL(ms config.ModelSource) string {
	u, err := url.Parse(ms.URL)
	if err != nil {
		return ""
	}
	u.User = nil
	u.RawQuery = ""
	u.ForceQuery = false
	u.Fragment = ""
	u.RawFragment = ""
	return u.String()
}

// reuseURLDownload reports whether dest already holds the bytes ms asks
// for, so the pass can skip the downloader entirely -- not even a HEAD.
// That is the point: a machine with no route to the upstream still boots
// a model it already has.
//
// Anything other than a clean hit returns false and says why, since the
// interesting cases are indistinguishable in the result: never
// downloaded here, downloaded from a different source, or downloaded and
// since damaged.
func (m *Manager) reuseURLDownload(ctx context.Context, cfg config.Config, ms config.ModelSource, dest string, level verify.Level) bool {
	path := dest + urlSidecarSuffix
	sc, err := verify.Read(path)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			slog.Info("download record unusable, downloading",
				slog.String("path", path), slog.String("err", err.Error()))
		}
		return false
	}
	want := urlSourceIdentity(ms)
	if !sc.Source.Equal(want) {
		slog.Info("download record describes a different source, downloading",
			slog.String("recorded", sc.Source.URLRedacted),
			slog.String("configured", want.URLRedacted))
		return false
	}
	rep, err := sc.CheckLocal(ctx, m.opts.Downloader, filepath.Dir(dest), level)
	if err != nil {
		slog.Warn("local model bytes failed verification, downloading again",
			slog.String("dest", dest),
			slog.String("level", string(level)),
			slog.String("err", err.Error()))
		return false
	}
	slog.Info("reusing verified local model bytes",
		slog.String("dest", dest),
		slog.String("level", string(level)),
		slog.Int("files", rep.Files),
		slog.Int("digest_checked", rep.Digest))

	if level == verify.LevelRemote {
		v, detail := checkURLDrift(ctx, ms, sc)
		return m.resolveDrift(cfg, sc.Source.URLRedacted, v, detail)
	}
	return true
}

// writeURLSidecar records what a completed download produced.
//
// The digest is free when MODEL_SOURCE declared one: downloadURL has
// already compared the file against it, so the file's sha256 is that
// value and re-reading the bytes would prove nothing new. Without a
// declared digest one is computed only when the level asked for digests,
// because hashing a multi-gigabyte model that nobody will check back is
// a full read for nothing.
//
// A failure here is logged and swallowed. The bytes are on disk and the
// engine can load them; the only cost of a missing record is that the
// next boot downloads again.
func (m *Manager) writeURLSidecar(ctx context.Context, ms config.ModelSource, dest string, level verify.Level, etag string) {
	info, err := os.Stat(dest)
	if err != nil {
		slog.Warn("skipping download record", slog.String("dest", dest), slog.String("err", err.Error()))
		return
	}

	sha := ms.URLSha256
	if sha == "" && level.HashesLocally() {
		if h, herr := fetch.HashFile(ctx, dest); herr == nil {
			sha = h
		} else {
			slog.Warn("could not hash the download for its record",
				slog.String("dest", dest), slog.String("err", herr.Error()))
		}
	}

	sc := verify.Sidecar{
		Source: urlSourceIdentity(ms),
		Files: []verify.File{{
			Path:   filepath.Base(dest),
			Size:   info.Size(),
			SHA256: sha,
			ETag:   etag,
		}},
	}
	if err := verify.Write(dest+urlSidecarSuffix, sc); err != nil {
		slog.Warn("could not write the download record",
			slog.String("dest", dest), slog.String("err", err.Error()))
	}
}

// removeURLSidecar drops the record before bytes it describes are
// replaced, so a download interrupted halfway cannot leave a record
// claiming the old file is still intact.
func removeURLSidecar(dest string) {
	if err := verify.Remove(dest + urlSidecarSuffix); err != nil {
		slog.Warn("could not remove the stale download record", slog.String("err", err.Error()))
	}
}

// hfSidecarDir is the per-repo directory holding llm-init's records. It
// sits inside the repo's cache directory rather than beside it so that
// deleting a repo from the cache deletes its records with it.
//
// It is created 0755 by whichever application downloads the repo first.
// Applications sharing an HF cache under different uids can therefore
// read each other's records but not add their own, and the second one
// re-downloads on every boot with "could not write the hf download
// record" in its log. Loosening the mode is not the fix -- the blobs
// beside it have the same constraint, so a cache shared across uids is
// already broken -- but that log line is the symptom to recognise.
const hfSidecarDir = ".llm-init"

// hfSidecarPath names the record for one HF request against one
// snapshot. Both halves of the name are needed: a repo directory holds
// a snapshot per revision, and one snapshot serves several requests --
// a main GGUF and its mmproj sibling are two MODEL_SOURCE segments
// against the same repo and revision that differ only in --include, and
// filing both under the commit alone would have each overwrite the
// other and re-download on every boot.
func hfSidecarPath(cacheRoot, repo, sha string, src verify.Source) string {
	return filepath.Join(hfwrap.HFRepoCacheDir(cacheRoot, repo), hfSidecarDir,
		sha+"-"+src.Fingerprint()+".json")
}

// removeHFSidecars drops every record this source wrote, across all
// snapshots of the repo, before --force-download replaces the bytes
// they describe. Only this source's records go: a sibling MODEL_SOURCE
// segment against the same repo writes its own, and clearing the whole
// directory would delete the record the previous segment of this very
// pass just wrote.
func removeHFSidecars(cacheRoot, repo string, src verify.Source) {
	dir := filepath.Join(hfwrap.HFRepoCacheDir(cacheRoot, repo), hfSidecarDir)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	suffix := "-" + src.Fingerprint() + ".json"
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), suffix) {
			continue
		}
		if err := verify.Remove(filepath.Join(dir, e.Name())); err != nil {
			slog.Warn("could not remove the stale hf download record",
				slog.String("name", e.Name()), slog.String("err", err.Error()))
		}
	}
}

// hfSourceIdentity describes an HF request well enough to tell it from
// any other request against the same cache.
func hfSourceIdentity(cfg config.Config, ms config.ModelSource) verify.Source {
	return verify.Source{
		Kind:     verify.KindHF,
		Repo:     ms.HFRepo,
		Revision: ms.HFRevision,
		Endpoint: cfg.HFEndpoint,
		Include:  ms.HFInclude,
		Exclude:  ms.HFExclude,
		Subdir:   ms.HFSubdir,
	}
}

// reuseHFDownload reports whether the HF cache already holds what ms
// asks for, in which case the `hf` CLI is never spawned. The returned
// Result stands in for the one that subprocess would have produced.
//
// Skipping it is the whole point. `hf download` against a fully-cached
// snapshot is not free: it reaches the Hub for a metadata diff, so a
// boot with no route to huggingface.co fails on a model that is sitting
// complete on the volume.
//
// Which revision is on disk comes from the cache's own refs file, so a
// source tracking a branch resolves to whatever commit was last fetched
// -- not to whatever the branch points at today. Noticing that the
// branch has moved is the remote check's job, and deliberately not
// something an ordinary boot does.
func (m *Manager) reuseHFDownload(ctx context.Context, cfg config.Config, ms config.ModelSource, level verify.Level) (hfwrap.Result, bool) {
	root := m.hfCacheRoot()
	sha, err := hfwrap.ReadRefCommit(root, ms.HFRepo, ms.HFRevision)
	if err != nil || sha == "" {
		return hfwrap.Result{}, false
	}
	want := hfSourceIdentity(cfg, ms)
	path := hfSidecarPath(root, ms.HFRepo, sha, want)
	sc, err := verify.Read(path)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			slog.Info("hf download record unusable, downloading",
				slog.String("path", path), slog.String("err", err.Error()))
		}
		return hfwrap.Result{}, false
	}
	if !sc.Source.Equal(want) {
		// The fingerprint is in the filename, so this is a collision
		// rather than a changed request, and worth saying out loud.
		slog.Info("hf download record describes a different request, downloading",
			slog.String("repo", ms.HFRepo))
		return hfwrap.Result{}, false
	}

	snapshot := hfwrap.SnapshotPath(root, ms.HFRepo, sha)
	rep, err := sc.CheckLocal(ctx, m.opts.Downloader, snapshot, level)
	if err != nil {
		slog.Warn("cached hf snapshot failed verification, downloading again",
			slog.String("repo", ms.HFRepo),
			slog.String("commit", sha),
			slog.String("level", string(level)),
			slog.String("err", err.Error()))
		return hfwrap.Result{}, false
	}
	slog.Info("reusing verified hf snapshot",
		slog.String("repo", ms.HFRepo),
		slog.String("commit", sha),
		slog.String("level", string(level)),
		slog.Int("files", rep.Files),
		slog.Int("digest_checked", rep.Digest))

	if level == verify.LevelRemote {
		v, detail := checkHFDrift(ctx, cfg, ms, sc)
		if !m.resolveDrift(cfg, ms.HFRepo, v, detail) {
			return hfwrap.Result{}, false
		}
	}

	enginePath := snapshot
	if sc.EnginePath != "" {
		enginePath = filepath.Join(snapshot, filepath.FromSlash(sc.EnginePath))
	}
	return hfwrap.Result{Path: enginePath, Commit: sha, Status: "cached"}, true
}

// writeHFSidecar records the snapshot a completed `hf download`
// produced. Failure is logged and swallowed: the bytes are usable, and
// the only cost of a missing record is downloading again next boot.
func (m *Manager) writeHFSidecar(cfg config.Config, ms config.ModelSource, res hfwrap.Result) {
	root := m.hfCacheRoot()
	sha := res.Commit
	if !isResolvedCommit(sha) {
		var err error
		if sha, err = hfwrap.ReadRefCommit(root, ms.HFRepo, ms.HFRevision); err != nil || sha == "" {
			slog.Info("skipping hf download record: no resolved commit",
				slog.String("repo", ms.HFRepo))
			return
		}
	}
	snapshot := hfwrap.SnapshotPath(root, ms.HFRepo, sha)
	enginePath := relativeEnginePath(snapshot, res.Path)
	files, err := scanHFSnapshot(snapshot, ms, enginePath)
	if err != nil || len(files) == 0 {
		slog.Info("skipping hf download record: nothing found in the snapshot",
			slog.String("snapshot", snapshot))
		return
	}

	src := hfSourceIdentity(cfg, ms)
	// Commit is not part of the identity -- a moved branch still names
	// the same configured source -- but recording which commit these
	// bytes came from is what lets a remote check say what moved.
	src.Commit = sha
	sc := verify.Sidecar{Source: src, Files: files, EnginePath: enginePath}
	if err := verify.Write(hfSidecarPath(root, ms.HFRepo, sha, src), sc); err != nil {
		slog.Warn("could not write the hf download record",
			slog.String("repo", ms.HFRepo), slog.String("err", err.Error()))
	}
}

// scanHFSnapshot lists what this request put in the snapshot directory,
// taking each file's digest from the cache layout rather than by
// reading it.
//
// A snapshot entry is a symlink into blobs/, named after the ETag the
// Hub served. For an LFS file -- every weight file worth verifying --
// that ETag is the sha256 of the content, so the digest of a
// multi-gigabyte model costs one readlink. Small files are stored under
// their git object id instead, which is a sha1 of different bytes and
// no use as a content digest, so they are recorded with a size only.
//
// The rule is "a 64-character hex blob name is that file's sha256". It
// is a guess about somebody else's cache layout, and the failure mode
// if it is ever wrong is a mismatch at the digest level: a wasted
// download rather than bytes served unchecked.
//
// A snapshot is shared. One directory holds whatever every MODEL_SOURCE
// against that repo and revision has fetched, so recording all of it
// would put the main GGUF in the projector's record as well as its own
// -- and under VERIFY_LEVEL=sha256 that is the same twenty gigabytes
// hashed once per source, every boot. The scan is therefore filtered by
// the request's own --include and --exclude. A filter that matches
// nothing falls back to the whole directory: over-describing costs
// time, and an empty record costs a re-download on every boot.
func scanHFSnapshot(snapshot string, ms config.ModelSource, enginePath string) ([]verify.File, error) {
	var all []verify.File
	err := filepath.Walk(snapshot, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		rel, rerr := filepath.Rel(snapshot, p)
		if rerr != nil {
			return nil
		}
		st, serr := os.Stat(p)
		if serr != nil {
			return nil
		}
		f := verify.File{Path: filepath.ToSlash(rel), Size: st.Size()}
		if target, lerr := os.Readlink(p); lerr == nil {
			if base := filepath.Base(target); isSHA256Hex(base) {
				f.SHA256 = base
			}
		}
		all = append(all, f)
		return nil
	})
	if err != nil {
		return nil, err
	}

	var mine []verify.File
	for _, f := range all {
		if f.Path == enginePath || matchesHFPatterns(f.Path, ms.HFInclude, ms.HFExclude) {
			mine = append(mine, f)
		}
	}
	if len(mine) == 0 {
		return all, nil
	}
	return mine, nil
}

// matchesHFPatterns applies one request's --include and --exclude to a
// snapshot-relative path. The matching rules live in hfwrap so the
// download budget and the sidecar record the same set of files.
func matchesHFPatterns(rel string, include, exclude []string) bool {
	return hfwrap.MatchPatterns(rel, include, exclude)
}

// relativeEnginePath expresses what hfwrap resolved relative to the
// snapshot root. An empty or non-descendant path records as empty,
// meaning "the snapshot directory itself".
func relativeEnginePath(snapshot, resolved string) string {
	if resolved == "" || resolved == snapshot {
		return ""
	}
	rel, err := filepath.Rel(snapshot, resolved)
	if err != nil || strings.HasPrefix(rel, "..") {
		return ""
	}
	return filepath.ToSlash(rel)
}

// isSHA256Hex reports whether s is 64 lowercase hex characters.
func isSHA256Hex(s string) bool { return isLowerHex(s, 64) }

// isResolvedCommit reports whether s is a 40-character lowercase hex
// commit, the only form huggingface_hub writes.
func isResolvedCommit(s string) bool { return isLowerHex(s, 40) }

func isLowerHex(s string, n int) bool {
	if len(s) != n {
		return false
	}
	for _, r := range s {
		switch {
		case r >= '0' && r <= '9':
		case r >= 'a' && r <= 'f':
		default:
			return false
		}
	}
	return true
}
