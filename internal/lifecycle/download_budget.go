package lifecycle

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/llm-init/llm-init/internal/adapter/hfwrap"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/fetch"
	"github.com/llm-init/llm-init/internal/progress"
)

// passState is the bookkeeping for one ensure pass.
//
// It lives on the Manager rather than being threaded through six
// signatures because ensure runs on a single goroutine: Run drives it,
// and every other trigger -- /api/retry, the health loop's self-heal --
// wakes that loop rather than calling ensure itself.
//
// It outlives its pass. ensure replaces it wholesale on the way in, so
// nothing stale is ever read within a pass, and leaving the last one in
// place is what lets a test ask afterwards what the pass did. Nothing
// outside ensure may read it: between passes its cfg is a snapshot of a
// configuration that may since have been edited.
type passState struct {
	cfg     config.Config
	main    *config.ModelSource
	mmproj  *config.ModelSource
	extras  []config.ModelSource
	force   bool
	entered bool
	// hfTrees is the Hub listing each hf:// source resolved to while
	// the budget was estimated, keyed by MODEL_SOURCE segment index.
	// runHF hands it to hfwrap so a tqdm bar can be named after the
	// repository file rather than after the file name it shares with
	// its namesakes in other directories.
	hfTrees map[int][]hfwrap.TreeEntry
	// hfCommits is the commit each hf:// source's revision resolved to
	// while the budget was estimated, by segment index. runHF downloads
	// at it, so the progress seeded from that commit's listing describes
	// the bytes hf fetches even if the branch moves in between.
	hfCommits map[int]string
	// driftNotes is one remote verdict per source, and driftNote is
	// what they last rendered to. See Manager.setDriftNote.
	driftNotes map[string]string
	driftNote  string
}

// hfTree returns the listing recorded for one source, or nil when the
// budget could not be estimated and no tree was fetched.
func (p *passState) hfTree(ms config.ModelSource) []hfwrap.TreeEntry {
	if p == nil {
		return nil
	}
	return p.hfTrees[ms.Index]
}

// hfCommit returns the commit recorded for one source, or "" when its
// revision was not resolved.
func (p *passState) hfCommit(ms config.ModelSource) string {
	if p == nil {
		return ""
	}
	return p.hfCommits[ms.Index]
}

// enterDownloadPhase moves the pass into phase=download and pins the
// byte budget, and does it the first time this pass finds it actually
// has something to fetch rather than on the way in.
//
// Entering is not free. setPhase(PhaseDownload) answers every /v1/*
// request with 503 for the length of the pass, and the budget estimate
// that follows asks the Hub about every repo and HEADs every URL. Once
// the download records exist, most passes have nothing to fetch --
// every boot on a warm volume, and every self-heal the health loop
// triggers on a model that is serving fine -- and paying both costs to
// discover that took a healthy model offline and made it depend on an
// upstream it did not need.
//
// When the aggregate size is known for every file in the pass it also
// pins bytes_total and any already-on-disk progress. Seeding
// bytes_completed matters on retry: setPhase(download) zeroes the
// gauges, and fetch/hfwrap OnBytes only report *new* transfer (URL
// Range resume) or absolute tqdm deltas from a fresh Aggregator, so
// without a seed a mid-file retry under-reports until the reconcile at
// the end.
func (m *Manager) enterDownloadPhase(ctx context.Context) {
	p := m.pass
	if p == nil || p.entered {
		return
	}
	p.entered = true
	m.setPhase(progress.PhaseDownload)
	budgetCtx, cancel := ctx, context.CancelFunc(func() {})
	if m.opts.MetadataBudget > 0 {
		budgetCtx, cancel = context.WithTimeout(ctx, m.opts.MetadataBudget)
	}
	b := m.estimateDownloadBudget(budgetCtx, p.cfg, p.main, p.mmproj, p.extras)
	if errors.Is(budgetCtx.Err(), context.DeadlineExceeded) && ctx.Err() == nil {
		slog.Warn("download metadata lookups ran past their budget; downloading with what they found",
			slog.Duration("budget", m.opts.MetadataBudget))
	}
	cancel()
	p.hfTrees = b.trees
	p.hfCommits = b.commits
	if b.canPin && b.bytes > 0 {
		m.opts.Manager.SeedDownloadBudget(b.bytes, seedCompletedForPass(p.force, b.bytesCached))
		if queue := fileListForPass(p.force, b.queue); len(queue) > 0 {
			m.opts.Manager.SeedDownloadFileList(queue)
		} else if b.files > 0 {
			m.opts.Manager.SeedDownloadFiles(b.files, seedCompletedFiles(p.force, b.filesCached))
		}
	}
}

func fileListForPass(force bool, queue []progress.FileProgress) []progress.FileProgress {
	if len(queue) == 0 {
		return nil
	}
	out := append([]progress.FileProgress(nil), queue...)
	if !force {
		return out
	}
	for i := range out {
		out[i].Status = progress.FileWaiting
		out[i].BytesCompleted = 0
	}
	return out
}

func seedCompletedFiles(force bool, cached int) int {
	if force {
		return 0
	}
	return cached
}

// seedCompletedForPass drops on-disk resume bytes when Force will wipe
// them after the seed (URL .part / finished dest). Otherwise the UI
// would show a high bytes_completed while the re-fetch starts at 0.
func seedCompletedForPass(force bool, cached int64) int64 {
	if force {
		return 0
	}
	return cached
}

// downloadBudget is the pinned denominator for one ensure pass.
type downloadBudget struct {
	bytes, bytesCached int64
	files, filesCached int
	canPin             bool
	queue              []progress.FileProgress
	// trees is the Hub listing per hf:// source, by segment index. It
	// survives the early returns that give up on pinning a denominator:
	// naming a tqdm bar needs the listing, not a complete estimate.
	trees map[int][]hfwrap.TreeEntry
	// commits is the resolved commit per hf:// source, by segment index.
	commits map[int]string
}

// estimateDownloadBudget sums what this pass will fetch across every
// source it drives: the extras, the mmproj sibling, and the main source.
// A single unestimable source gives up the whole pin (canPin=false) rather
// than pinning a total the stream would then overshoot.
func (m *Manager) estimateDownloadBudget(ctx context.Context, cfg config.Config, main, mmproj *config.ModelSource, extras []config.ModelSource) downloadBudget {
	var urlList []string
	var urlQueue []progress.FileProgress
	var hfSources []config.ModelSource
	var out downloadBudget

	sources := append([]config.ModelSource(nil), extras...)
	if mmproj != nil {
		sources = append(sources, *mmproj)
	}
	if main != nil {
		sources = append(sources, *main)
	}
	// Resolved ahead of everything that can give up on the estimate:
	// runHF needs the commit whether or not a denominator gets pinned.
	targets := m.resolveHFTargets(ctx, cfg, sources)
	out.commits = commitsOf(targets)
	for i := range sources {
		if sources[i].Kind == config.KindOllama {
			return downloadBudget{commits: out.commits}
		}
		if u := urlForSource(sources[i]); u != "" {
			urlList = append(urlList, u)
			fp := progress.FileProgress{Path: urlDisplayName(sources[i]), Status: progress.FileWaiting}
			n := cachedURLResumeBytes(cfg, sources[i])
			out.bytesCached += n
			if finishedURLFile(cfg, sources[i]) {
				out.filesCached++
				fp.Status = progress.FileDone
				fp.BytesCompleted = n
			} else if n > 0 {
				fp.BytesCompleted = n
			}
			urlQueue = append(urlQueue, fp)
		}
		if sources[i].Kind == config.KindHF {
			hfSources = append(hfSources, sources[i])
		}
	}

	if len(urlList) > 0 {
		if sum, complete := estimateHFResolveURLList(ctx, cfg, urlList); complete {
			out.bytes += sum
			out.files += len(urlList)
			out.queue = append(out.queue, urlQueue...)
		} else if sum, ok, allKnown := fetch.SumProbeContentLength(ctx, nil, urlList); ok && allKnown {
			out.bytes += sum
			out.files += len(urlList)
			if len(urlQueue) == 1 {
				urlQueue[0].BytesTotal = sum
			}
			out.queue = append(out.queue, urlQueue...)
		} else {
			return out.givenUp()
		}
	}

	for i := range hfSources {
		est, err := estimateHFTarget(ctx, cfg, hfSources[i], targets[hfSources[i].Index])
		if len(est.Entries) > 0 {
			if out.trees == nil {
				out.trees = make(map[int][]hfwrap.TreeEntry, len(hfSources))
			}
			out.trees[hfSources[i].Index] = est.Entries
		}
		if err != nil || !est.Complete || est.Bytes <= 0 {
			return out.givenUp()
		}
		out.bytes += est.Bytes
		out.files += est.Files
		for _, e := range est.Entries {
			fp := progress.FileProgress{Path: e.Path, BytesTotal: e.Size, Status: progress.FileWaiting}
			if hasHFFile(m.hfCacheRoot(), hfSources[i].HFRepo, est.Commit, e) {
				fp.Status = progress.FileDone
				fp.BytesCompleted = e.Size
				out.bytesCached += e.Size
				out.filesCached++
			}
			out.queue = append(out.queue, fp)
		}
	}

	out.canPin = out.bytes > 0
	return out
}

// givenUp is the budget to return when one source could not be sized:
// no denominator to pin, but the cached counts and whatever listings
// were fetched before giving up are still worth carrying out.
func (b downloadBudget) givenUp() downloadBudget {
	return downloadBudget{
		bytesCached: b.bytesCached,
		filesCached: b.filesCached,
		trees:       b.trees,
		commits:     b.commits,
	}
}

// hfTarget is what the budget learned about one hf:// source before
// sizing it: the commit its revision resolved to ("" when it did not),
// and for a source naming one file, that file as the HEAD described it.
type hfTarget struct {
	commit string
	entry  *hfwrap.TreeEntry
}

func commitsOf(targets map[int]hfTarget) map[int]string {
	var commits map[int]string
	for idx, t := range targets {
		if t.commit == "" {
			continue
		}
		if commits == nil {
			commits = make(map[int]string, len(targets))
		}
		commits[idx] = t.commit
	}
	return commits
}

// resolveHFTargets resolves every hf:// source, by segment index.
//
// A repo and revision is resolved once per pass. A main GGUF and its
// projector usually share one, and the second source then only needs
// its size, asked at the commit the first resolved to so both files
// come from one revision. A revision that could not be resolved is not
// tried again for the next source: the remote just failed every attempt.
func (m *Manager) resolveHFTargets(ctx context.Context, cfg config.Config, sources []config.ModelSource) map[int]hfTarget {
	targets := make(map[int]hfTarget)
	resolved := make(map[string]string)
	for i := range sources {
		src := sources[i]
		if src.Kind != config.KindHF {
			continue
		}
		key := src.HFRepo + "@" + src.HFRevision
		commit, done := resolved[key]
		if !done {
			targets[src.Index] = m.resolveHFTarget(ctx, cfg, src)
			resolved[key] = targets[src.Index].commit
			continue
		}
		t := hfTarget{commit: commit}
		if file := singleHFFile(src); file != "" && commit != "" {
			if h, err := hfwrap.HeadFile(ctx, hfLookupConfig(cfg, src, commit, file)); err == nil && h.Commit == commit {
				t.entry = sizedEntry(h.Entry)
			}
		}
		targets[src.Index] = t
	}
	return targets
}

// hfResolveAttempts is how many lookups a revision gets before the
// remote counts as unreachable for this pass.
const hfResolveAttempts = 3

// hfHeadGrace is how long an attempt that already has the commit from
// the revision API still waits for the HEAD, whose answer also sizes
// the file.
const hfHeadGrace = 3 * time.Second

// resolveHFTarget looks up the commit a source's revision names now.
// A target without a commit means every attempt failed: runHF then
// downloads at the revision as given, and hf falls back to the snapshot
// refs/<revision> points at if it cannot reach the remote either.
func (m *Manager) resolveHFTarget(ctx context.Context, cfg config.Config, source config.ModelSource) hfTarget {
	file := singleHFFile(source)
	wait := m.opts.HFResolveBackoff
	for attempt := 1; ; attempt++ {
		t, err := lookupHFTarget(ctx, cfg, source, file)
		if err == nil {
			return t
		}
		if attempt == hfResolveAttempts {
			slog.Warn("remote unreachable: hf revision not resolved; a cached snapshot may be used instead",
				slog.String("repo", source.HFRepo),
				slog.String("revision", source.HFRevision),
				slog.Int("attempts", attempt),
				slog.String("err", err.Error()))
			return hfTarget{}
		}
		slog.Warn("hf revision lookup failed, retrying",
			slog.String("repo", source.HFRepo),
			slog.String("revision", source.HFRevision),
			slog.Int("attempt", attempt),
			slog.Int("attempts", hfResolveAttempts),
			slog.String("err", err.Error()))
		select {
		case <-ctx.Done():
			return hfTarget{}
		case <-time.After(wait):
		}
		wait *= 2
	}
}

// lookupHFTarget is one attempt. It asks the revision API and, when the
// source names one file, HEADs that file, both at once: asked one after
// the other, an unreachable remote cost two timeouts per attempt.
func lookupHFTarget(ctx context.Context, cfg config.Config, source config.ModelSource, file string) (hfTarget, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	type answer struct {
		target hfTarget
		err    error
		head   bool
	}
	answers := make(chan answer, 2)
	pending := 1
	go func() {
		c, err := hfwrap.ResolveCommit(ctx, hfLookupConfig(cfg, source, source.HFRevision, ""))
		answers <- answer{target: hfTarget{commit: c}, err: err}
	}()
	if file != "" {
		pending++
		go func() {
			h, err := hfwrap.HeadFile(ctx, hfLookupConfig(cfg, source, source.HFRevision, file))
			answers <- answer{target: hfTarget{commit: h.Commit, entry: sizedEntry(h.Entry)}, err: err, head: true}
		}()
	}

	var fromAPI *hfTarget
	var errs []error
	var grace <-chan time.Time
	for pending > 0 {
		select {
		case a := <-answers:
			pending--
			switch {
			case a.err != nil:
				errs = append(errs, a.err)
			case a.head || pending == 0:
				return a.target, nil
			default:
				fromAPI = &a.target
				grace = time.After(hfHeadGrace)
			}
		case <-grace:
			return *fromAPI, nil
		}
	}
	if fromAPI != nil {
		return *fromAPI, nil
	}
	return hfTarget{}, errors.Join(errs...)
}

// singleHFFile is the one file a source downloads by name, or "" when
// a HEAD on it would not describe everything the source fetches: a
// pattern, a --subdir the tree walk filters by, or an --exclude that
// drops it.
func singleHFFile(ms config.ModelSource) string {
	if len(ms.HFInclude) != 1 || ms.HFSubdir != "" {
		return ""
	}
	f := strings.TrimPrefix(strings.TrimSpace(ms.HFInclude[0]), "/")
	if f == "" || strings.ContainsAny(f, "*?[") || !hfwrap.MatchPatterns(f, nil, ms.HFExclude) {
		return ""
	}
	return f
}

func hfLookupConfig(cfg config.Config, src config.ModelSource, revision, file string) hfwrap.Config {
	return hfwrap.Config{
		Repo:     src.HFRepo,
		Revision: revision,
		Token:    cfg.HFToken,
		Endpoint: cfg.HFEndpoint,
		HFFile:   file,
	}
}

// sizedEntry keeps a HEAD's description only when it carried a size;
// without one the budget lists the tree instead.
func sizedEntry(e hfwrap.TreeEntry) *hfwrap.TreeEntry {
	if e.Size <= 0 {
		return nil
	}
	return &e
}

// estimateHFTarget sizes one source from its HEAD when that answered,
// and from the tree at the resolved commit otherwise.
func estimateHFTarget(ctx context.Context, cfg config.Config, source config.ModelSource, t hfTarget) (hfwrap.RepoEstimate, error) {
	if t.entry != nil {
		return hfwrap.RepoEstimate{
			Bytes:    t.entry.Size,
			Files:    1,
			Complete: true,
			Entries:  []hfwrap.TreeEntry{*t.entry},
			Commit:   t.commit,
		}, nil
	}
	return estimateHFSource(ctx, cfg, source, t.commit)
}

func estimateHFSource(ctx context.Context, cfg config.Config, source config.ModelSource, commit string) (hfwrap.RepoEstimate, error) {
	rev := source.HFRevision
	if commit != "" {
		rev = commit
	}
	wrap := hfwrap.Config{
		Repo:     source.HFRepo,
		Revision: rev,
		Token:    cfg.HFToken,
		Endpoint: cfg.HFEndpoint,
	}
	return hfwrap.EstimateRepo(ctx, wrap, source.HFInclude, source.HFExclude, source.HFSubdir)
}

func finishedURLFile(cfg config.Config, ms config.ModelSource) bool {
	switch ms.Kind {
	case config.KindURL:
		return regularFileSize(ms.LocalPath) > 0
	case config.KindOllamaURL:
		if cfg.Runtime.RunDir == "" {
			return false
		}
		dest := filepath.Join(cfg.Runtime.RunDir, "url-fetched", urlDestName(ms))
		return regularFileSize(dest) > 0
	default:
		return false
	}
}

func urlDisplayName(ms config.ModelSource) string {
	switch ms.Kind {
	case config.KindURL:
		if ms.LocalPath != "" {
			return filepath.Base(ms.LocalPath)
		}
	case config.KindOllamaURL:
		return urlDestName(ms)
	}
	if ms.URL != "" {
		return filepath.Base(strings.Split(ms.URL, "?")[0])
	}
	return "download"
}

func urlForSource(ms config.ModelSource) string {
	switch ms.Kind {
	case config.KindURL, config.KindOllamaURL:
		return ms.URL
	default:
		return ""
	}
}

// cachedURLResumeBytes returns finished-file or .part size for a URL
// source. Matches fetch's contract that OnBytes only counts newly-fetched
// bytes after Range resume.
func cachedURLResumeBytes(cfg config.Config, ms config.ModelSource) int64 {
	switch ms.Kind {
	case config.KindURL:
		return resumeAwareFileBytes(ms.LocalPath)
	case config.KindOllamaURL:
		if cfg.Runtime.RunDir == "" {
			return 0
		}
		dest := filepath.Join(cfg.Runtime.RunDir, "url-fetched", urlDestName(ms))
		return resumeAwareFileBytes(dest)
	}
	return 0
}

// resumeAwareFileBytes prefers a finished dest file; otherwise the
// RangeDownloader's stable "<dest>.part" partial.
func resumeAwareFileBytes(path string) int64 {
	if path == "" {
		return 0
	}
	if n := regularFileSize(path); n > 0 {
		return n
	}
	return regularFileSize(path + ".part")
}

func regularFileSize(path string) int64 {
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return 0
	}
	return info.Size()
}

// hasHFFile reports whether the content the target commit lists for one
// file is already in the local cache.
//
// The local refs/<revision> is no guide here: it names the snapshot an
// earlier pass fetched, and when the branch has moved since, that
// snapshot holds files with the same names and different bytes. Seeding
// those as done counted the old model toward the new one, and the
// transfer of the new one on top of that drove bytes_completed past
// bytes_total. Both checks below are keyed by the target, so an old
// version never matches:
//
//   - snapshots/<commit>/<path> exists only once hf finished that file
//     for that commit, however it named the blob. Mirrors that omit
//     X-Linked-ETag (the Olares one) make hf name blobs by the mirror's
//     ETag, which no Hub id predicts.
//   - blobs/<id> catches content the target shares with a revision
//     fetched earlier, which hf skips without linking it in yet.
//
// A `<id>.incomplete` partial is not counted: hf resumes it and the
// first tqdm tick already carries the bytes it holds.
func hasHFFile(cacheRoot, repo, commit string, e hfwrap.TreeEntry) bool {
	if hasHFBlob(cacheRoot, repo, e) {
		return true
	}
	if cacheRoot == "" || repo == "" || !isResolvedCommit(commit) || !filepath.IsLocal(filepath.FromSlash(e.Path)) {
		return false
	}
	p := filepath.Join(hfwrap.SnapshotPath(cacheRoot, repo, commit), filepath.FromSlash(e.Path))
	info, err := os.Stat(p)
	return err == nil && info.Mode().IsRegular() && info.Size() == e.Size
}

// hasHFBlob looks the content up under blobs/ by the id the Hub gave it.
func hasHFBlob(cacheRoot, repo string, e hfwrap.TreeEntry) bool {
	if cacheRoot == "" || repo == "" || !isHFBlobID(e.BlobID) {
		return false
	}
	info, err := os.Stat(filepath.Join(hfwrap.HFRepoCacheDir(cacheRoot, repo), "blobs", e.BlobID))
	return err == nil && info.Mode().IsRegular() && info.Size() == e.Size
}

// isHFBlobID accepts the two forms the cache names blobs by, which also
// keeps an id from a misbehaving endpoint from naming a path elsewhere.
func isHFBlobID(s string) bool { return isSHA256Hex(s) || isResolvedCommit(s) }
