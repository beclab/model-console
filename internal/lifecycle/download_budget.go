package lifecycle

import (
	"context"
	"os"
	"path/filepath"
	"strings"

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
	b := m.estimateDownloadBudget(ctx, p.cfg, p.main, p.mmproj, p.extras)
	p.hfTrees = b.trees
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
	for i := range sources {
		if sources[i].Kind == config.KindOllama {
			return downloadBudget{}
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
			b, n := cachedHFSnapshot(m.hfCacheRoot(), sources[i])
			out.bytesCached += b
			out.filesCached += n
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
		est, err := estimateHFSource(ctx, cfg, hfSources[i])
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
		cached := cachedHFSnapshotFiles(m.hfCacheRoot(), hfSources[i])
		for _, e := range est.Entries {
			fp := progress.FileProgress{Path: e.Path, BytesTotal: e.Size, Status: progress.FileWaiting}
			if n := cached[e.Path]; n > 0 {
				fp.Status = progress.FileDone
				fp.BytesCompleted = n
				if fp.BytesTotal > 0 && fp.BytesCompleted > fp.BytesTotal {
					fp.BytesCompleted = fp.BytesTotal
				}
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
	}
}

func estimateHFSource(ctx context.Context, cfg config.Config, source config.ModelSource) (hfwrap.RepoEstimate, error) {
	wrap := hfwrap.Config{
		Repo:     source.HFRepo,
		Revision: source.HFRevision,
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

// cachedHFSnapshot sums finished snapshot files for an hf:// source
// using the same include / exclude / subdir rules as EstimateRepo.
// Hub `.incomplete` blobs are ignored — their names are etag+uuid and
// are not reliably resumable across hf CLI reruns.
func cachedHFSnapshot(cacheRoot string, ms config.ModelSource) (bytes int64, files int) {
	sizes := cachedHFSnapshotFiles(cacheRoot, ms)
	for _, n := range sizes {
		bytes += n
		files++
	}
	return bytes, files
}

func cachedHFSnapshotFiles(cacheRoot string, ms config.ModelSource) map[string]int64 {
	out := make(map[string]int64)
	if ms.Kind != config.KindHF || ms.HFRepo == "" || cacheRoot == "" {
		return out
	}
	sha, err := hfwrap.ReadRefCommit(cacheRoot, ms.HFRepo, ms.HFRevision)
	if err != nil || sha == "" {
		return out
	}
	snap := hfwrap.SnapshotPath(cacheRoot, ms.HFRepo, sha)
	_ = filepath.WalkDir(snap, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, rerr := filepath.Rel(snap, p)
		if rerr != nil {
			return nil
		}
		rel = filepath.ToSlash(rel)
		if strings.HasSuffix(rel, ".incomplete") {
			return nil
		}
		if !hfwrap.UnderSubdir(rel, ms.HFSubdir) {
			return nil
		}
		if !hfwrap.MatchPatterns(rel, ms.HFInclude, ms.HFExclude) {
			return nil
		}
		n := regularFileSize(p)
		if n <= 0 {
			return nil
		}
		out[rel] = n
		return nil
	})
	return out
}
