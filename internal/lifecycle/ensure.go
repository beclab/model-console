package lifecycle

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/llm-init/llm-init/internal/adapter/hfwrap"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/fetch"
	"github.com/llm-init/llm-init/internal/progress"
	"github.com/llm-init/llm-init/internal/sentinel"
	"github.com/llm-init/llm-init/internal/verify"
)

// ensure is one full pass of the lifecycle state machine. It dispatches
// on the main source's Kind: the first MODEL_SOURCE segment, which is the
// only entry that binds the engine.
//
// A ROLE=mmproj sibling (inline --role mmproj) is handled inline by
// ensureHF; ROLE=extra segments are pre-fetched before the main dispatch.
// After a successful main ensure, filesystem paths for ROLE=extra are
// written to /run/llm-init/extra_model_path (mmproj never appears there —
// it stays on the ensureHF inline path).
func (m *Manager) ensure(ctx context.Context) error {
	cfg := m.opts.Config
	opts := m.consumePendingOpts()

	// Any loading this pass writes is the pass's own, so the health loop
	// stops claiming the one it wrote for a relaunch before this.
	m.relaunchLoading.Store(false)

	main, mmproj, extras := splitSources(cfg.Sources)
	if main == nil {
		return m.fail("no main MODEL_SOURCE configured", false)
	}

	m.pass = &passState{cfg: cfg, main: main, mmproj: mmproj, extras: extras, force: opts.Force}

	extraPaths := make([]string, 0, len(extras))
	for i := range extras {
		path, err := m.ensureExtra(ctx, cfg, extras[i], opts)
		if err != nil {
			return err
		}
		if path != "" {
			extraPaths = append(extraPaths, path)
		}
	}

	var err error
	switch main.Kind {
	case config.KindOllama:
		err = m.ensureOllamaName(ctx, *main)
	case config.KindOllamaURL:
		err = m.ensureOllamaURL(ctx, cfg, *main, opts, extraPaths)
	case config.KindHF:
		err = m.ensureHF(ctx, cfg, *main, mmproj, opts, extraPaths)
	case config.KindURL:
		err = m.ensureURL(ctx, cfg, *main, opts, extraPaths)
	default:
		return m.fail(fmt.Sprintf("unsupported source kind: %q", main.Kind), false)
	}
	if err != nil {
		return err
	}
	if err := m.writeExtraPaths(cfg, extraPaths); err != nil {
		return m.fail("sentinel.WriteExtra: "+err.Error(), false)
	}
	return nil
}

// splitSources returns the main source (the first Role==main entry, or the
// first entry when no role says so), the mmproj sibling if any, and every
// ROLE=extra entry in comma-segment order.
func splitSources(sources []config.ModelSource) (main *config.ModelSource, mmproj *config.ModelSource, extras []config.ModelSource) {
	for i := range sources {
		switch sources[i].Role {
		case config.RoleMain, "":
			if main == nil {
				ms := sources[i]
				main = &ms
			}
		case config.RoleMmproj:
			if mmproj == nil {
				ms := sources[i]
				mmproj = &ms
			}
		case config.RoleExtra:
			extras = append(extras, sources[i])
		}
	}
	if main == nil && len(sources) > 0 {
		ms := sources[0]
		main = &ms
	}
	return main, mmproj, extras
}

// ensureExtra downloads one ROLE=extra source without binding the engine
// (no model_path sentinel, no Register, no PhaseReady). It returns the
// absolute filesystem path consumers should read from extra_model_path,
// or "" when the kind has no on-disk artefact (e.g. ollama:// library tag).
// The same path also feeds the final progress reconciliation.
//
// HF extras use the same resolveHFEnginePath rules as main. URL extras use
// MODEL_SOURCE_LOCAL. ROLE=mmproj is never routed here (see splitSources).
func (m *Manager) ensureExtra(ctx context.Context, cfg config.Config, ms config.ModelSource, opts RetryOptions) (string, error) {
	sink := m.opts.Manager.Sink()

	switch ms.Kind {
	case config.KindOllama:
		if ms.OllamaTag == "" {
			return "", m.fail("ollama:// extra source missing tag", false)
		}
		m.enterDownloadPhase(ctx)
		slog.Info("ollama extra pull starting",
			slog.Int("source_index", ms.Index),
			slog.String("role", ms.Role),
			slog.String("tag", ms.OllamaTag))
		if err := m.installer.Pull(ctx, ms.OllamaTag, sink); err != nil {
			return "", m.fail(ollamaPullUserMessage(ms.OllamaTag, err), true)
		}
		return "", nil
	case config.KindHF:
		slog.Info("hf extra download starting",
			slog.Int("source_index", ms.Index),
			slog.String("role", ms.Role),
			slog.String("repo", ms.HFRepo))
		res, err := m.runHF(ctx, cfg, ms, opts)
		if err != nil {
			return "", err
		}
		path, err := resolveHFEnginePath(ms, res)
		if err != nil {
			return "", m.fail(err.Error(), false)
		}
		return path, nil
	case config.KindURL:
		if ms.LocalPath == "" {
			return "", m.fail("MODEL_SOURCE_LOCAL is required for https?:// extra source", false)
		}
		if err := os.MkdirAll(filepath.Dir(ms.LocalPath), 0o755); err != nil {
			return "", m.fail(fmt.Sprintf("MkdirAll %s: %v", filepath.Dir(ms.LocalPath), err), true)
		}
		level := passLevel(cfg, opts)
		if opts.Force {
			removeURLSidecar(ms.LocalPath)
			_ = os.Remove(ms.LocalPath)
			_ = os.Remove(ms.LocalPath + ".part")
		}
		if !m.reuseURLDownload(ctx, cfg, ms, ms.LocalPath, level) {
			if err := m.downloadAndRecordURL(ctx, ms, ms.LocalPath, level); err != nil {
				return "", err
			}
		}
		return ms.LocalPath, nil
	case config.KindOllamaURL:
		path, err := m.ensureExtraOllamaURL(ctx, cfg, ms, opts)
		return path, err
	default:
		return "", m.fail(fmt.Sprintf("unsupported extra source kind: %q", ms.Kind), false)
	}
}

// ensureExtraOllamaURL fetches a URL into the run dir for cache warm-up
// without registering it as MODEL_NAME in the daemon. Returns the dest path
// so it can be listed in extra_model_path when desired.
func (m *Manager) ensureExtraOllamaURL(ctx context.Context, cfg config.Config, ms config.ModelSource, opts RetryOptions) (string, error) {
	if cfg.Runtime.RunDir == "" {
		return "", m.fail("RUN_DIR is required for ollama:// URL extra source", false)
	}
	dir := filepath.Join(cfg.Runtime.RunDir, "url-fetched")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", m.fail(fmt.Sprintf("MkdirAll %s: %v", dir, err), true)
	}
	dest := filepath.Join(dir, urlDestName(ms))
	level := passLevel(cfg, opts)
	if opts.Force {
		removeURLSidecar(dest)
		_ = os.Remove(dest)
		_ = os.Remove(dest + ".part")
	}
	urlMS := ms
	urlMS.LocalPath = dest
	if m.reuseURLDownload(ctx, cfg, urlMS, dest, level) {
		return dest, nil
	}
	slog.Info("ollama-url extra fetch starting",
		slog.Int("source_index", ms.Index),
		slog.String("role", ms.Role),
		slog.String("url", ms.URL))
	if err := m.downloadAndRecordURL(ctx, urlMS, dest, level); err != nil {
		return "", err
	}
	return dest, nil
}

// ensureOllamaName drives Adapter.Pull for ollama:// Library tags.
// The daemon owns the model bytes; lifecycle just kicks the daemon and
// waits for /api/tags to confirm.
func (m *Manager) ensureOllamaName(ctx context.Context, ms config.ModelSource) error {
	if ms.OllamaTag == "" {
		return m.fail("ollama:// source missing tag", false)
	}
	// The daemon owns these bytes, so llm-init has no record to consult
	// and no way to know whether the pull will transfer anything. The
	// phase is entered unconditionally, as it always was.
	m.enterDownloadPhase(ctx)
	sink := m.opts.Manager.Sink()
	if err := m.installer.Pull(ctx, ms.OllamaTag, sink); err != nil {
		return m.fail(ollamaPullUserMessage(ms.OllamaTag, err), true)
	}
	m.setPhase(progress.PhaseLoading)
	m.setPhase(progress.PhaseReady)
	if rs, err := m.opts.Adapter.Ready(ctx); err == nil {
		m.setModelBytes(rs.ModelBytes)
		m.lastReady.Store(&rs)
	}
	return nil
}

// ensureHF dispatches an HF download via the `hf` CLI (hfwrap.Run).
// `hf download --cache-dir <root>` is naturally idempotent: a second
// call against a fully-cached snapshot is a fast metadata diff. Force
// bypasses the fast-path via --force-download.
//
// mmproj (when supplied) is downloaded in a SECOND `hf` invocation
// against its own repo+revision and narrower --include, so the projector
// may sit beside the main GGUF or in a repo of its own while model_path
// still points at the main file.
//
// Engine-facing register/sentinel paths point at the resolved snapshot
// directory `<cache>/models--<owner>--<repo>/snapshots/<sha>/`, a single
// file inside it (one --include / llama.cpp GGUF), or snapshot/<subdir>
// when MODEL_SOURCE carries --subdir (Embed unified HF repos).
func (m *Manager) ensureHF(ctx context.Context, cfg config.Config, main config.ModelSource, mmproj *config.ModelSource, opts RetryOptions, progressPaths []string) error {
	mainRes, err := m.runHF(ctx, cfg, main, opts)
	if err != nil {
		return err
	}

	// One --include resolves hfwrap's path to the projector file itself,
	// which is what Register and progress need; config rejects any other
	// shape, so no other shape reaches here.
	var mmprojPath string
	if mmproj != nil && mmproj.Kind == config.KindHF && len(mmproj.HFInclude) == 1 {
		mmprojRes, mmErr := m.runHF(ctx, cfg, *mmproj, opts)
		if mmErr != nil {
			return mmErr
		}
		mmprojPath = mmprojRes.Path
	}

	// Map hfwrap's download path to what engine wrappers consume. When
	// --subdir is set this becomes .../snapshots/<sha>/<subdir> and is
	// written to /run/llm-init/model_path for embed.sh → EMBED_MODEL_DIR.
	enginePath, err := resolveHFEnginePath(main, mainRes)
	if err != nil {
		return m.fail(err.Error(), false)
	}

	// Progress covers every local source; registration and sentinel remain
	// bound to the main engine path.
	registerPaths := m.collectHFRegisterPaths(mainRes, mmprojPath)
	progressPaths = appendHFProgressPaths(progressPaths, mainRes.Path, mmprojPath)
	if enginePath != "" {
		if len(registerPaths) > 0 {
			registerPaths[0] = enginePath
		} else {
			registerPaths = []string{enginePath}
		}
	}
	m.reconcileModelBytes(progressPaths)

	m.setPhase(progress.PhaseLoading)
	if err := m.installer.Register(ctx, registerPaths, nil, m.opts.Manager.Sink()); err != nil {
		return m.fail("adapter.Register: "+err.Error(), false)
	}

	if err := m.writeSentinelHF(cfg, enginePath); err != nil {
		return m.fail("sentinel.Write: "+err.Error(), false)
	}
	m.recordVerify(true)
	m.observeVerify("ok", "startup")
	// Proxy engines stay in PhaseLoading until the engine reports alive;
	// the transition to PhaseReady is owned by manager.Run after WaitAlive.
	return nil
}

// runHF invokes hfwrap.Run for one MODEL_SOURCE. The
// HF cache root is fixed at /cache/hf/hub by the deploy template
// contract (Dockerfile sets HF_HUB_CACHE so the runtime image and the
// `hf` CLI agree without per-process injection).
//
// A record left by an earlier pass short-circuits the subprocess
// entirely, which is what lets a machine with no route to the Hub boot
// a model it already has. `hf download` cannot do that itself: even
// against a fully-cached snapshot it asks the Hub what the revision
// resolves to now.
func (m *Manager) runHF(ctx context.Context, cfg config.Config, ms config.ModelSource, opts RetryOptions) (hfwrap.Result, error) {
	if opts.Force {
		removeHFSidecars(m.hfCacheRoot(), ms.HFRepo, hfSourceIdentity(cfg, ms))
	} else if res, ok := m.reuseHFDownload(ctx, cfg, ms, passLevel(cfg, opts)); ok {
		return res, nil
	}

	m.enterDownloadPhase(ctx)

	allow := append([]string(nil), ms.HFInclude...)
	var hfFile string
	if len(allow) == 1 {
		hfFile = allow[0]
		allow = nil
	}

	slog.Info("hf download starting",
		slog.Int("source_index", ms.Index),
		slog.String("role", ms.Role),
		slog.String("repo", ms.HFRepo),
		slog.String("revision", ms.HFRevision),
		slog.String("cache_dir", m.hfCacheRoot()),
		slog.Bool("force", opts.Force),
		slog.Any("allow_patterns", allow),
		slog.Any("exclude_patterns", ms.HFExclude),
		slog.String("hf_file", hfFile))

	wrapCfg := hfwrap.Config{
		Repo:            ms.HFRepo,
		Revision:        ms.HFRevision,
		CacheDir:        m.hfCacheRoot(),
		HFFile:          hfFile,
		AllowPatterns:   allow,
		ExcludePatterns: append([]string(nil), ms.HFExclude...),
		Token:           cfg.HFToken,
		Endpoint:        cfg.HFEndpoint,
		MaxWorkers:      m.opts.MaxConcurrentDownloads,
		ForceReload:     opts.Force,
		EnableXet:       cfg.Runtime.EnableXet,
		Tree:            m.pass.hfTree(ms),
	}
	runFn := m.opts.HFRun
	if runFn == nil {
		runFn = hfwrap.Run
	}
	res, err := runFn(ctx, wrapCfg, m.opts.Manager.Sink())
	if err != nil {
		retriable := hfwrap.IsRetriable(err)
		if m.opts.Metrics != nil {
			m.opts.Metrics.DownloadRetriesTotal.
				WithLabelValues(string(config.KindHF), classifyHFWrapErr(err)).
				Inc()
			m.opts.Metrics.TransportRetriesTotal.
				WithLabelValues(classifyHFWrapErr(err)).
				Inc()
		}
		return hfwrap.Result{}, m.fail(hfUserMessage(err), retriable)
	}
	m.writeHFSidecar(cfg, ms, res)
	return res, nil
}

// defaultHFCacheRoot is the deploy-pinned HF_HUB_CACHE path (Dockerfile
// sets HF_HUB_CACHE=/cache/hf/hub, deploy templates mount the volume
// there). All hf:// MODEL_SOURCE entries land under this root.
const defaultHFCacheRoot = "/cache/hf/hub"

// hfCacheRoot is where this manager reads and writes the HF cache.
func (m *Manager) hfCacheRoot() string {
	if m.opts.HFCacheRoot != "" {
		return m.opts.HFCacheRoot
	}
	return defaultHFCacheRoot
}

// ensureURL drives the in-tree RangeDownloader for a single MODEL_SOURCE
// https?:// download. The downloader short-circuits when the file is
// already complete (Range/ETag/HEAD).
func (m *Manager) ensureURL(ctx context.Context, cfg config.Config, ms config.ModelSource, opts RetryOptions, progressPaths []string) error {
	if ms.LocalPath == "" {
		return m.fail("MODEL_SOURCE_LOCAL is required for https?:// source", false)
	}
	if err := os.MkdirAll(filepath.Dir(ms.LocalPath), 0o755); err != nil {
		return m.fail(fmt.Sprintf("MkdirAll %s: %v", filepath.Dir(ms.LocalPath), err), true)
	}

	level := passLevel(cfg, opts)
	if opts.Force {
		m.clearSentinelForForce(cfg)
		removeURLSidecar(ms.LocalPath)
		_ = os.Remove(ms.LocalPath)
		_ = os.Remove(ms.LocalPath + ".part")
	}

	if !m.reuseURLDownload(ctx, cfg, ms, ms.LocalPath, level) {
		if err := m.downloadAndRecordURL(ctx, ms, ms.LocalPath, level); err != nil {
			return err
		}
	}
	m.reconcileModelBytes(append(progressPaths, ms.LocalPath))

	m.setPhase(progress.PhaseLoading)
	if err := m.installer.Register(ctx, []string{ms.LocalPath}, nil, m.opts.Manager.Sink()); err != nil {
		return m.fail("adapter.Register: "+err.Error(), false)
	}

	if err := m.writeSentinelPath(cfg, ms.LocalPath); err != nil {
		return m.fail("sentinel.Write: "+err.Error(), false)
	}
	m.recordVerify(true)
	m.observeVerify("ok", "startup")
	// Proxy engines stay in PhaseLoading until the engine reports alive;
	// the transition to PhaseReady is owned by manager.Run after WaitAlive.
	return nil
}

// ensureOllamaURL fetches a URL into the run dir, then asks the daemon
// to register the blob with Modelfile=`FROM @sha256:<digest>` (no
// template/system/parameters injection - those are baked into modern
// GGUF metadata). Implemented by re-using Adapter.Register against
// the downloaded file: v1.1.0 retired the GGUF_* envs and dropped
// the corresponding Config fields, so Register's body collapses to
// a single-step /api/create with files=<sha> map.
func (m *Manager) ensureOllamaURL(ctx context.Context, cfg config.Config, ms config.ModelSource, opts RetryOptions, progressPaths []string) error {
	if cfg.Runtime.RunDir == "" {
		return m.fail("RUN_DIR is required for ollama:// URL overload", false)
	}
	dir := filepath.Join(cfg.Runtime.RunDir, "url-fetched")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return m.fail(fmt.Sprintf("MkdirAll %s: %v", dir, err), true)
	}
	dest := filepath.Join(dir, urlDestName(ms))

	level := passLevel(cfg, opts)
	if opts.Force {
		m.clearSentinelForForce(cfg)
		removeURLSidecar(dest)
		_ = os.Remove(dest)
		_ = os.Remove(dest + ".part")
	}

	urlMS := ms
	urlMS.LocalPath = dest
	if !m.reuseURLDownload(ctx, cfg, urlMS, dest, level) {
		if err := m.downloadAndRecordURL(ctx, urlMS, dest, level); err != nil {
			return err
		}
	}
	m.reconcileModelBytes(append(progressPaths, dest))

	m.setPhase(progress.PhaseLoading)
	if err := m.registerWithKnownDigests(ctx, dest); err != nil {
		return m.fail("adapter.Register: "+err.Error(), false)
	}

	if err := m.writeSentinelPath(cfg, dest); err != nil {
		return m.fail("sentinel.Write: "+err.Error(), false)
	}
	m.recordVerify(true)
	m.observeVerify("ok", "startup")
	// Ollama's daemon is alive before boot, so the engine is ready as soon
	// as the model is registered; flip straight to ready.
	m.setPhase(progress.PhaseReady)
	if rs, err := m.opts.Adapter.Ready(ctx); err == nil {
		m.lastReady.Store(&rs)
	}
	return nil
}

const defaultURLModelFilename = "model.bin"

// urlDestName picks a stable on-disk filename for an ollama:// URL
// overload. SHA256 fragment when supplied, otherwise the URL basename.
func urlDestName(ms config.ModelSource) string {
	if ms.URLSha256 != "" {
		return ms.URLSha256 + ".gguf"
	}
	if i := strings.LastIndexByte(ms.URL, '/'); i >= 0 && i < len(ms.URL)-1 {
		base := ms.URL[i+1:]
		if q := strings.IndexAny(base, "?#"); q >= 0 {
			base = base[:q]
		}
		// Reduce to a single path element and reject traversal so a
		// crafted URL (".../../../etc/passwd") cannot escape the
		// url-fetched dir via filepath.Join above.
		base = filepath.Base(base)
		if base != "" && base != "." && base != ".." {
			return base
		}
	}
	return defaultURLModelFilename
}

// downloadAndRecordURL fetches one URL source and records what landed,
// so a later pass can recognise these bytes without asking the upstream.
//
// The old record is dropped first. It describes bytes that are about to
// be overwritten, and a download interrupted between the two would
// otherwise leave a record vouching for a file that is now half old and
// half new.
func (m *Manager) downloadAndRecordURL(ctx context.Context, ms config.ModelSource, dest string, level verify.Level) error {
	m.enterDownloadPhase(ctx)
	removeURLSidecar(dest)
	etag, err := m.downloadURL(ctx, ms)
	if err != nil {
		return err
	}
	m.writeURLSidecar(ctx, ms, dest, level, etag)
	return nil
}

// downloadURL drives the RangeDownloader for one URL ModelSource and
// optionally verifies a #sha256 fragment. It returns the ETag the
// upstream advertised, empty when it sent none.
func (m *Manager) downloadURL(ctx context.Context, ms config.ModelSource) (string, error) {
	var etag string
	dl := fetch.RangeOptions{
		URL:      ms.URL,
		DestPath: ms.LocalPath,
		MaxBytes: m.opts.MaxDownloadBytes,
		OnMeta:   func(_ int64, e string) { etag = e },
	}
	if m.opts.Metrics != nil {
		dl.OnRetry = func(_ int, err error) {
			class := classifyDownloadErr(err)
			m.opts.Metrics.DownloadRetriesTotal.
				WithLabelValues(string(ms.Kind), class).
				Inc()
			m.opts.Metrics.TransportRetriesTotal.
				WithLabelValues(class).
				Inc()
		}
	}
	if err := m.opts.Downloader.Download(ctx, dl, m.opts.Manager.Sink()); err != nil {
		msg, retriable := urlDownloadUserMessage(err)
		return "", m.fail(msg, retriable)
	}
	if ms.URLSha256 != "" {
		expected := strings.TrimPrefix(ms.URLSha256, "sha256:")
		if err := m.opts.Downloader.Verify(ctx, ms.LocalPath, -1, expected); err != nil {
			_ = os.Remove(ms.LocalPath)
			return "", m.fail("download verify: "+err.Error(), true)
		}
	}
	return etag, nil
}

// collectHFRegisterPaths chooses what to hand to adapter.Register for
// HF sources. Engines that mount the whole snapshot dir (vLLM, sglang,
// llama.cpp non-GGUF) get `<cache>/models--owner--repo/snapshots/<sha>`;
// engines that consume a single file (Ollama, llama.cpp GGUF) get the
// resolved file path inside the snapshot so the adapter can stream
// the blob into the engine.
//
// mmprojPath is the projector file hfwrap resolved in its own pass, so a
// projector hosted in a different repo or revision than the main weights
// still reaches the engine.
func (m *Manager) collectHFRegisterPaths(mainRes hfwrap.Result, mmprojPath string) []string {
	if mainRes.Path == "" {
		return nil
	}
	paths := []string{mainRes.Path}
	if mmprojPath != "" {
		if _, err := os.Stat(mmprojPath); err == nil {
			paths = append(paths, mmprojPath)
		}
	}
	return paths
}

// appendHFProgressPaths adds this pass's HF artefacts to the progress
// reconciliation set. A projector inside the main path is left out: with a
// whole-snapshot main path the projector already sits under it, and
// reconcileModelBytes walks directories, so listing both double-counts it.
func appendHFProgressPaths(paths []string, mainPath, mmprojPath string) []string {
	if mainPath == "" {
		return paths
	}
	paths = append(paths, mainPath)
	if mmprojPath != "" && !strings.HasPrefix(mmprojPath, mainPath+string(filepath.Separator)) {
		paths = append(paths, mmprojPath)
	}
	return paths
}

// writeSentinelHF writes /run/llm-init/model_path (+ model_download_finish)
// for hf:// sources. enginePath is the final directory/file engines load:
// snapshot root, single GGUF file, or snapshot/<subdir> when --subdir set.
// Empty enginePath skips the write (legacy no-op when hfwrap returns no path).
// ROLE=extra paths are written separately via writeExtraPaths after ensure.
func (m *Manager) writeSentinelHF(cfg config.Config, enginePath string) error {
	if cfg.Runtime.RunDir == "" || enginePath == "" {
		return nil
	}
	return sentinel.Write(cfg.Runtime.RunDir, enginePath)
}

// clearSentinelForForce drops the run dir's readiness files before a
// force pass deletes the bytes they point at.
//
// Called from the URL and ollama-URL paths, which os.Remove the exact
// file model_path names and unconditionally rewrite the sentinel when
// the pass succeeds. Without this there is a window — minutes, for a
// large model — in which the sentinel says "ready, load this" about a
// path that no longer exists, and an engine container restarted by the
// kubelet in that window execs it and CrashLoops on a missing file
// instead of waiting.
//
// hf:// deliberately does not call this. `--force-download` re-downloads
// into the cache without invalidating the snapshot path in between, so
// model_path stays good the whole way; and writeSentinelHF is a no-op
// when hfwrap reports no path, which would leave a wrapper waiting on a
// sentinel nothing is going to write back.
//
// A failure to remove is logged and swallowed: the pass ahead can still
// succeed, and refusing to re-download because of a stale marker file
// would be the worse outcome.
func (m *Manager) clearSentinelForForce(cfg config.Config) {
	if cfg.Runtime.RunDir == "" {
		return
	}
	if err := sentinel.Remove(cfg.Runtime.RunDir); err != nil {
		slog.Warn("could not clear the readiness sentinel before a forced re-download",
			slog.String("run_dir", cfg.Runtime.RunDir),
			slog.String("error", err.Error()))
		return
	}
	slog.Info("cleared the readiness sentinel for a forced re-download",
		slog.String("run_dir", cfg.Runtime.RunDir))
}

// writeSentinelPath persists a sentinel pointing at one absolute path.
// Used by URL and ollama-URL paths where the downloaded file is the
// engine-facing artefact.
func (m *Manager) writeSentinelPath(cfg config.Config, target string) error {
	if cfg.Runtime.RunDir == "" {
		return nil
	}
	return sentinel.Write(cfg.Runtime.RunDir, target)
}

// writeExtraPaths persists /run/llm-init/extra_model_path for ROLE=extra
// sources (one absolute path per line, comma-segment order). Empty paths
// removes the file so retries cannot leave a stale entry. ROLE=mmproj is
// never included (handled inline by ensureHF).
func (m *Manager) writeExtraPaths(cfg config.Config, paths []string) error {
	if cfg.Runtime.RunDir == "" {
		return nil
	}
	return sentinel.WriteExtra(cfg.Runtime.RunDir, paths)
}

// fail records a failure phase and returns the error so Run can decide
// whether to back off (retriable=true) or sit idle waiting for /api/retry
// (retriable=false). The caller should always return the value — fail
// only logs to progress.
func (m *Manager) fail(msg string, retriable bool) error {
	phase := progress.PhaseFailed
	if retriable {
		phase = progress.PhaseDegraded
	}
	m.opts.Manager.Update(func(s *progress.State) {
		s.Phase = phase
		s.LastError = msg
		s.RetryCount++
	})
	if retriable {
		return retriableErr{err: errors.New(msg)}
	}
	return errors.New(msg)
}

// reconcileModelBytes completes the per-run byte counters using the actual
// engine model size without shrinking a larger aggregate source budget.
func (m *Manager) reconcileModelBytes(paths []string) {
	var total int64
	for _, p := range paths {
		if p == "" {
			continue
		}
		total += onDiskSize(p)
	}
	m.opts.Manager.Update(func(s *progress.State) {
		if s.BytesTotal > total {
			total = s.BytesTotal
		}
		if total > 0 {
			s.BytesTotal = total
			s.BytesCompleted = total
		}
	})
}

// setModelBytes marks the download 100% done at a known total size.
// total <= 0 (size unknown) is a no-op so the dashboard keeps showing
// "Initializing…" rather than a bogus 0 / 0. Used by the on-disk
// reconcile (HF / URL) and by the Ollama path (daemon-reported size).
func (m *Manager) setModelBytes(total int64) {
	if total <= 0 {
		return
	}
	m.opts.Manager.Update(func(s *progress.State) {
		s.BytesTotal = total
		s.BytesCompleted = total
	})
}

// onDiskSize sums the real byte size of a file or directory tree,
// following symlinks because HF snapshot dirs are symlinks into blobs/
// (an Lstat-based walk would only count the link inodes, ~0 bytes).
func onDiskSize(path string) int64 {
	var sum int64
	_ = filepath.Walk(path, func(p string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		if st, e := os.Stat(p); e == nil {
			sum += st.Size()
		}
		return nil
	})
	return sum
}

// setPhase updates progress.Manager.Phase. On the first transition to
// PhaseReady we also stamp EngineColdStartMS as (now - StartedAt).
//
// LastError is a *current-failure* signal for the dashboard (the Status
// card stays red while it is non-empty), not an audit log. Clear it on
// every fresh download attempt and on Ready so a recovered retry does
// not keep advertising the previous timeout. RetryCount /
// TransportRetries / StartedAt remain the durable counters.
func (m *Manager) setPhase(p progress.Phase) {
	now := m.opts.NowFunc()
	m.opts.Manager.Update(func(s *progress.State) {
		s.Phase = p
		switch p {
		case progress.PhaseDownload:
			// Zero per-download gauges so each ensure attempt starts
			// from a clean slate (v1.0.10-fix11). Also drop LastError:
			// re-entering download means the lifecycle is actively
			// trying again; keeping the prior message made the UI look
			// failed while bytes were flowing.
			s.BytesTotal = 0
			s.BytesCompleted = 0
			s.SpeedBytesPerSec = 0
			s.ETASeconds = 0
			s.FilesTotal = 0
			s.FilesCompleted = 0
			s.CurrentFile = ""
			s.Files = nil
			s.LastError = ""
		case progress.PhaseLoading:
			// The download stage of this pass is over, so nothing is
			// still arriving and no row may read `waiting` or
			// `downloading`. Without this a tqdm bar whose final line
			// never reached the aggregator — two files sharing one
			// label used to cost one of them every tick it had —
			// leaves a row stuck mid-transfer on a model that serves.
			settleFileRows(s)
		case progress.PhaseReady:
			s.LastError = ""
			if s.EngineColdStartMS == nil {
				ms := now.Sub(s.StartedAt).Milliseconds()
				if ms < 0 {
					ms = 0
				}
				s.EngineColdStartMS = &ms
			}
		}
	})
	if p == progress.PhaseDownload {
		m.opts.Manager.ClearDownloadBudget()
	}
}

// settleFileRows marks every file row complete. The bytes behind them
// are on disk whether this pass transferred them or found them cached,
// so the count follows the rows rather than the other way round.
func settleFileRows(s *progress.State) {
	done := 0
	for i := range s.Files {
		s.Files[i].Status = progress.FileDone
		if s.Files[i].BytesTotal > 0 {
			s.Files[i].BytesCompleted = s.Files[i].BytesTotal
		}
		done++
	}
	if done > s.FilesTotal {
		s.FilesTotal = done
	}
	s.FilesCompleted = s.FilesTotal
}

// retriableErr signals "ensure failed but a retry might succeed".
type retriableErr struct{ err error }

func (e retriableErr) Error() string { return e.err.Error() }
func (e retriableErr) Unwrap() error { return e.err }

// Download error_code taxonomy, advertised on the `error_code` label
// of `llm_init_download_retries_total`. Closed enum to keep cardinality
// bounded.
const (
	errCodeOther          = "other"
	errCode5xx            = "5xx"
	errCode429            = "429"
	errCodeNetTimeout     = "net_timeout"
	errCodeEOF            = "eof"
	errCodeHFRepoNotFound = "hf_repo_not_found"
	errCodeHFGated        = "hf_gated"
	errCodeHFRevision     = "hf_revision"
	errCodeHFTokenMissing = "hf_token_missing"
	errCodeHFNetwork      = "hf_network"
	errCodeHFDiskFull     = "hf_disk_full"
	errCodeHFInternal     = "hf_internal"
)

// classifyDownloadErr maps a fetch retry's underlying error to the
// bounded `error_code` enum.
func classifyDownloadErr(err error) string {
	if err == nil {
		return errCodeOther
	}
	var httpErr *fetch.HTTPStatusError
	if errors.As(err, &httpErr) {
		switch {
		case httpErr.StatusCode == 429:
			return errCode429
		case httpErr.StatusCode >= 500 && httpErr.StatusCode < 600:
			return errCode5xx
		}
	}
	var netErr interface{ Timeout() bool }
	if errors.As(err, &netErr) && netErr.Timeout() {
		return errCodeNetTimeout
	}
	if errors.Is(err, io.EOF) ||
		errors.Is(err, io.ErrUnexpectedEOF) ||
		errors.Is(err, syscall.ECONNRESET) ||
		errors.Is(err, syscall.EPIPE) ||
		errors.Is(err, syscall.ECONNABORTED) {
		return errCodeEOF
	}
	return errCodeOther
}

// hfUserMessage turns an hfwrap error into an operator-facing message
// surfaced verbatim on /api/progress's last_error (and the /readyz
// reason), so the dashboard tells the user what to fix — a wrong
// owner/repo, a missing or unauthorized HF_TOKEN, a gated model, a bad
// revision — instead of a raw subprocess exit string. The underlying
// detail is appended in parentheses for support tickets.
func hfUserMessage(err error) string {
	var e *hfwrap.Error
	if !errors.As(err, &e) {
		return "hf download failed: " + err.Error()
	}
	var hint string
	switch e.Code {
	case hfwrap.CodeRepoNotFound:
		hint = "HuggingFace repository not found: check the MODEL_SOURCE owner/repo is spelled correctly and the repo exists"
	case hfwrap.CodeGated:
		hint = "HuggingFace repository is gated: request access on the model page and set HF_TOKEN to an authorized token"
	case hfwrap.CodeTokenMissing:
		hint = "HuggingFace authentication failed: set HF_TOKEN to a token with access to this repository"
	case hfwrap.CodeRevision:
		hint = "HuggingFace revision not found: check --revision points to a valid commit, branch, or tag"
	case hfwrap.CodeDiskFull:
		hint = "Disk full while downloading from HuggingFace: free space on the model cache volume"
	case hfwrap.CodePermission:
		hint = "Permission denied writing the HuggingFace cache: check the cache volume mount permissions"
	case hfwrap.CodeBadArgs:
		hint = "Invalid HuggingFace download arguments: check the MODEL_SOURCE flags (--include/--exclude/--revision)"
	case hfwrap.CodeNetwork, hfwrap.CodeProcessKilled:
		hint = "Network error contacting HuggingFace; will retry"
	default:
		hint = "HuggingFace download failed"
	}
	return hint + " (" + e.Error() + ")"
}

// urlDownloadUserMessage maps a fetch.RangeDownloader failure to an
// operator-facing message plus a retriability verdict. A permanent
// upstream answer for a direct https?:// source — 404 (wrong URL), 401 /
// 403 (no permission), or a size mismatch — should surface as a failed
// phase the operator must fix, not loop forever in degraded. Throttling
// (408/429), 5xx, and transport errors stay retriable. The raw error is
// appended for diagnostics.
func urlDownloadUserMessage(err error) (msg string, retriable bool) {
	var httpErr *fetch.HTTPStatusError
	if errors.As(err, &httpErr) {
		switch {
		case httpErr.StatusCode == 401 || httpErr.StatusCode == 403:
			return fmt.Sprintf("Access denied (HTTP %d) downloading the model URL: check the URL credentials or that the object is publicly readable (%s)",
				httpErr.StatusCode, err.Error()), false
		case httpErr.StatusCode == 404:
			return fmt.Sprintf("Model URL not found (HTTP 404): check the MODEL_SOURCE URL is correct (%s)", err.Error()), false
		case httpErr.StatusCode == 408 || httpErr.StatusCode == 429:
			return fmt.Sprintf("Upstream throttled the download (HTTP %d); will retry (%s)", httpErr.StatusCode, err.Error()), true
		case httpErr.StatusCode >= 400 && httpErr.StatusCode < 500:
			return fmt.Sprintf("Download failed with HTTP %d: check the MODEL_SOURCE URL (%s)", httpErr.StatusCode, err.Error()), false
		}
	}
	if errors.Is(err, fetch.ErrSizeMismatch) {
		return "Downloaded size does not match the server's advertised size; check the MODEL_SOURCE URL (" + err.Error() + ")", false
	}
	return "download error; will retry (" + err.Error() + ")", true
}

// ollamaPullUserMessage maps an Ollama /api/pull failure to an
// operator-facing message: an unknown library tag (invalid input), a
// denied / unauthorized pull (permission), or an unreachable daemon.
// The raw client error is appended for diagnostics.
func ollamaPullUserMessage(tag string, err error) string {
	msg := err.Error()
	lower := strings.ToLower(msg)
	switch {
	case strings.Contains(lower, "status 404"),
		strings.Contains(lower, "not found"),
		strings.Contains(lower, "file does not exist"),
		strings.Contains(lower, "manifest unknown"):
		return fmt.Sprintf("Ollama model %q not found in the library: check the ollama:// tag is a valid model name (%s)", tag, msg)
	case strings.Contains(lower, "status 401"),
		strings.Contains(lower, "status 403"),
		strings.Contains(lower, "unauthorized"),
		strings.Contains(lower, "forbidden"),
		strings.Contains(lower, "access denied"):
		return fmt.Sprintf("Ollama denied access to model %q: the model may be private or require authentication (%s)", tag, msg)
	case strings.Contains(lower, "connection refused"),
		strings.Contains(lower, "no such host"),
		strings.Contains(lower, "dial tcp"),
		strings.Contains(lower, "connection reset"):
		return fmt.Sprintf("Cannot reach the Ollama daemon while pulling %q; will retry (%s)", tag, msg)
	default:
		return fmt.Sprintf("Ollama pull failed for %q (%s)", tag, msg)
	}
}

// classifyHFWrapErr maps a *hfwrap.Error into the same metric label
// taxonomy.
func classifyHFWrapErr(err error) string {
	if err == nil {
		return errCodeOther
	}
	var e *hfwrap.Error
	if !errors.As(err, &e) {
		return classifyDownloadErr(err)
	}
	switch e.Code {
	case hfwrap.CodeRepoNotFound:
		return errCodeHFRepoNotFound
	case hfwrap.CodeGated:
		return errCodeHFGated
	case hfwrap.CodeRevision:
		return errCodeHFRevision
	case hfwrap.CodeTokenMissing:
		return errCodeHFTokenMissing
	case hfwrap.CodeNetwork, hfwrap.CodeProcessKilled:
		return errCodeHFNetwork
	case hfwrap.CodeDiskFull:
		return errCodeHFDiskFull
	default:
		return errCodeHFInternal
	}
}

// recordVerify / observeVerify are kept as no-op-ish helpers so the
// existing /healthz JSON shape (last_verify_at / last_verify_ok) and
// the llm_init_verify_total counter do not break.
func (m *Manager) recordVerify(ok bool) {
	now := m.opts.NowFunc().UTC()
	okPtr := ok
	m.opts.Manager.Update(func(s *progress.State) {
		s.LastVerifyAt = now
		s.LastVerifyOK = &okPtr
	})
}

func (m *Manager) observeVerify(result, trigger string) {
	if m.opts.Metrics == nil {
		return
	}
	m.opts.Metrics.VerifyTotal.WithLabelValues(result, trigger).Inc()
}
