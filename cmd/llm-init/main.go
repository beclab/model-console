// Command llm-init is the unified sidecar that downloads, registers, and
// proxies LLM models for various inference engines (Ollama, vLLM, llama.cpp,
// SGLang).
//
// The entrypoint loads configuration via an injectable Getenv (so tests run
// hermetically without touching os.Getenv), wires adapter+source+lifecycle,
// starts the control plane HTTP server (livez/readyz/healthz/progress/events/
// retry/config/metrics + UI), launches the data plane (/v1/*) and the
// lifecycle goroutine, and drains gracefully on SIGTERM/SIGINT.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/llm-init/llm-init/internal/adapter/factory"
	"github.com/llm-init/llm-init/internal/adapter/ollama"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/controlplane"
	"github.com/llm-init/llm-init/internal/dataplane"
	"github.com/llm-init/llm-init/internal/diag"
	"github.com/llm-init/llm-init/internal/enginecapacity"
	"github.com/llm-init/llm-init/internal/fetch"
	"github.com/llm-init/llm-init/internal/handoff"
	"github.com/llm-init/llm-init/internal/lifecycle"
	"github.com/llm-init/llm-init/internal/obs"
	"github.com/llm-init/llm-init/internal/ollamanative"
	"github.com/llm-init/llm-init/internal/progress"
	"github.com/llm-init/llm-init/internal/runtimecfg"
	"github.com/llm-init/llm-init/internal/translate"
	"github.com/llm-init/llm-init/internal/version"
)

// exitCodes (documented process exit codes):
//
//	0  normal shutdown
//	1  healthcheck probe failed
//	2  config error
//	3  disk full / permission (reserved)
//	4  panic loop (reserved)
//	5  control plane bind failure
const (
	exitOK              = 0
	exitHealthcheckFail = 1
	exitConfig          = 2
	exitControlPlane    = 5
	shutdownGracePeriod = 5 * time.Second
	healthcheckTimeout  = 2 * time.Second
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr, os.Getenv))
}

// run is the testable entry point. It returns an exit code instead of calling
// os.Exit so tests can assert on cli behaviour without spawning a process.
//
// getenv abstracts environment lookup. Pass os.Getenv from main; tests inject
// a deterministic map so the parent shell's environment cannot leak in. nil
// falls back to os.Getenv to keep the signature ergonomic.
func run(args []string, stdout, stderr *os.File, getenv config.Getenv) int {
	if getenv == nil {
		getenv = os.Getenv
	}

	fs := flag.NewFlagSet("llm-init", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		showVersion bool
		showConfig  bool
		healthcheck bool
	)
	fs.BoolVar(&showVersion, "version", false, "print version info and exit")
	fs.BoolVar(&showConfig, "print-config", false, "print parsed config (redacted) and exit")
	fs.BoolVar(&healthcheck, "healthcheck", false, "probe local /livez and exit 0/1; for distroless container HEALTHCHECK")

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return exitOK
		}
		return exitConfig
	}

	if showVersion {
		_ = version.PrintTo(stdout)
		return exitOK
	}

	// --healthcheck runs before config.Load so a partially-configured
	// container (or one whose env is intentionally minimal) can still
	// be probed. Reads PORT directly with the same default config uses.
	if healthcheck {
		return doHealthcheck(getenv, stdout, stderr)
	}

	cfg, err := config.Load(getenv)
	if err != nil {
		fmt.Fprintf(stderr, "config error: %v\n", err)
		return exitConfig
	}

	logger := newLogger(cfg.Log.Level, cfg.Log.Format, stderr)
	slog.SetDefault(logger)

	// Publish the wrapper helpers first, before any other work: the
	// sibling engine container may already be up and blocked on this
	// file, and unlike the sentinel it is waiting for something that
	// exists in this image from the start. A failure here is fatal —
	// a wrapper that never gets its helpers never launches an engine,
	// and reporting that as a config error beats a Pod pair that waits
	// an hour and then times out with nothing to point at.
	if _, err := handoff.PublishWrapperHelpers(cfg.Runtime.RunDir, getenv("WRAPPER_SRC_DIR")); err != nil {
		fmt.Fprintf(stderr, "wrapper helpers: %v\n", err)
		return exitConfig
	}

	// Make disk the source of truth for the model spec: seed the
	// env-synthesised spec to Runtime.ModelSpecPath on first boot, or
	// reload an existing file (disk wins). Runs after the logger is set
	// so the seed/load decision is visible, and before NewServer so the
	// control plane serves the reconciled spec.
	if err := config.ReconcileModelSpecFile(&cfg); err != nil {
		fmt.Fprintf(stderr, "config error: %v\n", err)
		return exitConfig
	}

	info := version.Get()
	logger.Info("llm-init starting",
		slog.String("version", info.Version),
		slog.String("commit", info.Commit),
		slog.String("engine", string(cfg.Engine.Kind)),
		slog.String("source", string(cfg.PrimarySourceKind())),
		slog.String("model", cfg.Model.Name),
		slog.Int("port", cfg.Runtime.Port),
	)

	if showConfig {
		fmt.Fprintf(stdout, "%+v\n", cfg.Redacted())
		return exitOK
	}

	ctx, cancel := signalContext()
	defer cancel()

	mgr := progress.New(time.Now())
	if cfg.Runtime.ProgressStatePath != "" {
		if err := mgr.LoadFrom(cfg.Runtime.ProgressStatePath); err != nil && !os.IsNotExist(err) {
			logger.Warn("progress state load failed; starting fresh",
				slog.String("path", cfg.Runtime.ProgressStatePath),
				slog.String("err", err.Error()))
		}
	}
	// Sanitise restored state for the new pod's lifetime.
	//
	// Background: progress is persisted to disk on graceful shutdown so
	// dashboards survive restarts. The trade-off is that a Pod that
	// crashed mid-download — or was OOM-killed by the kubelet — leaves
	// behind a `phase=failed` snapshot. When the next Pod boots it loads
	// that snapshot and the dashboard renders the prior failure until
	// the next Update fires.
	//
	// Reset rules for the new run:
	//   - Phase           → PhaseInit (the actual fresh-boot truth)
	//   - LastError       → cleared. v1.0.10-fix9 left it populated as
	//                       "audit trail", but operators reported the
	//                       resulting STATUS card looked like "we are
	//                       currently failing" all the way through the
	//                       new download — exactly the misread we
	//                       wanted to avoid in the first place. The
	//                       boot-time "previous run had failure …"
	//                       INFO log below preserves the same audit
	//                       value for grep / log search; the dashboard
	//                       gets a clean slate.
	//   - Bytes / Speed / ETA   → reset (volatile per-run, would
	//                     otherwise show "278 MB" residual on the
	//                     new run).
	//   - StartedAt     → now (this is when *this* Pod started).
	// Preserved across restarts because they are unambiguous counters
	// that don't read as "currently failing":
	//   - RetryCount, TransportRetries
	if loaded := mgr.Snapshot(); loaded.Phase != "" && loaded.Phase != progress.PhaseReady {
		// Surface the prior failure ONCE on boot so the audit trail
		// is grep-able even though the dashboard no longer renders
		// it. Done before mgr.Update so the snapshot we read still
		// reflects the persisted state.
		if loaded.LastError != "" {
			logger.Info("previous run had failure; reset for fresh boot",
				slog.String("prior_phase", string(loaded.Phase)),
				slog.String("last_error", loaded.LastError),
				slog.Int("retry_count", loaded.RetryCount),
				slog.Int("transport_retries", loaded.TransportRetries))
		}
		mgr.Update(func(s *progress.State) {
			s.Phase = progress.PhaseInit
			s.LastError = ""
			s.BytesTotal = 0
			s.BytesCompleted = 0
			s.SpeedBytesPerSec = 0
			s.ETASeconds = 0
			s.StartedAt = time.Now()
		})
	}
	// The file list names what the process that wrote it was
	// transferring, and this process is transferring nothing yet, so a
	// restored row reading `waiting` or `downloading` describes a
	// transfer that cannot resume. The block above does not catch it:
	// a snapshot saved as `ready` skips the reset entirely, which is
	// how two rows of a finished FireRedTTS3 download stayed
	// `downloading` on the dashboard across every restart.
	//
	// enterDownloadPhase re-seeds the list from the Hub listing when
	// this boot has something to fetch. When it has not, an empty list
	// is the honest answer and the aggregate bar still reports the
	// reconciled on-disk total.
	mgr.Update(func(s *progress.State) {
		s.Files = nil
		s.FilesTotal = 0
		s.FilesCompleted = 0
		s.CurrentFile = ""
	})

	metrics := obs.NewMetrics()
	// One-shot deprecation banner for renamed/removed metrics. Currently
	// a no-op (no deprecated metrics in v1.1); kept wired so the next
	// deprecation only needs an obs.LogDeprecations edit.
	obs.LogDeprecations()

	// Wire adapter → lifecycle. The adapter is a stateless wrapper;
	// lifecycle.New validates non-nil deps; dataplane.Mount reuses the
	// same Adapter reference. As of v1.1 lifecycle dispatches directly
	// off cfg.Sources, so there is no source.New step.
	ad, err := factory.New(cfg, metrics)
	if err != nil {
		fmt.Fprintf(stderr, "adapter init: %v\n", err)
		return exitConfig
	}
	// The restart tracker needs a live probe, not lc.Readiness: that
	// carries the 10s health loop's last verdict, and the dip it has to
	// catch can be shorter than one of its ticks.
	//
	// Built before lifecycle because the health loop reads it: a failing
	// probe means something different while a relaunch is outstanding,
	// and the loop is the only thing positioned to report that as a phase.
	//
	// Knowing about a relaunch is not enough on its own: the health loop
	// picks its next gap at the end of the current one, so a wait armed at
	// the steady interval just before the signal outlasts the whole
	// relaunch. Buffered by one and sent to without blocking, because
	// coalescing is the right answer to two signals in a row — the loop
	// only needs to reconsider once.
	relaunchWake := make(chan struct{}, 1)
	tracker := handoff.NewTracker(handoff.Config{
		RunDir: cfg.Runtime.RunDir,
		Probe: func(ctx context.Context) bool {
			rs, err := ad.Ready(ctx)
			return err == nil && rs.Alive
		},
		OnRequest: func() {
			select {
			case relaunchWake <- struct{}{}:
			default:
			}
		},
	})
	lc, err := lifecycle.New(lifecycle.Options{
		Config:                 cfg,
		Adapter:                ad,
		Manager:                mgr,
		Downloader:             fetch.New(nil),
		Metrics:                metrics,
		MaxConcurrentDownloads: cfg.Runtime.MaxConcurrentDownloads,
		MaxDownloadBytes:       cfg.Runtime.MaxDownloadBytes,
		RelaunchInFlight:       tracker.Relaunching,
		RelaunchRequested:      relaunchWake,
	})
	if err != nil {
		fmt.Fprintf(stderr, "lifecycle init: %v\n", err)
		return exitConfig
	}

	// Ollama-native metadata routes (Track Q.1, v1.0.9). Only wired
	// when the configured engine is Ollama: vLLM / llama.cpp / SGLang
	// expose their own native APIs upstream and llm-init has no
	// licence to translate them. Construct the registrar separately so
	// the closure captures a dedicated *ollama.Client (not the one
	// held by Adapter), keeping ollamanative independent of the
	// adapter's internals -- the package compiles in every binary
	// regardless of engine kind, but only attaches handlers when this
	// branch fires.
	var ollamaNativeMount controlplane.OllamaNativeRegistrar
	if cfg.Engine.Kind == config.EngineOllama {
		nativeClient := ollama.NewClient(cfg.Engine.URL)
		// Mirror the UPSTREAM_TRACE_LEVEL env onto the ollama-native
		// passthrough client so /api/tags, /api/ps, /api/show, /api/version
		// share the same per-call audit logging as the adapter's data plane.
		nativeClient.TraceLevel = ollama.TraceLevel(cfg.Log.UpstreamTrace)
		nativeClient.SetResponseHeaderTimeout(cfg.Runtime.ResponseHeaderTimeout())
		ollamaNativeMount = ollamanative.New(cfg, nativeClient, lc.Ready, mgr).Apply
	}

	// Everything assembled above holds cfg by value and stops seeing
	// changes to it; the control plane is the one component that must,
	// because it is where the model card is edited. From here on cfg is
	// the store's, and the boot copy above is only history.
	runtimeCfg := runtimecfg.New(cfg, tracker)

	// Download-only mode (empty ENGINE_KIND) has no engine to proxy, so
	// the /v1/* data plane is not mounted: the control plane serves its
	// 501 stub there while keeping livez/readyz/progress/config alive as
	// a sidecar. lc.Ready still flips once the download reaches
	// PhaseReady, so /readyz reflects download completion.
	//
	// Optional side surfaces that need the engine (e.g. translate) are
	// composed here so controlplane stays unaware of feature packages.
	var dataPlane controlplane.DataPlaneRegistrar
	if cfg.Engine.Kind != "" {
		dataPlane = func(m *http.ServeMux) {
			dataplane.Mount(m, dataplane.Options{
				Ready:   lc.Ready,
				Manager: mgr,
				Adapter: ad,
				Config:  cfg,
				Metrics: metrics,
				// Nil for every engine but llama.cpp, the only one that
				// lets its slots promise more of a shared pool than it
				// holds. Whether the flags currently say so is the
				// gate's own question, asked per request.
				KVBudget: dataplane.KVBudgetFor(dataplane.KVBudgetOptions{
					// Through the store, not the boot copy: a card edit
					// rewrites the launch flags and the engine is
					// relaunched under them, and both whether the pool
					// needs accounting and what a request may reserve
					// are read off those flags.
					Snapshot: runtimeCfg.Snapshot,
					// Through the store as well, because the pool size
					// worth accounting against is the one the engine
					// reported after it started -- which does not exist
					// yet here.
					PoolTokens: func() (int, bool) {
						return enginecapacity.PoolTokensOf(runtimeCfg.Snapshot())
					},
				}),
			})
			if cfg.Model.Type == config.ModelTranslate {
				translate.Mount(m, translate.Options{
					Wrap: func(h http.Handler) http.Handler {
						return dataplane.NotReadyGuard(lc.Ready, mgr, h)
					},
					Completer: translate.Completer{
						Handler:   ad.OpenAIHandler(cfg),
						ModelName: cfg.Model.Name,
					},
					Extensions:  cfg.Spec.Extensions,
					ContextSize: cfg.Spec.ContextSize,
				})
			}
		}
	}

	server := controlplane.NewServer(controlplane.Options{
		Manager:        mgr,
		Metrics:        metrics,
		Config:         runtimeCfg,
		Version:        info,
		RetryRateLimit: cfg.Runtime.RetryRateLimit,
		Readiness: func() controlplane.Readiness {
			r := lc.Readiness()
			return controlplane.Readiness{
				Phase:       r.Phase,
				EngineAlive: r.EngineAlive,
				ModelExists: r.ModelExists,
				Ready:       r.Ready,
				Reason:      r.Reason,
			}
		},
		Retry: func(_ context.Context, opts controlplane.RetryOptions) error {
			// Translate the controlplane shape into the lifecycle one.
			// They are kept as separate types so lifecycle does not
			// depend on controlplane (and the import graph stays a DAG).
			lc.RetryWith(lifecycle.RetryOptions{
				Force: opts.Force,
				Level: opts.Level,
			})
			return nil
		},
		DataPlane: dataPlane,
		Diag: func(m *http.ServeMux) {
			diag.Mount(m, &diag.Handler{
				Adapter: ad,
				Config:  cfg,
				Metrics: metrics,
			})
		},
		OllamaNative: ollamaNativeMount,
	})

	addr := ":" + strconv.Itoa(cfg.Runtime.Port)
	httpServer := buildHTTPServer(addr, server)

	errCh := make(chan error, 1)
	go func() {
		logger.Info("control plane listening", slog.String("addr", addr))
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
			return
		}
		errCh <- nil
	}()

	// lifecycle runs in its own goroutine until ctx is cancelled. Run
	// returns ctx.Err on shutdown (expected) or a permanent failure
	// (already reflected in PhaseFailed; logged for the operator).
	lcCtx, lcCancel := context.WithCancel(ctx)
	lcDone := make(chan struct{})
	go func() {
		defer close(lcDone)
		if err := lc.Run(lcCtx); err != nil &&
			!errors.Is(err, context.Canceled) &&
			!errors.Is(err, context.DeadlineExceeded) {
			logger.Error("lifecycle terminated", slog.String("err", err.Error()))
		}
	}()

	// Mirror progress.State updates into Prometheus collectors. Pre-v1.0.3
	// `llm_init_phase`, `llm_init_download_*` were declared but never
	// updated after process start; this subscriber is the producer.
	go obs.PhaseAndDownloadMetricsSubscriber(
		lcCtx, mgr, metrics, controlplane.AllPhases, string(cfg.PrimarySourceKind()),
	)

	// Ask the engine what it can actually hold once it is up, and put the
	// answer on the card for Router. The launch flags are what this
	// process was told; three of the four engines resolve them downwards
	// against the hardware without saying so, and a caller deciding
	// whether to admit a request needs the resolved number. Skipped in
	// download-only mode, which has no engine to ask.
	if cfg.Engine.Kind != "" {
		reporter := &enginecapacity.Reporter{
			// Read through the store, not the boot copy: a card edit
			// rewrites the launch flags and restarts the engine, and the
			// next reading has to be compared against the new ones.
			Target: func() enginecapacity.Options {
				snap := runtimeCfg.Snapshot()
				return enginecapacity.Options{
					Kind:                     snap.Engine.Kind,
					Args:                     snap.Engine.Args,
					EngineURL:                snap.Engine.URL,
					ConfiguredMaxConcurrency: snap.Engine.MaxConcurrency,
					Model:                    snap.Model.Name,
				}
			},
			Ready: lc.Ready,
			// The relaunch a card edit causes happens inside the health
			// loop's grace period, so readiness alone would leave this
			// reporting the engine that has just been replaced.
			Restarted: runtimeCfg.Handoff().Relaunch,
			Record:    runtimeCfg.RecordCapacity,
		}
		go reporter.Run(lcCtx)
	}

	select {
	case <-ctx.Done():
		logger.Info("received shutdown signal, draining")
	case err := <-errCh:
		if err != nil {
			logger.Error("control plane terminated", slog.String("err", err.Error()))
			lcCancel()
			<-lcDone
			return exitControlPlane
		}
		lcCancel()
		<-lcDone
		return exitOK
	}

	// Cancel lifecycle first so it stops touching the engine before we
	// shut down the HTTP server (which otherwise would leave dangling
	// in-flight /v1/* requests).
	lcCancel()
	<-lcDone

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), shutdownGracePeriod)
	defer shutdownCancel()
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		logger.Warn("graceful shutdown failed", slog.String("err", err.Error()))
	}

	if cfg.Runtime.ProgressStatePath != "" {
		if err := mgr.SaveTo(cfg.Runtime.ProgressStatePath); err != nil {
			logger.Warn("progress state save failed",
				slog.String("path", cfg.Runtime.ProgressStatePath),
				slog.String("err", err.Error()))
		}
	}

	logger.Info("exited")
	return exitOK
}

// signalContext returns a context cancelled on SIGINT or SIGTERM.
func signalContext() (context.Context, context.CancelFunc) {
	return signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
}

// buildHTTPServer constructs the control-plane http.Server with the
// timeout policy used by run(). Extracted so unit tests can assert on
// the timeout fields without standing the whole process up.
//
//   - ReadHeaderTimeout: 10s — bounds slowloris on the request line +
//     headers without affecting bodies (multipart uploads, future
//     tools API).
//   - ReadTimeout / WriteTimeout: deliberately unset — /v1/* streams
//     SSE responses for several minutes for long completions; either
//     timeout would truncate the stream. Slowloris on the body itself
//     is mitigated by the 1 MiB body cap inside the adapter handlers.
//   - IdleTimeout: 120s — only fires when no request is in flight, so
//     it caps the cost of kept-alive idle connections without ever
//     truncating an active stream. Matches the proxy adapter's
//     upstream IdleConnTimeout so client and upstream idle connections
//     are reaped on similar schedules.
func buildHTTPServer(addr string, handler http.Handler) *http.Server {
	return &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
	}
}

// doHealthcheck performs a one-shot HTTP GET to the local /livez endpoint
// with a 2s timeout. Used by Dockerfile HEALTHCHECK and docker-compose
// healthcheck so that distroless images (which have no shell) can still be
// probed. Returns exitOK on a 2xx, exitHealthcheckFail otherwise.
//
// PORT is read from the injected getenv with the same default (8080) as
// config.Load, so the probe and the server agree without sharing config.
func doHealthcheck(getenv config.Getenv, stdout, stderr *os.File) int {
	port := getenv("PORT")
	if port == "" {
		port = "8080"
	}
	url := "http://127.0.0.1:" + port + "/livez"
	client := &http.Client{Timeout: healthcheckTimeout}
	resp, err := client.Get(url)
	if err != nil {
		fmt.Fprintf(stderr, "healthcheck: %v\n", err)
		return exitHealthcheckFail
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		fmt.Fprintf(stderr, "healthcheck: status %d\n", resp.StatusCode)
		return exitHealthcheckFail
	}
	fmt.Fprintln(stdout, "ok")
	return exitOK
}

// newLogger builds a slog.Logger honouring LOG_LEVEL and LOG_FORMAT, wrapped
// with obs.RedactingHandler so HF / Bearer / sk- tokens never escape.
func newLogger(level, format string, w *os.File) *slog.Logger {
	var lvl slog.Level
	switch level {
	case "debug":
		lvl = slog.LevelDebug
	case "warn":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}
	opts := &slog.HandlerOptions{Level: lvl}
	var inner slog.Handler
	if format == config.LogFormatText {
		inner = slog.NewTextHandler(w, opts)
	} else {
		inner = slog.NewJSONHandler(w, opts)
	}
	return slog.New(obs.NewRedactingHandler(inner))
}
