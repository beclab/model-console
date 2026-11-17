package lifecycle

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"path/filepath"
	"strings"
	"time"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/handoff"
	"github.com/llm-init/llm-init/internal/sentinel"
)

// Relauncher asks the engine's supervise loop to relaunch it and reports
// what became of the request. *handoff.Tracker implements it.
type Relauncher interface {
	Request() (handoff.Receipt, error)
	Status() handoff.Status
}

// relaunchPollInterval is how often a pass looks at the tracker while
// waiting for its verdict on a relaunch.
const relaunchPollInterval = 500 * time.Millisecond

// loadedModel is what the run dir currently tells an engine to load:
// model_path and the extra paths, or "" when there is no model_path.
func loadedModel(runDir string) string {
	if runDir == "" {
		return ""
	}
	path, err := sentinel.ReadModelPath(runDir)
	if err != nil || path == "" {
		return ""
	}
	extras, _ := sentinel.ReadExtraPaths(runDir)
	return strings.Join(append([]string{path}, extras...), "\n")
}

// errEngineServesPreviousModel is why a pass that replaced the model is
// not ready: the engine was not seen restarting onto the new one.
var errEngineServesPreviousModel = errors.New("the engine still serves the previous model; restart it to load the new one")

// relaunchIfModelChanged restarts an engine that serves a model other
// than the one the run dir names, and fails when the engine was not seen
// to restart.
//
// A wrapper waits for the sentinel once and keeps the engine it started,
// and an engine that resolves the model by name (llama.cpp -hf) picks the
// revision when it starts. On an upgrade that engine started on the
// previous sentinel before the new revision was downloaded, so it serves
// the previous model, and WaitAlive would pass on it.
//
// An engine that reports its model path (adapter.ServedModelReporter) is
// asked directly, which holds across an llm-init restart and a restart
// done by hand. Otherwise, and when only an extra such as the projector
// changed, the run dir as the pass found it is compared with what it
// wrote, and a failed restart stays pending in m.pendingRelaunch: the run
// dir already names the new model, so the next pass would otherwise see
// no change. A first install has no previous model and an engine still
// waiting for the sentinel, so it is left alone. StateUnverified is left
// to WaitAlive, since it says nothing about which model is served.
func (m *Manager) relaunchIfModelChanged(ctx context.Context) error {
	p := m.pass
	if p == nil || m.opts.Relauncher == nil {
		return nil
	}
	current := loadedModel(p.cfg.Runtime.RunDir)
	if current == "" {
		return nil
	}
	if served, ok := m.servedModelPath(ctx); ok {
		if !sameModelFile(served, mainModel(current)) {
			if err := m.relaunch(ctx, served, current); err != nil {
				return err
			}
			m.pendingRelaunch = ""
			return nil
		}
		if !extrasOnlyChange(p.prevModel, current) {
			m.pendingRelaunch = ""
			return nil
		}
	}
	if p.prevModel == "" || current == p.prevModel {
		m.pendingRelaunch = ""
		return nil
	}
	if err := m.relaunch(ctx, p.prevModel, current); err != nil {
		m.pendingRelaunch = p.prevModel
		return err
	}
	m.pendingRelaunch = ""
	return nil
}

// relaunch asks the engine to restart onto current and waits for the
// tracker's verdict.
func (m *Manager) relaunch(ctx context.Context, previous, current string) error {
	receipt, err := m.opts.Relauncher.Request()
	if err != nil {
		slog.Warn("the model changed but the engine could not be asked to restart; it still serves the previous model",
			slog.String("previous", previous),
			slog.String("current", current),
			slog.String("err", err.Error()))
		return fmt.Errorf("%w: %w", errEngineServesPreviousModel, err)
	}
	slog.Info("the model changed; restarting the engine so it loads the new one",
		slog.String("previous", previous),
		slog.String("current", current),
		slog.String("generation", receipt.Generation))
	st := m.awaitRelaunchVerdict(ctx, receipt.Generation)
	if err := ctx.Err(); err != nil {
		return err
	}
	if st.State == handoff.StateNoSupervisor {
		slog.Warn("the engine did not restart after the model changed and still serves the previous model; restart it to load the new one",
			slog.String("previous", previous),
			slog.String("current", current))
		return errEngineServesPreviousModel
	}
	m.pass.relaunched = true
	return nil
}

// servedModelAttempts is how many times verifyServedModel asks an alive
// engine which model it loaded before letting it through unanswered.
const servedModelAttempts = 6

// verifyServedModel asks an alive engine which model it loaded, whether
// or not the pass restarted it: a query before WaitAlive can fail while
// the engine loads, which hides an engine still on the previous model.
// One that serves another model than the run dir's is restarted once
// within the pass, and the pass fails when it comes back on another
// model again.
//
// An engine that does not answer after servedModelAttempts is let
// through, since WaitAlive passed and a llama.cpp build without /props
// would otherwise never be ready.
func (m *Manager) verifyServedModel(ctx context.Context) error {
	p := m.pass
	if p == nil || m.opts.Relauncher == nil {
		return nil
	}
	if _, ok := m.opts.Adapter.(adapter.ServedModelReporter); !ok {
		return nil
	}
	current := loadedModel(p.cfg.Runtime.RunDir)
	want := mainModel(current)
	if want == "" {
		return nil
	}
	for {
		served, ok := m.awaitServedModelPath(ctx)
		if err := ctx.Err(); err != nil {
			return err
		}
		if !ok {
			slog.Warn("the engine is alive but did not say which model it serves; calling it ready",
				slog.String("want", want))
			return nil
		}
		if sameModelFile(served, want) {
			return nil
		}
		if p.relaunched {
			slog.Warn("the engine restarted but loaded a model other than the one downloaded",
				slog.String("served", served),
				slog.String("want", want))
			return fmt.Errorf("the engine restarted on %s, not %s", served, want)
		}
		if err := m.relaunch(ctx, served, current); err != nil {
			return err
		}
		if err := m.opts.Adapter.WaitAlive(ctx); err != nil {
			return fmt.Errorf("engine never came alive: %w", err)
		}
	}
}

// awaitServedModelPath is servedModelPath retried every
// ServedModelRetryInterval, up to servedModelAttempts times.
func (m *Manager) awaitServedModelPath(ctx context.Context) (string, bool) {
	for attempt := 1; ; attempt++ {
		if served, ok := m.servedModelPath(ctx); ok {
			return served, true
		}
		if attempt == servedModelAttempts {
			return "", false
		}
		select {
		case <-ctx.Done():
			return "", false
		case <-time.After(m.opts.ServedModelRetryInterval):
		}
	}
}

// servedModelPath asks the engine which model file it loaded. ok is false
// when the engine cannot say.
func (m *Manager) servedModelPath(ctx context.Context) (string, bool) {
	r, ok := m.opts.Adapter.(adapter.ServedModelReporter)
	if !ok {
		return "", false
	}
	path, err := r.ServedModelPath(ctx)
	if err != nil || path == "" {
		return "", false
	}
	return path, true
}

// mainModel is the model_path line of a loadedModel value.
func mainModel(loaded string) string {
	main, _, _ := strings.Cut(loaded, "\n")
	return main
}

// extrasOnlyChange reports whether the pass changed an extra path but
// kept the main model, which an engine's own report cannot show.
func extrasOnlyChange(prev, current string) bool {
	return prev != "" && prev != current && mainModel(prev) == mainModel(current)
}

// sameModelFile compares two paths to a model file. Paths into an HF
// cache are compared from the models--<org>--<repo> directory on, since
// the engine and llm-init may mount the cache at different roots.
func sameModelFile(a, b string) bool {
	a, b = filepath.ToSlash(filepath.Clean(a)), filepath.ToSlash(filepath.Clean(b))
	if a == b {
		return true
	}
	ia, ib := strings.LastIndex(a, "/models--"), strings.LastIndex(b, "/models--")
	return ia >= 0 && ib >= 0 && a[ia:] == b[ib:]
}

// awaitRelaunchVerdict waits until the tracker stops watching the request
// it was given: the engine went down, or the window closed without it.
// WaitAlive then waits for the relaunched engine rather than passing on
// the one about to be stopped.
func (m *Manager) awaitRelaunchVerdict(ctx context.Context, generation string) handoff.Status {
	for {
		st := m.opts.Relauncher.Status()
		if st.Generation != generation || st.State != handoff.StateWatching {
			return st
		}
		select {
		case <-ctx.Done():
			return st
		case <-time.After(relaunchPollInterval):
		}
	}
}
