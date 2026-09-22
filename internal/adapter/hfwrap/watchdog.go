package hfwrap

import (
	"context"
	"log/slog"
	"os/exec"
	"sync"
	"sync/atomic"
	"time"
)

// defaultActivityCheckInterval is how often the watchdog re-checks the
// last-activity timestamp in production.
const defaultActivityCheckInterval = 30 * time.Second

// defaultActivityTimeout is the inactivity window after which the
// watchdog kills the subprocess. 10 minutes matches download-server's
// value; long enough for cold xet caches to warm up but short enough
// that a stuck connection eventually frees the lifecycle to retry.
const defaultActivityTimeout = 10 * time.Minute

// Watchdog tracks subprocess activity and SIGKILLs the process when no
// stderr line has been seen for activityTimeout.
//
// Touch() is safe for concurrent use; goroutines pumping stderr call it
// for every parsed line. Stop() must be invoked once the subprocess has
// exited so the goroutine can release the timer.
type Watchdog struct {
	cmd           *exec.Cmd
	repo          string
	checkInterval time.Duration
	timeout       time.Duration
	lastSeen      atomic.Int64

	stopOnce sync.Once
	done     chan struct{}
	killed   atomic.Bool
}

// NewWatchdog wires a Watchdog to cmd with the production timings. ctx
// cancellation also stops the goroutine; cmd is the just-Started
// exec.Cmd so the watchdog can Process.Kill on inactivity. repo is the
// human label written into the kill log so multi-source deploys can
// pinpoint which download stalled.
func NewWatchdog(ctx context.Context, cmd *exec.Cmd, repo string) *Watchdog {
	return newWatchdog(ctx, cmd, repo, defaultActivityCheckInterval, defaultActivityTimeout)
}

// newWatchdog is the timing-parameterised constructor. The timings are
// stored per-instance (not in package globals) so concurrent tests can
// use different values without racing each other or the production
// watchdogs spawned by the runner.
func newWatchdog(ctx context.Context, cmd *exec.Cmd, repo string, checkInterval, timeout time.Duration) *Watchdog {
	w := &Watchdog{
		cmd:           cmd,
		repo:          repo,
		checkInterval: checkInterval,
		timeout:       timeout,
		done:          make(chan struct{}),
	}
	w.lastSeen.Store(time.Now().UnixNano())
	go w.loop(ctx)
	return w
}

// Touch marks the subprocess as active "right now". The runner's
// stderr pump calls this for every non-empty parsed line.
func (w *Watchdog) Touch() {
	w.lastSeen.Store(time.Now().UnixNano())
}

// Stop releases the watchdog goroutine. Idempotent; safe to defer
// alongside Run's wait/cleanup.
func (w *Watchdog) Stop() {
	w.stopOnce.Do(func() {
		close(w.done)
	})
}

// Killed reports whether Watchdog killed the subprocess due to
// inactivity. The caller (Run) uses this to bias error classification
// toward CodeProcessKilled / retriable.
func (w *Watchdog) Killed() bool {
	return w.killed.Load()
}

func (w *Watchdog) loop(ctx context.Context) {
	t := time.NewTicker(w.checkInterval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-w.done:
			return
		case now := <-t.C:
			last := time.Unix(0, w.lastSeen.Load())
			if now.Sub(last) >= w.timeout {
				w.kill()
				return
			}
		}
	}
}

func (w *Watchdog) kill() {
	if w.killed.Swap(true) {
		return
	}
	if w.cmd == nil || w.cmd.Process == nil {
		return
	}
	slog.Warn("hfwrap watchdog: killing inactive subprocess",
		slog.String("repo", w.repo),
		slog.Duration("inactivity_threshold", w.timeout))
	_ = w.cmd.Process.Kill()
}
