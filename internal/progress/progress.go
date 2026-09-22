// Package progress is the central state machine for llm-init.
//
// A single Manager instance is shared by lifecycle, fetch, adapter and the
// control plane. Producers call Update or push events through a Sink; the
// control plane reads via Snapshot and Subscribe. The data model is defined
// returned by the /api/progress endpoint.
package progress

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Phase enumerates lifecycle states from process start to ready/failed.
type Phase string

// Phase values. Order matches the state machine drawn in
// the lifecycle state machine;
// transitions PhaseInit -> Download -> Loading -> Ready are the happy path,
// with Degraded/Failed as terminal-ish branches. Loading spans the window
// between "model fully downloaded + registered" and "engine reports alive":
// for proxy engines (vLLM/sglang/llama.cpp) it covers the engine cold start;
// any failure in Loading is terminal (PhaseFailed), not retriable Degraded.
const (
	PhaseInit     Phase = "init"
	PhaseDownload Phase = "download"
	PhaseLoading  Phase = "loading"
	PhaseReady    Phase = "ready"
	PhaseDegraded Phase = "degraded"
	PhaseFailed   Phase = "failed"
)

// FileStatus is the per-file row on the download card.
type FileStatus string

// FileStatus values for ProgressState.files[]. Waiting is seeded from
// the Hub tree before any transfer; downloading / done follow OnFileStart
// and OnFileDone.
const (
	FileWaiting     FileStatus = "waiting"
	FileDownloading FileStatus = "downloading"
	FileDone        FileStatus = "done"
)

// FileProgress is one file this pass will fetch. Seeded from the Hub
// tree so the dashboard can list waiting files, not only the one in
// flight.
type FileProgress struct {
	Path           string     `json:"path"`
	BytesTotal     int64      `json:"bytes_total,omitempty"`
	BytesCompleted int64      `json:"bytes_completed,omitempty"`
	Status         FileStatus `json:"status"`
}

// State is a JSON-serialisable snapshot consumed by /api/progress and the UI.
type State struct {
	Phase            Phase     `json:"phase"`
	StartedAt        time.Time `json:"started_at"`
	UpdatedAt        time.Time `json:"updated_at"`
	BytesTotal       int64     `json:"bytes_total"`
	BytesCompleted   int64     `json:"bytes_completed"`
	SpeedBytesPerSec float64   `json:"speed_bytes_per_sec"`
	ETASeconds       int64     `json:"eta_seconds"`
	// FilesTotal / FilesCompleted describe how many files this download
	// pass will fetch. Zero means the producer has not announced a
	// count yet (single-URL streams often never do). When both are
	// set, the dashboard treats bytes_completed >= bytes_total as
	// finished only if every file is accounted for — otherwise a
	// multi-file HF pass looks done after the first file.
	FilesTotal     int    `json:"files_total,omitempty"`
	FilesCompleted int    `json:"files_completed,omitempty"`
	CurrentFile    string `json:"current_file,omitempty"`
	// Files is the per-file queue for this pass. Seeded from the Hub
	// tree (or exact --include / URL names) so the dashboard can list
	// waiting files before the first byte moves. Empty when the
	// producer has not announced names (typical for a single URL
	// before HEAD, or a tree that failed).
	Files []FileProgress `json:"files,omitempty"`
	// LastError is the current failure message shown on the Status
	// diagnostics card. Cleared when lifecycle re-enters download or
	// reaches ready; not an audit log (use RetryCount /
	// TransportRetries for history).
	LastError string `json:"last_error"`
	// RetryCount counts ensure-loop retry attempts (lifecycle.fail or
	// operator-triggered RetryWith). Pre-v1.0.5 it conflated transport-
	// level retries (every retried HTTP attempt inside fetch's inner
	// loop) with operator-visible retries; the new TransportRetries
	// field below isolates the inner-loop count so RetryCount stays a
	// stable signal of "the lifecycle has tried again".
	RetryCount int `json:"retry_count"`
	// TransportRetries counts transport-level retries observed via
	// Sink.OnError -- 503 / EOF / connection-reset attempts that
	// fetch's inner retry loop chose to repeat. A noisy network bumps
	// this freely; a healthy run leaves it at 0. Operators looking
	// for "is something wrong with the upstream?" should watch this;
	// "did llm-init fail and retry the whole pass?" remains
	// RetryCount.
	TransportRetries int `json:"transport_retries"`

	// LastVerifyAt is the wall-clock timestamp of the most recent
	// verification of the bytes on disk. Nothing runs one on a timer, so it
	// only moves on startup, on POST /api/retry, and on the health loop's
	// re-registration. Zero value means "no verify has run yet".
	LastVerifyAt time.Time `json:"last_verify_at,omitempty"`

	// LastVerifyOK is the result of the most recent verification: nil if
	// none yet, *true if Diff was clean, *false if Diff reported missing
	// or stale files. Pointer-bool so consumers can distinguish "no data"
	// from "verify ran and disagreed".
	LastVerifyOK *bool `json:"last_verify_ok,omitempty"`

	// EngineColdStartMS is the wall-clock duration (in milliseconds)
	// between StartedAt (lifecycle process start) and the first
	// PhaseReady transition. Captured ONCE per process start;
	// subsequent retries / repairs / degrade-then-recover cycles do
	// not overwrite it. nil = "engine has never been ready in this
	// process".
	//
	// Surfaces in /api/progress for operator dashboards; clients
	// should treat null as "still booting" and a positive int as an
	// immutable boot stat.
	EngineColdStartMS *int64 `json:"engine_cold_start_ms,omitempty"`

	// ResolvedCommit is the 40-char HF commit SHA the active HF
	// download has resolved against (empty string for non-HF sources
	// or before the first ref resolution).
	ResolvedCommit string `json:"resolved_commit,omitempty"`

	// Note is a short operator-facing message attached to the
	// progress card (e.g. "xet acceleration: tqdm progress
	// unavailable"). Empty string clears the banner.
	Note string `json:"note,omitempty"`
}

// MarshalJSON omits LastVerifyAt when it is the Go zero time. encoding/json's
// omitempty on time.Time is supposed to do this already, but persisted
// progress_state.json from older builds (or hand-edited files) can carry an
// explicit "0001-01-01T00:00:00Z" that browsers render as "1/1/1" with a
// nonsense "Nd ago". Using a pointer in the wire shape makes the omission
// deterministic for SSE /api/progress frames.
func (s State) MarshalJSON() ([]byte, error) {
	type stateWire struct {
		Phase             Phase          `json:"phase"`
		StartedAt         time.Time      `json:"started_at"`
		UpdatedAt         time.Time      `json:"updated_at"`
		BytesTotal        int64          `json:"bytes_total"`
		BytesCompleted    int64          `json:"bytes_completed"`
		SpeedBytesPerSec  float64        `json:"speed_bytes_per_sec"`
		ETASeconds        int64          `json:"eta_seconds"`
		FilesTotal        int            `json:"files_total,omitempty"`
		FilesCompleted    int            `json:"files_completed,omitempty"`
		CurrentFile       string         `json:"current_file,omitempty"`
		Files             []FileProgress `json:"files,omitempty"`
		LastError         string         `json:"last_error"`
		RetryCount        int            `json:"retry_count"`
		TransportRetries  int            `json:"transport_retries"`
		LastVerifyAt      *time.Time     `json:"last_verify_at,omitempty"`
		LastVerifyOK      *bool          `json:"last_verify_ok,omitempty"`
		EngineColdStartMS *int64         `json:"engine_cold_start_ms,omitempty"`
		ResolvedCommit    string         `json:"resolved_commit,omitempty"`
		Note              string         `json:"note,omitempty"`
	}
	out := stateWire{
		Phase:             s.Phase,
		StartedAt:         s.StartedAt,
		UpdatedAt:         s.UpdatedAt,
		BytesTotal:        s.BytesTotal,
		BytesCompleted:    s.BytesCompleted,
		SpeedBytesPerSec:  s.SpeedBytesPerSec,
		ETASeconds:        s.ETASeconds,
		FilesTotal:        s.FilesTotal,
		FilesCompleted:    s.FilesCompleted,
		CurrentFile:       s.CurrentFile,
		Files:             s.Files,
		LastError:         s.LastError,
		RetryCount:        s.RetryCount,
		TransportRetries:  s.TransportRetries,
		LastVerifyOK:      s.LastVerifyOK,
		EngineColdStartMS: s.EngineColdStartMS,
		ResolvedCommit:    s.ResolvedCommit,
		Note:              s.Note,
	}
	if !s.LastVerifyAt.IsZero() {
		t := s.LastVerifyAt.UTC()
		out.LastVerifyAt = &t
	} else {
		// Drop orphaned verify-ok flags when no real timestamp exists.
		out.LastVerifyOK = nil
	}
	return json.Marshal(out)
}

// Manager owns the canonical State and broadcasts updates to subscribers.
//
// All methods are safe for concurrent use.
type Manager interface {
	// Snapshot returns a value copy; callers may mutate the returned struct
	// without affecting the manager.
	Snapshot() State

	// Update applies fn under the manager's lock and broadcasts the new
	// State to subscribers (non-blocking; slow subscribers may miss frames).
	Update(fn func(*State))

	// Subscribe registers a buffered channel for state updates. The returned
	// unsubscribe func must be called when the consumer is done; it is safe
	// to call multiple times.
	Subscribe() (<-chan State, func())

	// Sink returns a Sink whose method calls translate to Update calls.
	Sink() Sink

	// SeedDownloadBudget pins bytes_total and optional bytes_completed
	// (URL .part resume / finished HF snapshot) for a download pass.
	// While active, Sink.OnFileStart does not grow bytes_total.
	SeedDownloadBudget(total, completed int64)

	// SeedDownloadFiles pins files_total and optional files_completed
	// (finished HF snapshot files) for a multi-file pass. While active,
	// OnFileStart does not grow files_total. OnFilesTotal never shrinks
	// the pin; it is a hint, not a lock, when no seed is set.
	SeedDownloadFiles(total, completed int)

	// SeedDownloadFileList publishes the named queue for this pass
	// (waiting / already-on-disk) and pins files_total to its length.
	// OnFileStart matches a row by path or basename instead of
	// appending a duplicate.
	SeedDownloadFileList(files []FileProgress)

	// ClearDownloadBudget ends pinned-total mode when a new download pass
	// resets counters (setPhase download).
	ClearDownloadBudget()

	// SaveTo writes the current Snapshot as JSON to path (atomic rename).
	SaveTo(path string) error

	// LoadFrom reads JSON from path and replaces the current State.
	// os.ErrNotExist is returned when path does not exist.
	LoadFrom(path string) error
}

// emaAlpha is the smoothing factor for the EMA download speed; 0.2 reaches
// ~95% of a steady value within ~14 samples. Lower than the original 0.3
// so the per-window EMA does a touch more secondary smoothing on top of
// the minSampleInterval averaging below.
const emaAlpha = 0.2

// minSampleInterval is the floor on the speed-sampling window. recomputeRate
// only recomputes instant rate once this much wall-clock has elapsed since
// the last sample, so each instant = delta/dt is averaged over >=1s rather
// than over the bursty sub-second cadence of concurrent/chunked downloads
// (which made the displayed speed jump). Matches the dashboard's 1s poll.
const minSampleInterval = 1 * time.Second

// minRateFileBytes is the size at which a file is treated as
// bandwidth-bound. Smaller files are handshake-bound: their window
// rate must not become SpeedBytesPerSec or the card shows 4 KB/s and
// a multi-hundred-hour ETA against a pinned multi-GB total.
const minRateFileBytes int64 = 8 << 20

// minRateUnknownDelta accepts a sample when OnFileStart never learned
// a size, but this window itself moved a mebibyte — enough that it is
// not three config files.
const minRateUnknownDelta int64 = 1 << 20

// fileQualifiesForRate reports whether fileTotal is large enough that
// a one-second window of it measures bandwidth, not RTT. A file that
// is already ≥ 1% of the pinned total also qualifies (a 5 MiB single
// GGUF is the whole download).
func fileQualifiesForRate(fileTotal, bytesTotal int64) bool {
	if fileTotal >= minRateFileBytes {
		return true
	}
	if fileTotal > 0 && bytesTotal > 0 && fileTotal >= bytesTotal/100 {
		return true
	}
	return false
}

// rateSampleEligible is the gate recomputeRate uses before writing
// SpeedBytesPerSec. Small current files are skipped; an unknown file
// size is accepted only when the window itself was bulky.
func rateSampleEligible(fileTotal, bytesTotal, windowDelta int64) bool {
	if fileQualifiesForRate(fileTotal, bytesTotal) {
		return true
	}
	return fileTotal <= 0 && windowDelta >= minRateUnknownDelta
}

// subBufferSize is the per-subscriber channel depth. Updates that find the
// channel full are dropped (subscribers see fewer frames but never stale ones).
const subBufferSize = 8

// New returns a default in-memory Manager initialised with PhaseInit and the
// supplied "now" timestamp (use time.Now to start the clock).
func New(now time.Time) Manager {
	return &memoryManager{
		state: State{
			Phase:     PhaseInit,
			StartedAt: now,
			UpdatedAt: now,
		},
		subs:       make(map[*subscription]struct{}),
		lastBytes:  0,
		lastSample: now,
		seenFiles:  make(map[string]struct{}),
		doneFiles:  make(map[string]struct{}),
	}
}

type subscription struct {
	ch chan State
}

type memoryManager struct {
	mu         sync.RWMutex
	state      State
	subs       map[*subscription]struct{}
	lastBytes  int64
	lastSample time.Time
	// bytesBudgetActive: OnFileStart must not add to BytesTotal (set by
	// SeedDownloadBudget for multi-file passes).
	bytesBudgetActive bool
	// filesBudgetActive: OnFileStart must not raise FilesTotal
	// (set only by SeedDownloadFiles). OnFilesTotal never sets this:
	// a CLI "Fetching N files" is one batch, not the whole pass.
	filesBudgetActive bool
	// seenFiles de-dupes OnFileStart path labels when the file count
	// is growing dynamically (no seed). FilesTotal is max(hint, seen).
	seenFiles map[string]struct{}
	// doneFiles de-dupes OnFileDone so a repeated label does not
	// increment files_completed twice.
	doneFiles map[string]struct{}
	// rateFilePath / rateFileBytes are the file OnFileStart last
	// opened. size top-ups pass a delta, so we only raise the stored
	// size, never replace a full total with a top-up.
	rateFilePath  string
	rateFileBytes int64
}

func (m *memoryManager) Snapshot() State {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.state
}

func (m *memoryManager) Update(fn func(*State)) {
	// Hold the write lock across the send loop too. The non-blocking
	// `select default` keeps each send to O(µs); the cost of holding
	// over N subscribers is bounded by N (10s typical, single-digit
	// hundreds at worst on the operator UI). The benefit is that
	// `unsubscribe` -- which closes s.ch -- cannot interleave between
	// our snapshot of the subs list and the actual send. Without the
	// hold, the race detector caught a real "send to closed channel"
	// at chansend / closechan during the v1.0.5 first-CI-run lane:
	// `go test -race` flagged the lifecycle goroutine sending into a
	// channel the obs.PhaseAndDownloadMetricsSubscriber unsubscribe
	// path had just closed.
	m.mu.Lock()
	defer m.mu.Unlock()
	old := m.state
	fn(&m.state)
	m.state.UpdatedAt = time.Now()
	m.recomputeRate(old)
	snapshot := m.state
	for s := range m.subs {
		select {
		case s.ch <- snapshot:
		default:
			// Slow subscriber: drop this frame. The next update will be
			// delivered if buffer is drained by then; subscribers should
			// also re-fetch /api/progress on reconnect.
		}
	}
}

// recomputeRate updates SpeedBytesPerSec (EMA) and ETASeconds based on the
// delta between successive samples. Called under m.mu.
func (m *memoryManager) recomputeRate(old State) {
	now := m.state.UpdatedAt
	dt := now.Sub(m.lastSample).Seconds()
	if dt < minSampleInterval.Seconds() {
		// Sampling window not full yet: do NOT advance lastBytes/lastSample,
		// so the bytes keep accumulating and instant = delta/dt is computed
		// over a >=1s window once it elapses (smooths concurrent-chunk bursts
		// and stops zero-delta events from dragging the rate down). Only
		// refresh ETA from the current known rate to keep the snapshot fresh.
		m.refreshETA()
		return
	}
	delta := m.state.BytesCompleted - m.lastBytes
	if delta < 0 {
		// BytesCompleted decreased (e.g. retry from zero). Reset baseline
		// and skip this sample.
		m.lastBytes = m.state.BytesCompleted
		m.lastSample = now
		m.state.SpeedBytesPerSec = 0
		m.refreshETA()
		return
	}
	// Always close the window so handshake time is not folded into
	// the first bulk sample. Only bandwidth-bound windows update
	// the displayed rate; a skipped window keeps the last good
	// speed (or 0 before any).
	if rateSampleEligible(m.rateFileBytes, m.state.BytesTotal, delta) {
		instant := float64(delta) / dt
		if old.SpeedBytesPerSec == 0 {
			m.state.SpeedBytesPerSec = instant
		} else {
			m.state.SpeedBytesPerSec = emaAlpha*instant + (1-emaAlpha)*old.SpeedBytesPerSec
		}
	}
	m.lastBytes = m.state.BytesCompleted
	m.lastSample = now
	m.refreshETA()
}

func (m *memoryManager) refreshETA() {
	remaining := m.state.BytesTotal - m.state.BytesCompleted
	if remaining <= 0 || m.state.SpeedBytesPerSec <= 0 {
		m.state.ETASeconds = 0
		return
	}
	m.state.ETASeconds = int64(float64(remaining) / m.state.SpeedBytesPerSec)
}

func (m *memoryManager) Subscribe() (<-chan State, func()) {
	s := &subscription{ch: make(chan State, subBufferSize)}

	m.mu.Lock()
	m.subs[s] = struct{}{}
	first := m.state
	// Send the current state immediately so new subscribers do not
	// have to wait for the next Update to learn the phase. Held under
	// the lock so a concurrent Update doesn't race with this initial
	// send (Update also holds the lock through its send loop).
	select {
	case s.ch <- first:
	default:
	}
	m.mu.Unlock()

	var once sync.Once
	unsubscribe := func() {
		once.Do(func() {
			// delete + close under the lock so Update cannot send to
			// s.ch after we close it. See Update's docstring for
			// the race this prevents.
			m.mu.Lock()
			delete(m.subs, s)
			close(s.ch)
			m.mu.Unlock()
		})
	}
	return s.ch, unsubscribe
}

func (m *memoryManager) Sink() Sink {
	return &managerSink{m: m}
}

// SeedDownloadBudget sets bytes_total and bytes_completed for a download
// pass where the aggregate size is known up front. Activates budget mode
// so per-file OnFileStart events do not inflate bytes_total mid-pass.
func (m *memoryManager) SeedDownloadBudget(total, completed int64) {
	if total < 0 {
		total = 0
	}
	if completed < 0 {
		completed = 0
	}
	if completed > total && total > 0 {
		completed = total
	}
	m.Update(func(st *State) {
		st.BytesTotal = total
		st.BytesCompleted = completed
	})
	m.mu.Lock()
	m.bytesBudgetActive = total > 0
	m.lastBytes = completed
	m.mu.Unlock()
}

func (m *memoryManager) SeedDownloadFileList(files []FileProgress) {
	cloned := make([]FileProgress, len(files))
	done := 0
	for i, f := range files {
		if f.Status == "" {
			f.Status = FileWaiting
		}
		if f.BytesCompleted < 0 {
			f.BytesCompleted = 0
		}
		if f.BytesTotal < 0 {
			f.BytesTotal = 0
		}
		if f.BytesTotal > 0 && f.BytesCompleted > f.BytesTotal {
			f.BytesCompleted = f.BytesTotal
		}
		if f.Status == FileDone {
			done++
		}
		cloned[i] = f
	}
	m.Update(func(st *State) {
		st.Files = cloned
		st.FilesTotal = len(cloned)
		st.FilesCompleted = done
	})
	m.mu.Lock()
	m.filesBudgetActive = len(cloned) > 0
	m.mu.Unlock()
}

func (m *memoryManager) SeedDownloadFiles(total, completed int) {
	if total < 0 {
		total = 0
	}
	if completed < 0 {
		completed = 0
	}
	if completed > total && total > 0 {
		completed = total
	}
	m.Update(func(st *State) {
		st.FilesTotal = total
		st.FilesCompleted = completed
	})
	m.mu.Lock()
	m.filesBudgetActive = total > 0
	m.mu.Unlock()
}

func (m *memoryManager) ClearDownloadBudget() {
	m.mu.Lock()
	m.bytesBudgetActive = false
	m.filesBudgetActive = false
	m.seenFiles = make(map[string]struct{})
	m.doneFiles = make(map[string]struct{})
	m.rateFilePath = ""
	m.rateFileBytes = 0
	m.mu.Unlock()
}

func (m *memoryManager) SaveTo(path string) error {
	state := m.Snapshot()
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("progress: marshal state: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("progress: mkdir: %w", err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("progress: write tmp: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("progress: rename: %w", err)
	}
	return nil
}

func (m *memoryManager) LoadFrom(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return os.ErrNotExist
		}
		return fmt.Errorf("progress: read: %w", err)
	}
	var s State
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("progress: unmarshal: %w", err)
	}
	if s.LastVerifyAt.IsZero() {
		s.LastVerifyOK = nil
	}
	m.mu.Lock()
	m.state = s
	m.lastBytes = s.BytesCompleted
	m.lastSample = s.UpdatedAt
	m.mu.Unlock()
	return nil
}
