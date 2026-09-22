package progress

import (
	"errors"
	"testing"
	"time"
)

func TestManagerSink_TrackBytes(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	s := m.Sink()

	s.OnFileStart("a.bin", 100)
	s.OnFileStart("b.bin", 200)

	if got := m.Snapshot().BytesTotal; got != 300 {
		t.Errorf("BytesTotal = %d, want 300", got)
	}

	s.OnBytes(50)
	s.OnBytes(50)
	if got := m.Snapshot().BytesCompleted; got != 100 {
		t.Errorf("BytesCompleted = %d, want 100", got)
	}

	s.OnFileDone("b.bin") // no-op, must not panic
}

// TestManagerSink_OnError locks the transport / lifecycle split:
// OnError bumps TransportRetries only. It must not touch RetryCount
// (lifecycle-level) or LastError (current-failure card — set by
// lifecycle.fail so a transient blip mid-download does not turn the
// Status card red while bytes are still flowing).
func TestManagerSink_OnError(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	s := m.Sink()

	s.OnError(errors.New("connection reset"))
	s.OnError(errors.New("timeout"))
	s.OnError(nil) // must not panic, must not bump TransportRetries

	snap := m.Snapshot()
	if snap.LastError != "" {
		t.Errorf("LastError = %q, want empty (OnError must not set current-failure)", snap.LastError)
	}
	if snap.TransportRetries != 2 {
		t.Errorf("TransportRetries = %d, want 2", snap.TransportRetries)
	}
	if snap.RetryCount != 0 {
		t.Errorf("RetryCount = %d, want 0 (transport errors must NOT bump operator-visible retry_count)", snap.RetryCount)
	}
}

func TestManagerSink_OnPhase(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	s := m.Sink()
	s.OnPhase(PhaseDownload)
	if got := m.Snapshot().Phase; got != PhaseDownload {
		t.Errorf("Phase = %q, want %q", got, PhaseDownload)
	}
}

func TestManagerSink_OnFileStartUnknownSize(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	s := m.Sink()
	s.OnFileStart("unknown", -1)
	if got := m.Snapshot().BytesTotal; got != 0 {
		t.Errorf("BytesTotal = %d, want 0 when size unknown", got)
	}
}

func TestNopSink_DoesNothing(t *testing.T) {
	t.Parallel()
	var s Sink = NopSink{}
	s.OnFileStart("x", 100)
	s.OnBytes(10)
	s.OnFileDone("x")
	s.OnFilesTotal(1)
	s.OnError(errors.New("ignored"))
	s.OnPhase(PhaseReady)
}

func TestManagerSink_OnBytesZero(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	s := m.Sink()
	before := m.Snapshot().UpdatedAt
	// No clock-advance sleep: OnBytes(0) is the operation under test
	// and should be a no-op that never reads time.Now(); waiting 2ms
	// here was an accidental deflake that did nothing useful (Windows
	// clock granularity can be coarser than 2ms anyway). The
	// after.Equal(before) assertion is exact and cheap.
	s.OnBytes(0)
	after := m.Snapshot().UpdatedAt
	if !after.Equal(before) {
		t.Errorf("UpdatedAt should not change for zero-byte event: before=%v after=%v", before, after)
	}
}
