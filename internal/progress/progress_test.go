package progress

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestNew_InitialState(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, 5, 1, 10, 0, 0, 0, time.UTC)
	m := New(now)

	got := m.Snapshot()
	if got.Phase != PhaseInit {
		t.Errorf("Phase = %q, want %q", got.Phase, PhaseInit)
	}
	if !got.StartedAt.Equal(now) {
		t.Errorf("StartedAt = %v, want %v", got.StartedAt, now)
	}
	if !got.UpdatedAt.Equal(now) {
		t.Errorf("UpdatedAt = %v, want %v", got.UpdatedAt, now)
	}
}

func TestUpdate_PhaseTransition(t *testing.T) {
	t.Parallel()
	m := New(time.Now())

	transitions := []Phase{
		PhaseInit, PhaseDownload, PhaseLoading,
		PhaseReady, PhaseDegraded, PhaseFailed,
	}
	for _, p := range transitions {
		m.Update(func(s *State) { s.Phase = p })
		if got := m.Snapshot().Phase; got != p {
			t.Errorf("Phase = %q, want %q", got, p)
		}
	}
}

func TestSubscribe_DeliversInitialAndUpdates(t *testing.T) {
	t.Parallel()
	m := New(time.Now())

	ch, unsub := m.Subscribe()
	defer unsub()

	first := <-ch
	if first.Phase != PhaseInit {
		t.Errorf("first frame phase = %q, want %q", first.Phase, PhaseInit)
	}

	m.Update(func(s *State) { s.Phase = PhaseDownload })

	select {
	case got := <-ch:
		if got.Phase != PhaseDownload {
			t.Errorf("update frame phase = %q, want %q", got.Phase, PhaseDownload)
		}
	case <-time.After(time.Second):
		t.Fatal("did not receive update within 1s")
	}
}

func TestSubscribe_ClosesOnUnsubscribe(t *testing.T) {
	t.Parallel()
	m := New(time.Now())

	ch, unsub := m.Subscribe()
	<-ch // drain initial
	unsub()

	_, ok := <-ch
	if ok {
		t.Error("channel should be closed after unsubscribe")
	}

	// Calling unsub twice must be a no-op (sync.Once).
	unsub()
}

func TestSubscribe_SlowConsumerDoesNotBlock(t *testing.T) {
	t.Parallel()
	m := New(time.Now())

	_, unsub := m.Subscribe()
	defer unsub()

	// Saturate the subscriber's buffer (8) and exceed it; Update must not block.
	done := make(chan struct{})
	go func() {
		for i := 0; i < 100; i++ {
			m.Update(func(s *State) { s.BytesCompleted = int64(i) })
		}
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Update blocked when subscriber buffer was full")
	}
}

func TestUpdate_EMASmoothsRate(t *testing.T) {
	t.Parallel()
	now := time.Now()
	m := New(now).(*memoryManager)

	// Simulate two distinct samples 1s apart with 1MiB each → instant rate
	// is ~1MiB/s; EMA should approach that value.
	m.lastSample = now
	m.lastBytes = 0

	step := func(after time.Duration, bytes int64) {
		m.mu.Lock()
		old := m.state
		m.state.BytesCompleted = bytes
		m.state.UpdatedAt = now.Add(after)
		m.recomputeRate(old)
		m.mu.Unlock()
	}

	step(time.Second, 1<<20)
	step(2*time.Second, 2<<20)
	step(3*time.Second, 3<<20)

	got := m.Snapshot()
	if got.SpeedBytesPerSec < 9e5 || got.SpeedBytesPerSec > 1.2e6 {
		t.Errorf("speed = %.2f, want ~1MiB/s", got.SpeedBytesPerSec)
	}
}

func TestUpdate_ETAComputed(t *testing.T) {
	t.Parallel()
	now := time.Now()
	m := New(now).(*memoryManager)

	m.Update(func(s *State) {
		s.BytesTotal = 10 << 20
	})

	// Advance one sample at 1MiB/s.
	m.mu.Lock()
	m.lastSample = now
	m.lastBytes = 0
	old := m.state
	m.state.BytesCompleted = 1 << 20
	m.state.UpdatedAt = now.Add(time.Second)
	m.recomputeRate(old)
	m.mu.Unlock()

	got := m.Snapshot()
	if got.ETASeconds < 8 || got.ETASeconds > 12 {
		t.Errorf("eta = %d, want ~9s", got.ETASeconds)
	}
}

func TestUpdate_ETAZeroWhenIdle(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	m.Update(func(s *State) {
		s.BytesTotal = 0
		s.BytesCompleted = 0
	})
	if eta := m.Snapshot().ETASeconds; eta != 0 {
		t.Errorf("eta = %d, want 0 when nothing to download", eta)
	}
}

func TestUpdate_RegressionResetsRate(t *testing.T) {
	t.Parallel()
	now := time.Now()
	m := New(now).(*memoryManager)

	m.mu.Lock()
	m.lastSample = now
	m.lastBytes = 5 << 20
	m.state.BytesCompleted = 5 << 20
	m.state.SpeedBytesPerSec = 1e6
	m.mu.Unlock()

	m.mu.Lock()
	old := m.state
	m.state.BytesCompleted = 0 // simulate retry from zero
	m.state.UpdatedAt = now.Add(time.Second)
	m.recomputeRate(old)
	m.mu.Unlock()

	if got := m.Snapshot().SpeedBytesPerSec; got != 0 {
		t.Errorf("speed = %.2f after regression, want 0", got)
	}
}

func TestConcurrent_UpdateAndSubscribe(t *testing.T) {
	t.Parallel()
	m := New(time.Now())

	const writers = 8
	const updates = 200

	var received atomic.Int64
	var subWG sync.WaitGroup
	unsubs := make([]func(), 0, 4)
	for i := 0; i < 4; i++ {
		ch, unsub := m.Subscribe()
		unsubs = append(unsubs, unsub)
		subWG.Add(1)
		go func() {
			defer subWG.Done()
			for range ch {
				received.Add(1)
			}
		}()
	}

	var wg sync.WaitGroup
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < updates; j++ {
				m.Update(func(s *State) { s.BytesCompleted++ })
			}
		}()
	}
	wg.Wait()

	want := int64(writers * updates)
	if got := m.Snapshot().BytesCompleted; got != want {
		t.Errorf("BytesCompleted = %d, want %d", got, want)
	}

	for _, u := range unsubs {
		u()
	}
	subWG.Wait()
	if received.Load() == 0 {
		t.Error("no frames received by subscribers")
	}
}

func TestState_MarshalJSON_OmitsZeroLastVerifyAt(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, 5, 1, 10, 0, 0, 0, time.UTC)
	ok := true
	s := State{
		Phase:        PhaseDownload,
		StartedAt:    now,
		UpdatedAt:    now,
		LastVerifyAt: time.Time{},
		LastVerifyOK: &ok,
	}
	data, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("Unmarshal map: %v", err)
	}
	if _, present := m["last_verify_at"]; present {
		t.Fatalf("last_verify_at should be omitted, got %v", m["last_verify_at"])
	}
	if _, present := m["last_verify_ok"]; present {
		t.Fatalf("last_verify_ok should be dropped without a timestamp, got %v", m["last_verify_ok"])
	}
}

func TestSaveLoad_RoundTrip(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	m := New(now)
	m.Update(func(s *State) {
		s.Phase = PhaseDownload
		s.BytesTotal = 999
		s.BytesCompleted = 333
	})

	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "state.json")

	if err := m.SaveTo(path); err != nil {
		t.Fatalf("SaveTo: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("stat: %v", err)
	}

	m2 := New(time.Now())
	if err := m2.LoadFrom(path); err != nil {
		t.Fatalf("LoadFrom: %v", err)
	}

	got := m2.Snapshot()
	want := m.Snapshot()
	if got.Phase != want.Phase ||
		got.BytesTotal != want.BytesTotal ||
		got.BytesCompleted != want.BytesCompleted {
		t.Errorf("roundtrip mismatch:\n got=%+v\nwant=%+v", got, want)
	}
}

func TestLoadFrom_MissingFile(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	err := m.LoadFrom(filepath.Join(t.TempDir(), "missing.json"))
	if !os.IsNotExist(err) {
		t.Errorf("err = %v, want os.ErrNotExist", err)
	}
}

func TestLoadFrom_BadJSON(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	m := New(time.Now())
	if err := m.LoadFrom(path); err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestSaveTo_Atomic(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "state.json")
	m := New(time.Now())
	if err := m.SaveTo(path); err != nil {
		t.Fatalf("SaveTo: %v", err)
	}
	// Tmp file should be gone after rename.
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Errorf("tmp file still exists: %v", err)
	}
	// File must be valid JSON.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var s State
	if err := json.Unmarshal(data, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
}
