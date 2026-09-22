package progress

import (
	"testing"
	"time"
)

func TestFileQualifiesForRate(t *testing.T) {
	t.Parallel()
	cases := []struct {
		file, total int64
		want        bool
	}{
		{8 << 20, 4_700_000_000, true},  // 8 MiB shard fragment
		{7 << 20, 4_700_000_000, false}, // still handshake-sized vs the pin
		{5 << 20, 5 << 20, true},        // 5 MiB is the whole download
		{2000, 4_700_000_000, false},
		{0, 4_700_000_000, false},
	}
	for _, tc := range cases {
		if got := fileQualifiesForRate(tc.file, tc.total); got != tc.want {
			t.Errorf("fileQualifiesForRate(%d, %d) = %v, want %v",
				tc.file, tc.total, got, tc.want)
		}
	}
}

func TestSmallFilesDoNotSetSpeed(t *testing.T) {
	t.Parallel()
	now := time.Now()
	m := New(now).(*memoryManager)
	m.SeedDownloadBudget(4_700_000_000, 0)
	s := m.Sink()
	s.OnFileStart("config.json", 2000)

	m.mu.Lock()
	m.lastSample = now
	m.lastBytes = 0
	old := m.state
	m.state.BytesCompleted = 2000
	m.state.UpdatedAt = now.Add(time.Second)
	m.recomputeRate(old)
	m.mu.Unlock()

	st := m.Snapshot()
	if st.SpeedBytesPerSec != 0 {
		t.Errorf("speed = %.2f, want 0 (config.json is not a rate sample)", st.SpeedBytesPerSec)
	}
	if st.ETASeconds != 0 {
		t.Errorf("eta = %d, want 0 until a bulk sample", st.ETASeconds)
	}
}

func TestLargeFileSetsSpeedAfterSmallFiles(t *testing.T) {
	t.Parallel()
	now := time.Now()
	m := New(now).(*memoryManager)
	m.SeedDownloadBudget(4_700_000_000, 0)
	s := m.Sink()
	s.OnFileStart("config.json", 2000)
	s.OnBytes(2000)
	s.OnFileDone("config.json")

	if m.Snapshot().SpeedBytesPerSec != 0 {
		t.Fatal("small file must not have set the rate")
	}

	s.OnFileStart("model-00001-of-00002.safetensors", 4_220_000_000)
	m.mu.Lock()
	m.lastSample = now
	m.lastBytes = m.state.BytesCompleted
	old := m.state
	m.state.BytesCompleted += 1 << 20
	m.state.UpdatedAt = now.Add(time.Second)
	m.recomputeRate(old)
	m.mu.Unlock()

	st := m.Snapshot()
	if st.SpeedBytesPerSec < 9e5 || st.SpeedBytesPerSec > 1.2e6 {
		t.Errorf("speed = %.2f, want ~1 MiB/s from the shard window", st.SpeedBytesPerSec)
	}
	if st.ETASeconds <= 0 {
		t.Error("eta should be set once a bulk sample exists")
	}
}

func TestSmallFileAfterShardKeepsSpeed(t *testing.T) {
	t.Parallel()
	now := time.Now()
	m := New(now).(*memoryManager)
	m.SeedDownloadBudget(4_700_000_000, 0)
	s := m.Sink()
	s.OnFileStart("model.safetensors", 4_220_000_000)

	m.mu.Lock()
	m.lastSample = now
	m.lastBytes = 0
	old := m.state
	m.state.BytesCompleted = 1 << 20
	m.state.UpdatedAt = now.Add(time.Second)
	m.recomputeRate(old)
	m.mu.Unlock()
	bulk := m.Snapshot().SpeedBytesPerSec
	if bulk < 9e5 {
		t.Fatalf("precondition: shard speed = %.2f", bulk)
	}

	s.OnFileDone("model.safetensors")
	s.OnFileStart("vocab.json", 3000)
	m.mu.Lock()
	old = m.state
	m.state.BytesCompleted += 3000
	m.state.UpdatedAt = now.Add(2 * time.Second)
	m.recomputeRate(old)
	m.mu.Unlock()

	if got := m.Snapshot().SpeedBytesPerSec; got != bulk {
		t.Errorf("speed = %.2f after vocab.json, want the shard rate %.2f", got, bulk)
	}
}

func TestOnFileStartTopUpDoesNotShrinkRateSize(t *testing.T) {
	t.Parallel()
	m := New(time.Now()).(*memoryManager)
	s := m.Sink()
	s.OnFileStart("big.bin", 4<<20)
	s.OnFileStart("big.bin", 100) // aggregator size top-up delta
	if m.rateFileBytes != 4<<20 {
		t.Errorf("rateFileBytes = %d, want the full total not the top-up", m.rateFileBytes)
	}
}
