package hfwrap

import (
	"sync"
	"testing"

	"github.com/llm-init/llm-init/internal/progress"
)

type aggSink struct {
	mu         sync.Mutex
	starts     []startEvt
	bytes      int64
	dones      []string
	filesTotal int
}

type startEvt struct {
	Path  string
	Total int64
}

func (s *aggSink) OnFileStart(path string, total int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.starts = append(s.starts, startEvt{Path: path, Total: total})
}
func (s *aggSink) OnBytes(d int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.bytes += d
}
func (s *aggSink) OnFileDone(path string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.dones = append(s.dones, path)
}
func (s *aggSink) OnError(error)          {}
func (s *aggSink) OnPhase(progress.Phase) {}
func (s *aggSink) OnFilesTotal(n int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.filesTotal = n
}
func (s *aggSink) OnResolvedCommit(string) {}
func (s *aggSink) OnNote(string)           {}

var _ progress.Sink = (*aggSink)(nil)

func TestAggregator_PerFileMonotone(t *testing.T) {
	s := &aggSink{}
	a := NewAggregator(s, nil)

	a.OnProgress(HFProgressEvent{Label: "a.bin", Downloaded: 100, Total: 1000})
	a.OnProgress(HFProgressEvent{Label: "b.bin", Downloaded: 500, Total: 2000})
	a.OnProgress(HFProgressEvent{Label: "a.bin", Downloaded: 700, Total: 1000})
	a.OnProgress(HFProgressEvent{Label: "b.bin", Downloaded: 1500, Total: 2000})
	a.OnProgress(HFProgressEvent{Label: "a.bin", Downloaded: 1000, Total: 1000})
	a.OnProgress(HFProgressEvent{Label: "b.bin", Downloaded: 2000, Total: 2000})

	if got, want := s.bytes, int64(1000+2000); got != want {
		t.Errorf("bytes=%d want %d (interleaved deltas must not double-count)", got, want)
	}
	if len(s.starts) != 2 {
		t.Errorf("starts=%d want 2 (one per file)", len(s.starts))
	}
	if len(s.dones) != 2 {
		t.Errorf("dones=%d want 2 (each file completes once)", len(s.dones))
	}
}

func TestAggregator_SkipsOuterAggregateBar(t *testing.T) {
	s := &aggSink{}
	a := NewAggregator(s, nil)
	a.OnProgress(HFProgressEvent{Label: "Fetching 13 files", Downloaded: 5, Total: 13})
	if s.bytes != 0 {
		t.Errorf("outer bar must not contribute bytes; got %d", s.bytes)
	}
	if len(s.starts) != 0 {
		t.Errorf("outer bar must not produce OnFileStart; got %d", len(s.starts))
	}
	if s.filesTotal != 13 {
		t.Errorf("outer bar must pin files_total; got %d", s.filesTotal)
	}
}

// The two model.safetensors of FireRedTeam/FireRedTTS3, with the totals
// tqdm prints for them. Before the tree named the bars, the second file
// landed on the first one's state and every one of its ticks was
// filtered out by the byte-delta gate: 3.5 GiB reported as nothing.
const (
	instructTotal = int64(8480000000)
	redaeTotal    = int64(3780000000)
	instructPath  = "fireredtts3_instruct/model.safetensors"
	redaePath     = "redae/model.safetensors"
)

func TestAggregator_DuplicateBasenameSequential(t *testing.T) {
	t.Parallel()
	s := &aggSink{}
	a := NewAggregator(s, fireRedTree)

	a.OnProgress(HFProgressEvent{Label: "model.safetensors", Downloaded: 10500000, Total: instructTotal})
	a.OnProgress(HFProgressEvent{Label: "model.safetensors", Downloaded: instructTotal, Total: instructTotal})
	a.OnProgress(HFProgressEvent{Label: "model.safetensors", Downloaded: 10500000, Total: redaeTotal})
	a.OnProgress(HFProgressEvent{Label: "model.safetensors", Downloaded: redaeTotal, Total: redaeTotal})

	if got, want := s.bytes, instructTotal+redaeTotal; got != want {
		t.Errorf("bytes=%d want %d (the namesake's transfer must be counted)", got, want)
	}
	assertPaths(t, "starts", startPaths(s.starts), []string{instructPath, redaePath})
	assertPaths(t, "dones", s.dones, []string{instructPath, redaePath})
}

func TestAggregator_DuplicateBasenameInterleaved(t *testing.T) {
	t.Parallel()
	s := &aggSink{}
	a := NewAggregator(s, fireRedTree)

	// max_workers > 1: both bars print under the same label, and only
	// the total each counts towards tells them apart.
	a.OnProgress(HFProgressEvent{Label: "model.safetensors", Downloaded: 1000, Total: instructTotal})
	a.OnProgress(HFProgressEvent{Label: "model.safetensors", Downloaded: 2000, Total: redaeTotal})
	a.OnProgress(HFProgressEvent{Label: "model.safetensors", Downloaded: instructTotal, Total: instructTotal})
	a.OnProgress(HFProgressEvent{Label: "model.safetensors", Downloaded: redaeTotal, Total: redaeTotal})

	if got, want := s.bytes, instructTotal+redaeTotal; got != want {
		t.Errorf("bytes=%d want %d", got, want)
	}
	assertPaths(t, "dones", s.dones, []string{instructPath, redaePath})
}

func TestAggregator_DuplicateBasenameWithoutTree(t *testing.T) {
	t.Parallel()
	s := &aggSink{}
	a := NewAggregator(s, nil)

	// No tree to name the bars, so the rewind is the only evidence that
	// a second file started. Sequential bars still have to be counted
	// separately.
	a.OnProgress(HFProgressEvent{Label: "model.safetensors", Downloaded: 500, Total: instructTotal})
	a.OnProgress(HFProgressEvent{Label: "model.safetensors", Downloaded: instructTotal, Total: instructTotal})
	a.OnProgress(HFProgressEvent{Label: "model.safetensors", Downloaded: 500, Total: redaeTotal})
	a.OnProgress(HFProgressEvent{Label: "model.safetensors", Downloaded: redaeTotal, Total: redaeTotal})

	if got, want := s.bytes, instructTotal+redaeTotal; got != want {
		t.Errorf("bytes=%d want %d", got, want)
	}
	if len(s.dones) != 2 {
		t.Errorf("dones=%v, want two completions", s.dones)
	}
}

func TestAggregator_TreeNamesBarWithRepositoryPath(t *testing.T) {
	t.Parallel()
	s := &aggSink{}
	a := NewAggregator(s, fireRedTree)

	// A name unique in the tree still resolves to its full path, which
	// is what lets the dashboard row match exactly instead of by
	// basename.
	a.OnProgress(HFProgressEvent{Label: ".gitattributes", Downloaded: 1585, Total: 1585})
	assertPaths(t, "dones", s.dones, []string{".gitattributes"})
}

func startPaths(evts []startEvt) []string {
	out := make([]string, 0, len(evts))
	for _, e := range evts {
		out = append(out, e.Path)
	}
	return out
}

func assertPaths(t *testing.T, what string, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s = %v, want %v", what, got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s = %v, want %v", what, got, want)
		}
	}
}

func TestAggregator_FlushIncomplete(t *testing.T) {
	s := &aggSink{}
	a := NewAggregator(s, nil)
	a.OnProgress(HFProgressEvent{Label: "stuck.bin", Downloaded: 100, Total: 1000})
	if len(s.dones) != 0 {
		t.Fatal("file should not be done yet")
	}
	a.FlushIncomplete()
	if len(s.dones) != 0 {
		t.Errorf("flush must not pretend incomplete files finished: got %v", s.dones)
	}
}
