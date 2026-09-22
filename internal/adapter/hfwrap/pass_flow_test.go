package hfwrap

import (
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/progress"
)

// tqdm-shaped events through the aggregator into a real progress.Manager,
// matching what pumpStderr produces for a whole-repo hf download.

func TestPassFlow_PinnedWholeRepoThroughAggregator(t *testing.T) {
	t.Parallel()
	m := progress.New(time.Now())
	m.SeedDownloadBudget(3000, 0)
	m.SeedDownloadFiles(3, 0)
	a := NewAggregator(m.Sink(), nil)

	a.OnProgress(HFProgressEvent{Label: "Fetching 3 files", Downloaded: 0, Total: 3})
	a.OnProgress(HFProgressEvent{Label: "Fetching 3 files", Downloaded: 1, Total: 3})
	playFile(a, "config.json", 1000)
	st := m.Snapshot()
	if st.BytesTotal != 3000 || st.FilesTotal != 3 || st.FilesCompleted != 1 {
		t.Fatalf("after file 1: bytes=%d/%d files=%d/%d",
			st.BytesCompleted, st.BytesTotal, st.FilesCompleted, st.FilesTotal)
	}

	playFile(a, "weights.safetensors", 1800)
	playFile(a, "tokenizer.json", 200)
	st = m.Snapshot()
	if st.BytesTotal != 3000 || st.BytesCompleted != 3000 {
		t.Errorf("bytes %d/%d, want 3000/3000", st.BytesCompleted, st.BytesTotal)
	}
	if st.FilesTotal != 3 || st.FilesCompleted != 3 {
		t.Errorf("files %d/%d, want 3/3", st.FilesCompleted, st.FilesTotal)
	}
}

func TestPassFlow_UnpinnedTwoSourcesThroughAggregator(t *testing.T) {
	t.Parallel()
	m := progress.New(time.Now())
	a := NewAggregator(m.Sink(), nil)

	a.OnProgress(HFProgressEvent{Label: "Fetching 2 files", Downloaded: 0, Total: 2})
	playFile(a, "extra/a.bin", 50)
	playFile(a, "extra/b.bin", 50)
	if got := m.Snapshot().FilesTotal; got != 2 {
		t.Fatalf("after extra: FilesTotal=%d, want 2", got)
	}

	a.OnProgress(HFProgressEvent{Label: "Fetching 1 files", Downloaded: 0, Total: 1})
	playFile(a, "model.gguf", 900)
	st := m.Snapshot()
	if st.FilesTotal != 3 {
		t.Errorf("after main: FilesTotal=%d, want 3", st.FilesTotal)
	}
	if st.FilesCompleted != 3 {
		t.Errorf("FilesCompleted=%d, want 3", st.FilesCompleted)
	}
	if st.BytesTotal != 1000 || st.BytesCompleted != 1000 {
		t.Errorf("bytes %d/%d, want 1000/1000", st.BytesCompleted, st.BytesTotal)
	}
}

func TestPassFlow_NoOuterBarStillCountsFiles(t *testing.T) {
	t.Parallel()
	m := progress.New(time.Now())
	a := NewAggregator(m.Sink(), nil)
	playFile(a, "a.bin", 10)
	playFile(a, "b.bin", 20)
	playFile(a, "c.bin", 30)
	st := m.Snapshot()
	if st.FilesTotal != 3 || st.FilesCompleted != 3 {
		t.Errorf("files %d/%d, want 3/3", st.FilesCompleted, st.FilesTotal)
	}
	if st.BytesTotal != 60 {
		t.Errorf("BytesTotal=%d, want 60", st.BytesTotal)
	}
}

func TestPassFlow_EmptyLabelDropped(t *testing.T) {
	t.Parallel()
	m := progress.New(time.Now())
	a := NewAggregator(m.Sink(), nil)
	a.OnProgress(HFProgressEvent{Label: "", Downloaded: 50, Total: 100})
	st := m.Snapshot()
	if st.BytesCompleted != 0 || st.FilesTotal != 0 {
		t.Errorf("empty label leaked into state: %+v", st)
	}
}

func TestPassFlow_LateTotalSettleUnpinned(t *testing.T) {
	t.Parallel()
	m := progress.New(time.Now())
	a := NewAggregator(m.Sink(), nil)
	a.OnProgress(HFProgressEvent{Label: "big.bin", Downloaded: 0, Total: 100})
	a.OnProgress(HFProgressEvent{Label: "big.bin", Downloaded: 50, Total: 400})
	a.OnProgress(HFProgressEvent{Label: "big.bin", Downloaded: 400, Total: 400})
	st := m.Snapshot()
	if st.BytesTotal != 400 {
		t.Errorf("BytesTotal=%d, want 400 after late settle", st.BytesTotal)
	}
	if st.FilesTotal != 1 || st.FilesCompleted != 1 {
		t.Errorf("files %d/%d, want 1/1", st.FilesCompleted, st.FilesTotal)
	}
}

func TestPassFlow_FlushDoesNotComplete(t *testing.T) {
	t.Parallel()
	m := progress.New(time.Now())
	a := NewAggregator(m.Sink(), nil)
	a.OnProgress(HFProgressEvent{Label: "stuck.bin", Downloaded: 10, Total: 100})
	a.FlushIncomplete()
	st := m.Snapshot()
	if st.FilesCompleted != 0 {
		t.Errorf("FilesCompleted=%d, want 0 after flush", st.FilesCompleted)
	}
	if st.BytesCompleted != 10 {
		t.Errorf("BytesCompleted=%d, want the 10 already transferred", st.BytesCompleted)
	}
}

func playFile(a *Aggregator, name string, size int64) {
	a.OnProgress(HFProgressEvent{Label: name, Downloaded: size / 2, Total: size})
	a.OnProgress(HFProgressEvent{Label: name, Downloaded: size, Total: size})
}
