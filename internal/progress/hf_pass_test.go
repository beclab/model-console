package progress

import (
	"testing"
	"time"
)

// These tests walk the Status-card contract for one HF ensure pass:
// what /api/progress looks like after each sink event, without a live
// Hub or hf CLI.

func TestHFPass_PinnedWholeRepoKeepsDenominator(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	m.SeedDownloadBudget(6000, 0)
	m.SeedDownloadFiles(3, 0)
	s := m.Sink()

	s.OnFilesTotal(3) // tqdm outer bar, repeated every tick
	s.OnFilesTotal(3)
	s.OnFileStart("config.json", 1000)
	s.OnBytes(1000)
	s.OnFileDone("config.json")

	st := m.Snapshot()
	if st.BytesTotal != 6000 {
		t.Errorf("after file 1: BytesTotal=%d, want 6000 (must not shrink to this file)", st.BytesTotal)
	}
	if st.BytesCompleted != 1000 {
		t.Errorf("after file 1: BytesCompleted=%d, want 1000", st.BytesCompleted)
	}
	if st.FilesTotal != 3 || st.FilesCompleted != 1 {
		t.Errorf("after file 1: files %d/%d, want 1/3", st.FilesCompleted, st.FilesTotal)
	}

	s.OnFileStart("model.safetensors", 4000)
	s.OnBytes(4000)
	s.OnFileDone("model.safetensors")
	s.OnFileStart("tokenizer.json", 1000)
	s.OnBytes(1000)
	s.OnFileDone("tokenizer.json")

	st = m.Snapshot()
	if st.BytesTotal != 6000 || st.BytesCompleted != 6000 {
		t.Errorf("after all files: bytes %d/%d, want 6000/6000", st.BytesCompleted, st.BytesTotal)
	}
	if st.FilesTotal != 3 || st.FilesCompleted != 3 {
		t.Errorf("after all files: files %d/%d, want 3/3", st.FilesCompleted, st.FilesTotal)
	}
	if st.CurrentFile != "" {
		t.Errorf("CurrentFile=%q, want empty", st.CurrentFile)
	}
}

func TestHFPass_UnpinnedFirstFileMustNotLockCount(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	s := m.Sink()

	s.OnFileStart("a.bin", 1000)
	s.OnBytes(1000)
	s.OnFileDone("a.bin")
	mid := m.Snapshot()
	if mid.BytesTotal != 1000 || mid.FilesTotal != 1 || mid.FilesCompleted != 1 {
		t.Fatalf("after file 1 (unpinned): bytes=%d/%d files=%d/%d",
			mid.BytesCompleted, mid.BytesTotal, mid.FilesCompleted, mid.FilesTotal)
	}

	s.OnFileStart("b.bin", 2000)
	s.OnBytes(2000)
	s.OnFileDone("b.bin")
	end := m.Snapshot()
	if end.BytesTotal != 3000 {
		t.Errorf("BytesTotal=%d, want 3000 after second file", end.BytesTotal)
	}
	if end.FilesTotal != 2 || end.FilesCompleted != 2 {
		t.Errorf("files %d/%d, want 2/2", end.FilesCompleted, end.FilesTotal)
	}
}

func TestHFPass_TwoCLIBatchesRaiseUnpinnedCount(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	s := m.Sink()

	s.OnFilesTotal(2)
	s.OnFileStart("extra/a.bin", 10)
	s.OnFileDone("extra/a.bin")
	s.OnFileStart("extra/b.bin", 10)
	s.OnFileDone("extra/b.bin")
	if got := m.Snapshot().FilesTotal; got != 2 {
		t.Fatalf("after first CLI: FilesTotal=%d, want 2", got)
	}

	s.OnFilesTotal(1) // second hf download; must not shrink
	s.OnFileStart("main.gguf", 100)
	s.OnFileDone("main.gguf")
	st := m.Snapshot()
	if st.FilesTotal != 3 {
		t.Errorf("after second CLI: FilesTotal=%d, want 3", st.FilesTotal)
	}
	if st.FilesCompleted != 3 {
		t.Errorf("FilesCompleted=%d, want 3", st.FilesCompleted)
	}
}

func TestHFPass_ResumeSeedPlusRemainingFiles(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	m.SeedDownloadBudget(1000, 400)
	m.SeedDownloadFiles(5, 2)
	s := m.Sink()

	s.OnFilesTotal(3) // hf only announces what is left
	s.OnFileStart("c.bin", 200)
	s.OnBytes(200)
	s.OnFileDone("c.bin")
	s.OnFileStart("d.bin", 200)
	s.OnBytes(200)
	s.OnFileDone("d.bin")
	s.OnFileStart("e.bin", 200)
	s.OnBytes(200)
	s.OnFileDone("e.bin")

	st := m.Snapshot()
	if st.BytesTotal != 1000 {
		t.Errorf("BytesTotal=%d, want 1000", st.BytesTotal)
	}
	if st.BytesCompleted != 1000 {
		t.Errorf("BytesCompleted=%d, want 400 seed + 600 new", st.BytesCompleted)
	}
	if st.FilesTotal != 5 {
		t.Errorf("FilesTotal=%d, want 5 (seed wins over Fetching 3)", st.FilesTotal)
	}
	if st.FilesCompleted != 5 {
		t.Errorf("FilesCompleted=%d, want 2 seed + 3 done", st.FilesCompleted)
	}
}

func TestHFPass_SeededLateSizeSettleDoesNotGrowTotal(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	m.SeedDownloadBudget(5000, 0)
	m.SeedDownloadFiles(1, 0)
	s := m.Sink()
	s.OnFileStart("model.safetensors", 1000)
	s.OnFileStart("model.safetensors", 4000) // tqdm total settled late
	st := m.Snapshot()
	if st.BytesTotal != 5000 {
		t.Errorf("BytesTotal=%d, want 5000 (pin ignores size top-up)", st.BytesTotal)
	}
	if st.FilesTotal != 1 {
		t.Errorf("FilesTotal=%d, want 1", st.FilesTotal)
	}
}

func TestHFPass_ClearBudgetForgetsSeenAndDone(t *testing.T) {
	t.Parallel()
	m := New(time.Now()).(*memoryManager)
	s := m.Sink()
	s.OnFileStart("a.bin", 100)
	s.OnFileDone("a.bin")
	m.ClearDownloadBudget()
	m.Update(func(st *State) {
		st.BytesTotal = 0
		st.BytesCompleted = 0
		st.FilesTotal = 0
		st.FilesCompleted = 0
	})
	s.OnFileStart("a.bin", 100)
	s.OnFileDone("a.bin")
	st := m.Snapshot()
	if st.FilesTotal != 1 || st.FilesCompleted != 1 {
		t.Errorf("after clear: files %d/%d, want 1/1 (same name is a new pass)", st.FilesCompleted, st.FilesTotal)
	}
}

func TestHFPass_DoneCappedAtSeededTotal(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	m.SeedDownloadFiles(2, 2)
	s := m.Sink()
	s.OnFileDone("already.bin")
	st := m.Snapshot()
	if st.FilesCompleted != 2 {
		t.Errorf("FilesCompleted=%d, want 2 (capped at seed)", st.FilesCompleted)
	}
}
