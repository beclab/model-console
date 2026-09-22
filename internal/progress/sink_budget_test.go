package progress

import (
	"testing"
	"time"
)

func TestSeedDownloadBudget_OnFileStartIgnored(t *testing.T) {
	t.Parallel()
	m := New(time.Now()).(*memoryManager)
	m.SeedDownloadBudget(1000, 100)
	sink := m.Sink()
	sink.OnFileStart("a", 500)
	sink.OnBytes(50)
	st := m.Snapshot()
	if st.BytesTotal != 1000 {
		t.Errorf("BytesTotal = %d, want 1000 (budget pinned)", st.BytesTotal)
	}
	if st.BytesCompleted != 150 {
		t.Errorf("BytesCompleted = %d, want 150", st.BytesCompleted)
	}
}

func TestClearDownloadBudget_RestoresOnFileStart(t *testing.T) {
	t.Parallel()
	m := New(time.Now()).(*memoryManager)
	m.SeedDownloadBudget(1000, 0)
	m.ClearDownloadBudget()
	m.Update(func(st *State) {
		st.BytesTotal = 0
		st.BytesCompleted = 0
	})
	sink := m.Sink()
	sink.OnFileStart("a", 300)
	st := m.Snapshot()
	if st.BytesTotal != 300 {
		t.Errorf("BytesTotal = %d, want 300", st.BytesTotal)
	}
}

func TestSeedDownloadFiles_OnFileStartDoesNotGrowCount(t *testing.T) {
	t.Parallel()
	m := New(time.Now()).(*memoryManager)
	m.SeedDownloadFiles(13, 2)
	sink := m.Sink()
	sink.OnFileStart("a.bin", 500)
	sink.OnFileDone("a.bin")
	st := m.Snapshot()
	if st.FilesTotal != 13 {
		t.Errorf("FilesTotal = %d, want 13 (pinned)", st.FilesTotal)
	}
	if st.FilesCompleted != 3 {
		t.Errorf("FilesCompleted = %d, want 3 (seed + done)", st.FilesCompleted)
	}
	if st.CurrentFile != "" {
		t.Errorf("CurrentFile = %q, want empty after OnFileDone", st.CurrentFile)
	}
}

func TestOnFilesTotal_RaisesAndIgnoresShrink(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	sink := m.Sink()
	sink.OnFilesTotal(7)
	sink.OnFileStart("a.bin", 100)
	sink.OnFilesTotal(3) // must not shrink
	st := m.Snapshot()
	if st.FilesTotal != 7 {
		t.Errorf("FilesTotal = %d, want 7", st.FilesTotal)
	}
}

func TestOnFilesTotal_DoesNotPinLaterStarts(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	sink := m.Sink()
	sink.OnFilesTotal(2)
	sink.OnFileStart("a.bin", 10)
	sink.OnFileStart("b.bin", 10)
	sink.OnFileStart("c.bin", 10) // second HF source, no new Fetching N
	st := m.Snapshot()
	if st.FilesTotal != 3 {
		t.Errorf("FilesTotal = %d, want 3 (seen files raise an unpinned hint)", st.FilesTotal)
	}
}

func TestOnFileDone_Idempotent(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	sink := m.Sink()
	sink.OnFileStart("a.bin", 100)
	sink.OnFileDone("a.bin")
	sink.OnFileDone("a.bin")
	st := m.Snapshot()
	if st.FilesCompleted != 1 {
		t.Errorf("FilesCompleted = %d, want 1", st.FilesCompleted)
	}
}

func TestOnFileStart_GrowsFilesTotalUntilPinned(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	sink := m.Sink()
	sink.OnFileStart("a.bin", 100)
	sink.OnFileStart("a.bin", 20) // size top-up, same file
	sink.OnFileStart("b.bin", 200)
	st := m.Snapshot()
	if st.FilesTotal != 2 {
		t.Errorf("FilesTotal = %d, want 2", st.FilesTotal)
	}
	if st.BytesTotal != 320 {
		t.Errorf("BytesTotal = %d, want 320", st.BytesTotal)
	}
}

func TestSeedDownloadBudget_OnBytesAddsToSeed(t *testing.T) {
	t.Parallel()
	m := New(time.Now()).(*memoryManager)
	m.SeedDownloadBudget(1000, 400)
	sink := m.Sink()
	sink.OnBytes(50)
	st := m.Snapshot()
	if st.BytesCompleted != 450 {
		t.Errorf("BytesCompleted = %d, want 450 (seed + delta)", st.BytesCompleted)
	}
	if st.BytesTotal != 1000 {
		t.Errorf("BytesTotal = %d, want 1000", st.BytesTotal)
	}
}
