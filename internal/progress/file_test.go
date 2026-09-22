package progress

import (
	"testing"
	"time"
)

func TestMatchFileIndex_BasenamePrefersWaiting(t *testing.T) {
	t.Parallel()
	files := []FileProgress{
		{Path: "weights/a.bin", Status: FileDone},
		{Path: "extra/a.bin", Status: FileWaiting},
	}
	if got := matchFileIndex(files, "a.bin"); got != 1 {
		t.Errorf("match a.bin = %d, want 1 (waiting extra/a.bin)", got)
	}
	if got := matchFileIndex(files, "weights/a.bin"); got != 0 {
		t.Errorf("exact path = %d, want 0", got)
	}
}

// Two files with one name in different directories each complete once,
// provided the producer names them by repository path. The de-dupe on
// OnFileDone is keyed on that name, so a producer sending the bare
// basename twice gets one completion and leaves the other row behind.
func TestOnFileDone_SameBasenameDistinctPaths(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	m.SeedDownloadFileList([]FileProgress{
		{Path: "instruct/model.safetensors", BytesTotal: 800, Status: FileWaiting},
		{Path: "redae/model.safetensors", BytesTotal: 300, Status: FileWaiting},
	})
	s := m.Sink()
	for _, f := range []struct {
		path string
		size int64
	}{{"instruct/model.safetensors", 800}, {"redae/model.safetensors", 300}} {
		s.OnFileStart(f.path, f.size)
		s.(FileByteSink).OnFileBytes(f.path, f.size)
		s.OnFileDone(f.path)
	}

	st := m.Snapshot()
	if st.FilesCompleted != 2 {
		t.Errorf("files_completed = %d, want 2", st.FilesCompleted)
	}
	for _, f := range st.Files {
		if f.Status != FileDone || f.BytesCompleted != f.BytesTotal {
			t.Errorf("row %s = %s %d/%d, want a full done row",
				f.Path, f.Status, f.BytesCompleted, f.BytesTotal)
		}
	}
	if st.BytesCompleted != 1100 {
		t.Errorf("bytes_completed = %d, want 1100", st.BytesCompleted)
	}
}

func TestSeedDownloadFileList_ThenStartAndBytes(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	m.SeedDownloadFileList([]FileProgress{
		{Path: "mmproj-F16.gguf", BytesTotal: 100, Status: FileWaiting},
		{Path: "model.gguf", BytesTotal: 900, Status: FileWaiting},
	})
	st := m.Snapshot()
	if st.FilesTotal != 2 || st.FilesCompleted != 0 || len(st.Files) != 2 {
		t.Fatalf("seed: total=%d done=%d rows=%d", st.FilesTotal, st.FilesCompleted, len(st.Files))
	}
	if st.Files[0].Status != FileWaiting || st.Files[1].Status != FileWaiting {
		t.Fatalf("seed statuses: %+v", st.Files)
	}

	s := m.Sink()
	s.OnFileStart("mmproj-F16.gguf", 100)
	if fb, ok := s.(FileByteSink); ok {
		fb.OnFileBytes("mmproj-F16.gguf", 40)
	} else {
		t.Fatal("Sink must implement FileByteSink")
	}
	st = m.Snapshot()
	if st.Files[0].Status != FileDownloading {
		t.Errorf("active file status=%q", st.Files[0].Status)
	}
	if st.Files[1].Status != FileWaiting {
		t.Errorf("queued file status=%q, want waiting", st.Files[1].Status)
	}
	if st.Files[0].BytesCompleted != 40 {
		t.Errorf("active bytes=%d, want 40", st.Files[0].BytesCompleted)
	}

	s.OnFileDone("mmproj-F16.gguf")
	s.OnFileStart("model.gguf", 900)
	st = m.Snapshot()
	if st.Files[0].Status != FileDone || st.Files[1].Status != FileDownloading {
		t.Errorf("after first done: %+v", st.Files)
	}
	if st.FilesCompleted != 1 || st.FilesTotal != 2 {
		t.Errorf("files %d/%d, want 1/2", st.FilesCompleted, st.FilesTotal)
	}
}

func TestOnFileBytes_InterleavedDoesNotStealRow(t *testing.T) {
	t.Parallel()
	m := New(time.Now())
	m.SeedDownloadFileList([]FileProgress{
		{Path: "a.bin", BytesTotal: 1000, Status: FileWaiting},
		{Path: "b.bin", BytesTotal: 2000, Status: FileWaiting},
	})
	sink := m.Sink()
	s, ok := sink.(FileByteSink)
	if !ok {
		t.Fatal("Sink must implement FileByteSink")
	}
	sink.OnFileStart("a.bin", 1000)
	sink.OnFileStart("b.bin", 2000)
	s.OnFileBytes("a.bin", 100)
	s.OnFileBytes("b.bin", 500)
	s.OnFileBytes("a.bin", 200)
	st := m.Snapshot()
	if st.Files[0].BytesCompleted != 300 {
		t.Errorf("a.bin bytes=%d, want 300", st.Files[0].BytesCompleted)
	}
	if st.Files[1].BytesCompleted != 500 {
		t.Errorf("b.bin bytes=%d, want 500", st.Files[1].BytesCompleted)
	}
	if st.BytesCompleted != 800 {
		t.Errorf("pass bytes=%d, want 800", st.BytesCompleted)
	}
}
