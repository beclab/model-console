package progress

import (
	"path/filepath"
	"strings"
)

// FileByteSink attributes a transfer delta to one named file as well as
// the pass totals. Manager.Sink() implements it. Producers that only
// know a byte delta (single-stream fetch) keep calling OnBytes, which
// lands on CurrentFile.
type FileByteSink interface {
	OnFileBytes(path string, delta int64)
}

func fileBase(path string) string {
	path = strings.ReplaceAll(path, "\\", "/")
	return filepath.Base(path)
}

// matchFileIndex finds the row for path. Exact path wins, then a
// waiting file with the same basename (so a tqdm label "a.bin" claims
// the seeded "weights/a.bin" that has not started), then any basename
// match.
func matchFileIndex(files []FileProgress, path string) int {
	if path == "" || len(files) == 0 {
		return -1
	}
	base := fileBase(path)
	exact, waiting, any := -1, -1, -1
	for i, f := range files {
		if f.Path == path {
			exact = i
			break
		}
		if fileBase(f.Path) != base {
			continue
		}
		if any < 0 {
			any = i
		}
		if f.Status == FileWaiting && waiting < 0 {
			waiting = i
		}
	}
	if exact >= 0 {
		return exact
	}
	if waiting >= 0 {
		return waiting
	}
	return any
}

func markFileStart(st *State, path string, totalBytes int64, growTotal bool) {
	if path == "" {
		return
	}
	st.CurrentFile = path
	idx := matchFileIndex(st.Files, path)
	if idx < 0 {
		fp := FileProgress{Path: path, Status: FileDownloading}
		if totalBytes > 0 {
			fp.BytesTotal = totalBytes
		}
		st.Files = append(st.Files, fp)
		if growTotal && st.FilesTotal < len(st.Files) {
			st.FilesTotal = len(st.Files)
		}
		return
	}
	f := &st.Files[idx]
	if f.Status != FileDone {
		f.Status = FileDownloading
	}
	if totalBytes > f.BytesTotal {
		f.BytesTotal = totalBytes
	}
}

func addFileBytes(st *State, path string, delta int64, growTotal bool) {
	if path == "" || delta == 0 {
		return
	}
	idx := matchFileIndex(st.Files, path)
	if idx < 0 {
		fp := FileProgress{
			Path:           path,
			BytesCompleted: delta,
			Status:         FileDownloading,
		}
		st.Files = append(st.Files, fp)
		if growTotal && st.FilesTotal < len(st.Files) {
			st.FilesTotal = len(st.Files)
		}
		return
	}
	f := &st.Files[idx]
	if f.Status == FileWaiting {
		f.Status = FileDownloading
	}
	f.BytesCompleted += delta
	if f.BytesCompleted < 0 {
		f.BytesCompleted = 0
	}
	if f.BytesTotal > 0 && f.BytesCompleted > f.BytesTotal {
		f.BytesCompleted = f.BytesTotal
	}
}

func markFileDone(st *State, path string, growTotal bool) {
	if path == "" {
		return
	}
	idx := matchFileIndex(st.Files, path)
	if idx < 0 {
		st.Files = append(st.Files, FileProgress{Path: path, Status: FileDone})
		if growTotal && st.FilesTotal < len(st.Files) {
			st.FilesTotal = len(st.Files)
		}
		return
	}
	f := &st.Files[idx]
	f.Status = FileDone
	if f.BytesTotal > 0 {
		f.BytesCompleted = f.BytesTotal
	}
}
