package progress

// Sink is the producer-facing interface used by fetch and adapter packages.
//
// Implementations route events into a Manager (typical) or aggregate child
// sinks for tests. Sink methods must be safe for concurrent use.
type Sink interface {
	OnFileStart(path string, totalBytes int64)
	OnBytes(delta int64)
	OnFileDone(path string)
	// OnFilesTotal announces how many files the current producer batch
	// will fetch (HF CLI "Fetching N files"). Raises files_total; never
	// shrinks it. Does not pin the pass: later OnFileStart events may
	// still raise the count (a second hf download in the same ensure).
	OnFilesTotal(n int)
	OnError(err error)
	OnPhase(p Phase)

	// OnResolvedCommit announces the 40-char HF commit SHA the
	// download is associated with. May fire twice during one HF run:
	// once at spawn time (when the operator pinned an explicit SHA)
	// and once after success (resolved from refs/<ref> for branches /
	// tags / "main"). Empty sha is silently ignored.
	OnResolvedCommit(sha string)

	// OnNote sets the dashboard banner text. Empty string clears it.
	// Used by hfwrap to flag xet-mode (HF_HUB_ENABLE_HF_TRANSFER=1)
	// where tqdm progress is unavailable.
	OnNote(text string)
}

// managerSink translates Sink calls into Manager.Update operations. It is
// returned by memoryManager.Sink().
type managerSink struct {
	m *memoryManager
}

var _ FileByteSink = (*managerSink)(nil)

func (s *managerSink) OnFileStart(path string, totalBytes int64) {
	s.m.mu.Lock()
	if path != "" {
		if path != s.m.rateFilePath {
			s.m.rateFilePath = path
			s.m.rateFileBytes = totalBytes
		} else if totalBytes > s.m.rateFileBytes {
			s.m.rateFileBytes = totalBytes
		}
	}
	bytesPinned := s.m.bytesBudgetActive
	filesPinned := s.m.filesBudgetActive
	seenCount := 0
	if path != "" && !filesPinned {
		if s.m.seenFiles == nil {
			s.m.seenFiles = make(map[string]struct{})
		}
		s.m.seenFiles[path] = struct{}{}
		seenCount = len(s.m.seenFiles)
	}
	s.m.mu.Unlock()

	// totalBytes < 0 means "unknown"; do not grow BytesTotal.
	growBytes := !bytesPinned && totalBytes > 0
	if !growBytes && path == "" && seenCount == 0 {
		return
	}
	s.m.Update(func(st *State) {
		if growBytes {
			st.BytesTotal += totalBytes
		}
		if seenCount > st.FilesTotal {
			st.FilesTotal = seenCount
		}
		if path != "" {
			markFileStart(st, path, totalBytes, !filesPinned)
		}
	})
}

func (s *managerSink) OnBytes(delta int64) {
	if delta == 0 {
		return
	}
	s.m.mu.Lock()
	pinned := s.m.filesBudgetActive
	s.m.mu.Unlock()
	s.m.Update(func(st *State) {
		st.BytesCompleted += delta
		if st.CurrentFile != "" {
			addFileBytes(st, st.CurrentFile, delta, !pinned)
		}
	})
}

// OnFileBytes implements FileByteSink. Use this when the producer
// knows which file moved (HF tqdm labels); OnBytes attributes to
// CurrentFile and is wrong when workers interleave.
func (s *managerSink) OnFileBytes(path string, delta int64) {
	if delta == 0 && path == "" {
		return
	}
	s.m.mu.Lock()
	pinned := s.m.filesBudgetActive
	s.m.mu.Unlock()
	s.m.Update(func(st *State) {
		if delta != 0 {
			st.BytesCompleted += delta
		}
		if path != "" {
			st.CurrentFile = path
			addFileBytes(st, path, delta, !pinned)
		}
	})
}

func (s *managerSink) OnFileDone(path string) {
	if path != "" {
		s.m.mu.Lock()
		if s.m.doneFiles == nil {
			s.m.doneFiles = make(map[string]struct{})
		}
		if _, seen := s.m.doneFiles[path]; seen {
			s.m.mu.Unlock()
			s.m.Update(func(st *State) {
				if st.CurrentFile == path {
					st.CurrentFile = ""
				}
			})
			return
		}
		s.m.doneFiles[path] = struct{}{}
		if s.m.rateFilePath == path {
			s.m.rateFilePath = ""
			s.m.rateFileBytes = 0
		}
		pinned := s.m.filesBudgetActive
		s.m.mu.Unlock()
		s.m.Update(func(st *State) {
			st.FilesCompleted++
			if st.FilesTotal > 0 && st.FilesCompleted > st.FilesTotal {
				st.FilesCompleted = st.FilesTotal
			}
			markFileDone(st, path, !pinned)
			if st.CurrentFile == path {
				st.CurrentFile = ""
			}
		})
		return
	}
	s.m.Update(func(st *State) {
		st.FilesCompleted++
		if st.FilesTotal > 0 && st.FilesCompleted > st.FilesTotal {
			st.FilesCompleted = st.FilesTotal
		}
		if path != "" && st.CurrentFile == path {
			st.CurrentFile = ""
		}
	})
}

func (s *managerSink) OnFilesTotal(n int) {
	if n <= 0 {
		return
	}
	s.m.Update(func(st *State) {
		if n > st.FilesTotal {
			st.FilesTotal = n
		}
	})
}

// OnError is the transport-error sink: fetch's inner retry loop (and
// similar producers) fire it for every retried attempt (503 / EOF /
// connection reset).
//
// It only bumps TransportRetries. LastError is reserved for
// lifecycle.fail / engine wait failures — the Status card's "current
// failure" signal. Writing LastError here re-lit the red diagnostics
// card on every transient blip even while the download kept going
// (and undid clear-on-download-reentry for in-pass retries).
//
// Pre-v1.0.5 this also bumped RetryCount, which conflated noisy
// networks with operator-visible "lifecycle has tried again" events.
// v1.0.5 split those counters; LastError is now split the same way.
func (s *managerSink) OnError(err error) {
	if err == nil {
		return
	}
	s.m.Update(func(st *State) {
		st.TransportRetries++
	})
}

func (s *managerSink) OnPhase(p Phase) {
	s.m.Update(func(st *State) {
		st.Phase = p
	})
}

func (s *managerSink) OnResolvedCommit(sha string) {
	if sha == "" {
		return
	}
	s.m.Update(func(st *State) {
		st.ResolvedCommit = sha
	})
}

func (s *managerSink) OnNote(text string) {
	s.m.Update(func(st *State) {
		st.Note = text
	})
}

// NopSink discards every event; useful for tests of components that need a
// non-nil Sink but do not assert on progress. All methods are no-ops; see
// the Sink interface for parameter semantics.
type NopSink struct{}

// OnFileStart implements Sink.
func (NopSink) OnFileStart(string, int64) {}

// OnBytes implements Sink.
func (NopSink) OnBytes(int64) {}

// OnFileDone implements Sink.
func (NopSink) OnFileDone(string) {}

// OnFilesTotal implements Sink.
func (NopSink) OnFilesTotal(int) {}

// OnError implements Sink.
func (NopSink) OnError(error) {}

// OnPhase implements Sink.
func (NopSink) OnPhase(Phase) {}

// OnResolvedCommit implements Sink.
func (NopSink) OnResolvedCommit(string) {}

// OnNote implements Sink.
func (NopSink) OnNote(string) {}
