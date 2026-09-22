package hfwrap

import (
	"sync"

	"github.com/llm-init/llm-init/internal/progress"
)

// Aggregator turns a stream of HFProgressEvent instances into bounded
// progress.Sink calls. It owns a per-file state so the "byte-delta
// against this file's last n" invariant survives N-way parallel tqdm
// interleaving (huggingface_hub's max_workers > 1).
//
// Which file a bar belongs to is the labelIndex's answer, not the bar's
// label: huggingface_hub labels a bar with the file name alone, so two
// files with one name in different directories print bars that read the
// same. Named through the tree, they get a state each and interleave as
// safely as two distinct names do. Unnameable bars fall back to keying
// on the label, where a bar that rewinds is read as the next file
// rather than as this one going backwards.
//
// No emit gate / ticker. Front-end polls /api/progress at 1Hz; the
// sink absorbs every parsed update without throttling so SpeedBytesPerSec
// can compute the EMA against fresh samples.
type Aggregator struct {
	sink progress.Sink
	tree *labelIndex

	mu       sync.Mutex
	byPath   map[string]*aggFileState
	byLabel  map[string]*aggFileState
	aggBytes int64
	aggTotal int64
}

type aggFileState struct {
	// path is what the sink is told this transfer is. It is the
	// repository path when the tree could name the bar, and the raw
	// tqdm label when it could not.
	path       string
	downloaded int64
	total      int64
	started    bool
	completed  bool
}

// NewAggregator returns a fresh Aggregator wired to sink. sink must be
// non-nil; pass progress.NopSink{} when there is nothing to track. tree
// is the repository listing for this download, used to name the file
// behind each bar; nil or empty is allowed and keys progress on the raw
// tqdm labels.
func NewAggregator(sink progress.Sink, tree []TreeEntry) *Aggregator {
	return &Aggregator{
		sink:    sink,
		tree:    newLabelIndex(tree),
		byPath:  make(map[string]*aggFileState),
		byLabel: make(map[string]*aggFileState),
	}
}

// OnProgress is the entry point per parsed tqdm line. Outer aggregate
// bars do not contribute bytes (they double-count files) but they pin
// files_total via OnFilesTotal. Empty labels are dropped (xet sometimes
// emits unlabelled bars we cannot route per-file).
func (a *Aggregator) OnProgress(ev HFProgressEvent) {
	if ev.Label == "" {
		return
	}
	if hfIsOuterAggregateLabel(ev.Label) {
		if n, ok := parseFetchingFilesCount(ev.Label); ok {
			a.sink.OnFilesTotal(n)
		} else if ev.Total > 0 {
			a.sink.OnFilesTotal(int(ev.Total))
		}
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()

	fs := a.route(ev)

	if !fs.started {
		fs.started = true
		fs.total = ev.Total
		if ev.Total > 0 {
			a.aggTotal += ev.Total
		}
		a.sink.OnFileStart(fs.path, ev.Total)
	} else if ev.Total > fs.total && ev.Total > 0 {
		// tqdm's total may settle late (HEAD redirect, content-length
		// learned only after first chunk). Top up BytesTotal so the
		// dashboard denominator catches up; never shrink.
		delta := ev.Total - fs.total
		fs.total = ev.Total
		a.aggTotal += delta
		a.sink.OnFileStart(fs.path, delta)
	}

	if ev.Downloaded > fs.downloaded {
		delta := ev.Downloaded - fs.downloaded
		fs.downloaded = ev.Downloaded
		a.aggBytes += delta
		if fbs, ok := a.sink.(progress.FileByteSink); ok {
			fbs.OnFileBytes(fs.path, delta)
		} else {
			a.sink.OnBytes(delta)
		}
	}

	if !fs.completed && fs.total > 0 && fs.downloaded >= fs.total {
		fs.completed = true
		a.sink.OnFileDone(fs.path)
	}
}

// route picks the state this line belongs to, creating it on the first
// tick of a bar. Called under a.mu.
func (a *Aggregator) route(ev HFProgressEvent) *aggFileState {
	if path, ok := a.tree.lookup(ev.Label, ev.Total); ok {
		fs := a.byPath[path]
		if fs == nil {
			fs = &aggFileState{path: path}
			a.byPath[path] = fs
		}
		return fs
	}
	if fs := a.byLabel[ev.Label]; fs != nil && !isNextBar(fs, ev) {
		return fs
	}
	fs := &aggFileState{path: ev.Label}
	a.byLabel[ev.Label] = fs
	return fs
}

// isNextBar reports whether ev belongs to a different file than the bar
// fs is tracking, for the labels the tree could not name.
//
// A tqdm bar only ever counts up, so a line that rewinds is the next
// file reusing the label rather than this one going backwards. So is a
// line counting towards a different total after this bar finished.
func isNextBar(fs *aggFileState, ev HFProgressEvent) bool {
	if ev.Downloaded < fs.downloaded {
		return true
	}
	return fs.completed && ev.Total > 0 && ev.Total != fs.total
}

// FlushIncomplete marks in-flight files abandoned. It does not emit
// OnFileDone: those files did not finish, and treating them as done
// made files_completed jump when the subprocess died. Called by the
// runner when stderr closes.
func (a *Aggregator) FlushIncomplete() {
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, m := range []map[string]*aggFileState{a.byPath, a.byLabel} {
		for _, fs := range m {
			if fs.started && !fs.completed {
				fs.completed = true
			}
		}
	}
}
