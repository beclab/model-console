package lifecycle

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"strings"

	"github.com/llm-init/llm-init/internal/adapter/hfwrap"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/fetch"
	"github.com/llm-init/llm-init/internal/progress"
	"github.com/llm-init/llm-init/internal/verify"
)

// driftVerdict is what asking the upstream produced.
//
// The three are deliberately separate. "The upstream serves the same
// bytes" and "the upstream could not be reached" both leave the local
// files in place, but only the first is a verified answer, and reporting
// them the same way would tell an operator their weights were confirmed
// against the Hub when nothing was confirmed at all.
type driftVerdict int

const (
	// driftNone means the upstream still serves what is on disk.
	driftNone driftVerdict = iota
	// driftFound means it serves something else.
	driftFound
	// driftUnknown means the question could not be answered.
	driftUnknown
)

func (v driftVerdict) metricResult() string {
	switch v {
	case driftFound:
		return "drift"
	case driftUnknown:
		return "drift_unknown"
	default:
		return "ok"
	}
}

// resolveDrift asks the upstream whether it still serves the recorded
// bytes and returns whether the local copy may be reused.
//
// Only LevelRemote gets here, and only after the local check has already
// passed, so a false return means "the bytes are intact but stale"
// rather than "the bytes are broken". The two lead to the same
// re-download and must not be conflated in the log: one is the operator
// tracking a moving branch, the other is a damaged volume.
//
// Reporting is the default because following is destructive in a way
// that is easy to under-estimate. A model application is usually
// serving requests by the time anybody runs a remote check, and
// re-downloading swaps the weights under it.
func (m *Manager) resolveDrift(cfg config.Config, what string, v driftVerdict, detail string) bool {
	m.observeVerify(v.metricResult(), "remote")
	switch v {
	case driftNone:
		slog.Info("upstream still serves the local bytes", slog.String("source", what))
		// Retract whatever an earlier check left about *this* source. A
		// drift notice that outlives the drift sends an operator looking
		// for a difference that is no longer there.
		m.setDriftNote(what, "")
		return true
	case driftUnknown:
		slog.Warn("could not ask the upstream whether it has moved; keeping the local bytes",
			slog.String("source", what), slog.String("err", detail))
		m.setDriftNote(what, fmt.Sprintf("远端复核未完成（%s）：保留本地模型", what))
		return true
	default:
		if cfg.Runtime.VerifyOnDrift == config.VerifyOnDriftFollow {
			slog.Warn("upstream has moved; downloading it as VERIFY_ON_DRIFT=follow asks",
				slog.String("source", what), slog.String("detail", detail))
			return false
		}
		slog.Warn("upstream has moved; keeping the local bytes as VERIFY_ON_DRIFT=report asks",
			slog.String("source", what), slog.String("detail", detail))
		m.setDriftNote(what, fmt.Sprintf("上游已变更（%s）：本地模型保持不变，需要跟随请设 VERIFY_ON_DRIFT=follow", what))
		return true
	}
}

// setDriftNote records one source's remote verdict and renders every
// verdict this pass has produced onto the dashboard banner.
//
// A model is routinely several sources -- a GGUF and its mmproj sibling,
// plus whatever ROLE=extra adds -- and the banner is one string. Writing
// it directly meant the last source checked decided what the operator
// saw: a pass where the weights had drifted and the projector had not
// ended with the projector's silence, and the finding the operator asked
// for was gone. Verdicts accumulate here and are shown together.
//
// Retracting is conditional for the same reason. Something else may own
// the banner by now -- the download path writes it too -- and a check
// with nothing to report must not clear a message it did not write.
func (m *Manager) setDriftNote(what, text string) {
	if m.pass == nil {
		m.note(text)
		return
	}
	if m.pass.driftNotes == nil {
		m.pass.driftNotes = map[string]string{}
	}
	if text == "" {
		delete(m.pass.driftNotes, what)
	} else {
		m.pass.driftNotes[what] = text
	}

	sources := make([]string, 0, len(m.pass.driftNotes))
	for s := range m.pass.driftNotes {
		sources = append(sources, s)
	}
	sort.Strings(sources)
	parts := make([]string, 0, len(sources))
	for _, s := range sources {
		parts = append(parts, m.pass.driftNotes[s])
	}
	rendered := strings.Join(parts, "；")

	previous := m.pass.driftNote
	m.pass.driftNote = rendered
	m.opts.Manager.Update(func(s *progress.State) {
		if rendered == "" && s.Note != previous {
			return
		}
		s.Note = rendered
	})
}

// note sets the dashboard banner. It is not LastError: nothing has
// failed, and a red Status card would misreport a model that is serving
// correctly from bytes an operator asked a question about.
func (m *Manager) note(text string) {
	m.opts.Manager.Update(func(s *progress.State) { s.Note = text })
}

// checkURLDrift compares what the upstream would serve now against what
// the record says was fetched.
//
// ETag is the primary signal and Content-Length the fallback, in that
// order, because a length can stay the same across a real change while
// an ETag that stays the same is the server promising the object did
// not. An upstream that offers neither cannot be checked, and says so
// rather than passing.
func checkURLDrift(ctx context.Context, ms config.ModelSource, sc verify.Sidecar) (driftVerdict, string) {
	if len(sc.Files) == 0 {
		return driftUnknown, "the record lists no files"
	}
	rec := sc.Files[0]
	size, etag, err := fetch.Probe(ctx, nil, ms.URL)
	if err != nil {
		return driftUnknown, err.Error()
	}
	switch {
	case rec.ETag != "" && etag != "":
		if etagIdentity(rec.ETag) != etagIdentity(etag) {
			return driftFound, fmt.Sprintf("ETag %s -> %s", rec.ETag, etag)
		}
		return driftNone, ""
	case size > 0 && rec.Size > 0:
		if size != rec.Size {
			return driftFound, fmt.Sprintf("Content-Length %d -> %d", rec.Size, size)
		}
		// A matching length is weak evidence, and saying so is the
		// point: the operator asked whether the upstream had changed
		// and the upstream declined to say.
		return driftUnknown, "the upstream sent no ETag; only its length could be compared"
	default:
		return driftUnknown, "the upstream sent neither an ETag nor a length"
	}
}

// etagIdentity strips the weak-comparison marker so `W/"abc"` and
// `"abc"` are recognised as the same object.
//
// The distinction the marker draws is not the one being made here. A
// weak ETag promises the representation is semantically equivalent
// rather than byte-identical, which matters to a cache deciding whether
// to combine ranges; here the question is whether a static model file
// was replaced, and a server that switches between the two spellings --
// a CDN in front of an origin usually does -- would otherwise report a
// drift on every check and re-download the model under follow.
func etagIdentity(tag string) string {
	return strings.TrimPrefix(strings.TrimSpace(tag), "W/")
}

// checkHFDrift compares the Hub's current tree against the record.
//
// The resolved commit is not the test. A revision whose README changed
// resolves to a new commit while every weight file is untouched, and
// treating that as drift would have VERIFY_ON_DRIFT=follow re-download
// gigabytes to land the same bytes. What matters is whether a file this
// source actually fetched now has a different LFS oid or length.
//
// Files the record could not digest -- config.json and friends, which
// the Hub stores under a git object id rather than an LFS oid -- are
// compared by length, which is all either side has.
func checkHFDrift(ctx context.Context, cfg config.Config, ms config.ModelSource, sc verify.Sidecar) (driftVerdict, string) {
	if len(sc.Files) == 0 {
		return driftUnknown, "the record lists no files"
	}
	entries, commit, err := hfwrap.RepoTree(ctx, hfwrap.Config{
		Repo:     ms.HFRepo,
		Revision: ms.HFRevision,
		Token:    cfg.HFToken,
		Endpoint: cfg.HFEndpoint,
	})
	if err != nil {
		return driftUnknown, err.Error()
	}
	upstream := make(map[string]hfwrap.TreeEntry, len(entries))
	for _, e := range entries {
		upstream[e.Path] = e
	}

	where := ms.HFRepo
	if commit != "" && sc.Source.Commit != "" && commit != sc.Source.Commit {
		where = fmt.Sprintf("%s %s -> %s", ms.HFRepo, sc.Source.Commit, commit)
	}
	for _, f := range sc.Files {
		up, ok := upstream[f.Path]
		if !ok {
			return driftFound, fmt.Sprintf("%s: %s is gone upstream", where, f.Path)
		}
		if f.SHA256 != "" && up.SHA256 != "" {
			if f.SHA256 != up.SHA256 {
				return driftFound, fmt.Sprintf("%s: %s now hashes to %s", where, f.Path, up.SHA256)
			}
			continue
		}
		if up.Size > 0 && f.Size > 0 && up.Size != f.Size {
			return driftFound, fmt.Sprintf("%s: %s is now %d bytes, was %d", where, f.Path, up.Size, f.Size)
		}
	}
	return driftNone, ""
}
