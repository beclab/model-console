// Package hfwrap is the Go-side bridge to the `hf` CLI shipped by
// huggingface_hub.
//
// As of v1.1.0 llm-init drives `hf download` directly from Go, with the
// HF standard cache layout (`<HF_HUB_CACHE>/models--<owner>--<repo>/{
// refs,snapshots,blobs}`). The previous Python wrapper
// (internal/scripts/hf_download.py) is deleted.
//
// The runner observes the subprocess through three channels:
//
//   - stderr: tqdm bars and free-form notices; parsed by
//     scanner.go + aggregator.go into bounded progress.Sink calls.
//   - exit code: a small bounded enum (see Code* constants below)
//     translated by classify.go into typed errors with a Retriable flag
//     so the lifecycle backoff policy can stay simple.
//   - refs/<ref>: the resolved 40-char commit SHA huggingface_hub
//     writes after a successful download; we read it via
//     refs_reader.go and surface it on Result.Commit + sink.OnResolvedCommit.
//
// The package owns no state — every Run is a fresh subprocess.
package hfwrap

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/llm-init/llm-init/internal/progress"
)

// Exit codes emitted by classifyExit. Closed enum so the dashboard's
// `error_code` label stays bounded.
const (
	CodeOK            = 0
	CodeBadArgs       = 20
	CodeRepoNotFound  = 30
	CodeGated         = 31
	CodeRevision      = 32
	CodeTokenMissing  = 33
	CodeNetwork       = 40
	CodeDiskFull      = 50
	CodeInternal      = 60
	CodeProcessKilled = -1 // synthetic: subprocess died before exiting
)

const stageSpawn = "spawn"

// hfDownloadSubcmd is the `hf` CLI subcommand the runner spawns.
const hfDownloadSubcmd = "download"

// Config bundles every input the `hf` CLI invocation needs. Repo is
// required; CacheDir defaults to HF_HUB_CACHE / ~/.cache/huggingface/hub.
type Config struct {
	// HFExe is the `hf` binary path. Defaults to "hf".
	HFExe string

	// CacheDir is forwarded as `--cache-dir`. Files land at
	// `<CacheDir>/models--<owner>--<repo>/{refs,snapshots,blobs}`.
	// Empty means "let hf use HF_HUB_CACHE / default".
	CacheDir string

	// Repo is "<owner>/<name>", e.g. "Qwen/Qwen3-27B".
	Repo string

	// Revision is the commit SHA, branch, or tag. Empty means "use
	// hf's default branch (typically main)". `main` and 40-char hex
	// behave identically inside `hf download`; we still pass them
	// through for explicit audit of intent.
	Revision string

	// AllowPatterns map to `--include`. Each non-empty entry is one
	// pattern; the CLI accepts globs (e.g. `*.safetensors`).
	AllowPatterns []string

	// ExcludePatterns map to `--exclude`. Parsed from MODEL_SOURCE
	// --exclude and forwarded verbatim to hf download (e.g.
	// "openvino/**" on unified repos).
	ExcludePatterns []string

	// HFFile names a single file to download (positional arg in
	// `hf download <repo> <filename>`). Empty means full-repo
	// download. Mutually exclusive with AllowPatterns at the wire
	// level but lifecycle never sets both today.
	HFFile string

	// Token is forwarded via env (HF_TOKEN) so it never appears in
	// /proc/<pid>/cmdline.
	Token string

	// Endpoint mirrors HF_ENDPOINT for non-default mirrors.
	Endpoint string

	// MaxWorkers caps `--max-workers`. 0 means "let hf decide".
	MaxWorkers int

	// ForceReload sets `--force-download`.
	ForceReload bool

	// EnableXet opts INTO the hf_xet chunked-transfer backend. The zero
	// value (false) is the safe default: we disable hf_xet and use the
	// standard LFS download path. hf_xet fetches many byte-range chunks
	// concurrently and buffers them in RAM before flushing to disk
	// (huggingface_hub#3300); on large sharded models that spike
	// OOM-killed this Pod mid-download. LFS is slower but streams each
	// file to disk with a stable, bounded footprint — reliability over
	// throughput. Translated to HF_HUB_DISABLE_XET in buildEnv.
	EnableXet bool

	// ExtraEnv is appended to os.Environ() — useful for HF_HUB_*
	// flags or test hooks.
	ExtraEnv []string

	// Tree is the repository listing this pass already fetched to size
	// the download, reused here to name the file behind each tqdm bar.
	// huggingface_hub labels a bar with the file name alone, so two
	// files with one name in different directories are indistinguishable
	// without it. Empty is allowed: progress then keys on the raw label.
	Tree []TreeEntry
}

// Result is the post-success state. Path is the snapshot directory
// (or single-file path inside it) the engine should load. Commit is
// the resolved 40-char hex SHA read from `refs/<ref>` (or the input
// Revision when it was already a SHA).
type Result struct {
	Path   string `json:"path"`
	Status string `json:"status"`
	Commit string `json:"commit"`
}

// Error is the typed error returned when the subprocess exits non-zero.
type Error struct {
	Code      int
	Retriable bool
	Stage     string
	Message   string
}

func (e *Error) Error() string {
	r := "permanent"
	if e.Retriable {
		r = "retriable"
	}
	return fmt.Sprintf("hfwrap exit=%d (%s) stage=%s: %s", e.Code, r, e.Stage, e.Message)
}

// IsRetriable reports whether the underlying *Error is retriable.
func IsRetriable(err error) bool {
	if err == nil {
		return false
	}
	var e *Error
	if errors.As(err, &e) {
		return e.Retriable
	}
	return false
}

// Run executes `hf download` synchronously with ctx cancellation
// support. Progress flows into sink; the returned Result is valid only
// when err == nil.
func Run(ctx context.Context, cfg Config, sink progress.Sink) (Result, error) {
	if cfg.Repo == "" {
		return Result{}, &Error{Code: CodeBadArgs, Stage: "args",
			Message: "Repo is required"}
	}
	if cfg.HFExe == "" {
		cfg.HFExe = "hf"
	}
	if sink == nil {
		sink = progress.NopSink{}
	}

	// xet bypasses tqdm; let the operator know progress will be
	// invisible until the download completes.
	for _, e := range cfg.ExtraEnv {
		if e == "HF_HUB_ENABLE_HF_TRANSFER=1" {
			sink.OnNote("xet acceleration: tqdm progress unavailable")
			break
		}
	}
	// Fire OnResolvedCommit eagerly when the operator pinned a
	// 40-char hex SHA. The post-download read in refs_reader is
	// idempotent so a second call with the same value is a no-op.
	if isHexSHA(cfg.Revision) {
		sink.OnResolvedCommit(cfg.Revision)
	}

	args := buildArgs(cfg)
	cmd := exec.CommandContext(ctx, cfg.HFExe, args...)
	cmd.Env = buildEnv(cfg)

	slog.Debug("hfwrap spawn",
		slog.String("hf", cfg.HFExe),
		slog.Any("argv", args),
		slog.String("repo", cfg.Repo),
		slog.String("revision", cfg.Revision),
		slog.String("cache_dir", cfg.CacheDir),
		slog.Int("max_workers", cfg.MaxWorkers),
		slog.Bool("force", cfg.ForceReload))
	startedAt := time.Now()

	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return Result{}, &Error{Code: CodeInternal, Stage: stageSpawn,
			Message: "stderr pipe: " + err.Error()}
	}
	// Drain stdout so the kernel pipe buffer never fills (hf prints a
	// few lines at the end summarising the snapshot path).
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return Result{}, &Error{Code: CodeInternal, Stage: stageSpawn,
			Message: "stdout pipe: " + err.Error()}
	}

	if err := cmd.Start(); err != nil {
		return Result{}, &Error{Code: CodeInternal, Stage: stageSpawn,
			Message: "start: " + err.Error()}
	}
	wd := NewWatchdog(ctx, cmd, cfg.Repo)
	defer wd.Stop()

	sink.OnPhase(progress.PhaseDownload)

	aggr := NewAggregator(sink, cfg.Tree)
	var (
		wg      sync.WaitGroup
		tailMu  sync.Mutex
		tailBuf []string
		stage   = "spawn"
	)

	wg.Add(1)
	go func() {
		defer wg.Done()
		stage = pumpStderr(stderrPipe, sink, aggr, wd, cfg.Repo, &tailMu, &tailBuf)
	}()
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, _ = io.Copy(io.Discard, stdoutPipe)
	}()

	wg.Wait()
	waitErr := cmd.Wait()
	exitCode := exitCodeOf(cmd, waitErr)
	elapsed := time.Since(startedAt)
	aggr.FlushIncomplete()

	tailMu.Lock()
	tailJoined := strings.Join(tailBuf, "\n")
	tailMu.Unlock()

	if exitCode != CodeOK {
		code, retriable := classifyExit(exitCode, tailJoined)
		// Watchdog-induced kills override classification: even if the
		// stderr tail looks like a transient repo error, the kill
		// itself is retriable on its own merit.
		if wd.Killed() {
			code = CodeProcessKilled
			retriable = true
		}
		msg := tailJoined
		if msg == "" {
			if waitErr != nil {
				msg = waitErr.Error()
			} else {
				msg = fmt.Sprintf("non-zero exit %d", exitCode)
			}
		}
		slog.Error("hfwrap failed",
			slog.String("repo", cfg.Repo),
			slog.Int("exit_code", exitCode),
			slog.Int("classified_code", code),
			slog.Bool("retriable", retriable),
			slog.Bool("watchdog_kill", wd.Killed()),
			slog.Duration("elapsed", elapsed),
			slog.String("tail", trimTail(msg, 1024)))
		return Result{}, &Error{
			Code:      code,
			Retriable: retriable,
			Stage:     stage,
			Message:   trimTail(msg, 1024),
		}
	}

	commit, refErr := ReadRefCommit(cfg.CacheDir, cfg.Repo, cfg.Revision)
	if refErr != nil {
		slog.Warn("hfwrap refs read failed",
			slog.String("repo", cfg.Repo),
			slog.String("err", refErr.Error()))
	}
	if commit == "" {
		commit = cfg.Revision
	}
	if commit != "" {
		sink.OnResolvedCommit(commit)
	}

	path := SnapshotPath(cfg.CacheDir, cfg.Repo, commit)
	if cfg.HFFile != "" && commit != "" {
		path = path + "/" + cfg.HFFile
	}

	slog.Info("hfwrap done",
		slog.String("repo", cfg.Repo),
		slog.String("commit", commit),
		slog.String("path", path),
		slog.Duration("elapsed", elapsed))

	return Result{
		Path:   path,
		Status: "ok",
		Commit: commit,
	}, nil
}

// buildArgs translates Config into the `hf download` argv. Centralised
// so unit tests can assert on exact spawn argv.
func buildArgs(cfg Config) []string {
	args := []string{hfDownloadSubcmd, cfg.Repo}
	if cfg.HFFile != "" {
		args = append(args, cfg.HFFile)
	}
	if cfg.CacheDir != "" {
		args = append(args, "--cache-dir", cfg.CacheDir)
	}
	if cfg.Revision != "" && cfg.Revision != defaultHFRef {
		args = append(args, "--revision", cfg.Revision)
	}
	for _, p := range cfg.AllowPatterns {
		if p == "" {
			continue
		}
		args = append(args, "--include", p)
	}
	for _, p := range cfg.ExcludePatterns {
		if p == "" {
			continue
		}
		args = append(args, "--exclude", p)
	}
	if cfg.MaxWorkers > 0 {
		args = append(args, "--max-workers", strconv.Itoa(cfg.MaxWorkers))
	}
	if cfg.ForceReload {
		args = append(args, "--force-download")
	}
	return args
}

// buildEnv composes the subprocess environment. We start from os.Environ
// so things like PATH, LANG, HF_HUB_CACHE survive, then layer the
// runner-specific values on top.
func buildEnv(cfg Config) []string {
	env := append([]string{}, os.Environ()...)
	if cfg.Token != "" {
		env = append(env, "HF_TOKEN="+cfg.Token)
	}
	if cfg.Endpoint != "" {
		env = append(env, "HF_ENDPOINT="+cfg.Endpoint)
	}
	// Force huggingface_hub to render tqdm progress bars even though our
	// stderr is a pipe (not a TTY). huggingface_hub's download path
	// (file_download.py) builds the bar with disable=is_tqdm_disabled():
	// that returns None on a normal run, and tqdm(disable=None) silently
	// suppresses itself on a non-TTY — so without this the dashboard
	// shows 0/0 for the whole download. huggingface_hub's documented
	// escape hatch is TQDM_POSITION=-1, which flips is_tqdm_disabled to
	// False (force-enable). This restores byte progress for BOTH the
	// LFS and the Xet (hf_xet) backends, which share the same tqdm bar.
	// Placed before ExtraEnv so an operator can still override it.
	env = append(env, "TQDM_POSITION=-1")
	// Default to the stable LFS download path (see Config.EnableXet). Placed
	// before ExtraEnv so an operator can still force a value either way.
	if !cfg.EnableXet {
		env = append(env, "HF_HUB_DISABLE_XET=1")
	}
	env = append(env, cfg.ExtraEnv...)
	return env
}

// pumpStderr reads `hf download` stderr line-by-line (CR or LF
// boundary), feeds parsed progress events into the aggregator, and
// retains the last several lines so classifyExit has something to
// pattern-match against.
func pumpStderr(r io.Reader, sink progress.Sink, aggr *Aggregator, wd *Watchdog, repo string, tailMu *sync.Mutex, tail *[]string) string {
	const tailMax = 32
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64<<10), 1<<20)
	scanner.Split(scanLinesOrCR)
	stage := "stderr"
	for scanner.Scan() {
		raw := scanner.Text()
		clean := strings.TrimSpace(stripANSIEscapes(raw))
		if clean == "" {
			continue
		}
		wd.Touch()

		if ev, ok := parseHFProgressLine(clean); ok {
			aggr.OnProgress(ev)
			stage = "progress"
			continue
		}
		if n, ok := parseFetchingFilesCount(clean); ok {
			sink.OnFilesTotal(n)
			stage = "progress"
			continue
		}
		// Non-progress notice: useful at debug, retain in tail for
		// classifier.
		slog.Debug("hfwrap stderr",
			slog.String("repo", repo),
			slog.String("line", trimTail(clean, 256)))
		tailMu.Lock()
		*tail = append(*tail, clean)
		if len(*tail) > tailMax {
			*tail = (*tail)[len(*tail)-tailMax:]
		}
		tailMu.Unlock()
	}
	if err := scanner.Err(); err != nil {
		sink.OnError(fmt.Errorf("hfwrap stderr scan: %w", err))
	}
	return stage
}

// exitCodeOf extracts the subprocess exit code, mapping signal-killed
// and other unusual termination paths to CodeProcessKilled.
func exitCodeOf(cmd *exec.Cmd, waitErr error) int {
	if waitErr == nil {
		return CodeOK
	}
	var ee *exec.ExitError
	if errors.As(waitErr, &ee) {
		if code := ee.ExitCode(); code >= 0 {
			return code
		}
	}
	if cmd.ProcessState != nil {
		if code := cmd.ProcessState.ExitCode(); code >= 0 {
			return code
		}
	}
	return CodeProcessKilled
}

// trimTail truncates s to at most n bytes (UTF-8 boundary unaware,
// good enough for ASCII-heavy hf stderr).
func trimTail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
