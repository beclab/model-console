// Command fakehf is a drop-in stand-in for the HuggingFace `hf` CLI used
// by the offline download test suite and the manual runbook. It emulates
// `hf download <repo> [file] --cache-dir <dir> [--revision <rev>] ...`:
// on success it writes the real cache layout
// (models--owner--repo/{refs,snapshots}/...) that internal/adapter/hfwrap
// reads back, and on failure it prints a stderr tail that hfwrap's
// classifier maps to a typed error (repo-not-found, gated, token-missing,
// disk-full, network, ...).
//
// Behaviour is driven entirely by env vars so a single binary covers
// every scenario:
//
//	FAKEHF_EXIT          exit code (default 0)
//	FAKEHF_STDERR        stderr tail to emit before exiting (drives
//	                     hfwrap classification on non-zero exit)
//	FAKEHF_COMMIT        40-hex commit written to refs/<ref>
//	                     (default deadbeef...; must be 40 lowercase hex)
//	FAKEHF_FILES         comma-separated snapshot files to create
//	                     (default: the positional file, else model.safetensors)
//	FAKEHF_SLEEP         duration to sleep before finishing (e.g. 100ms)
//	FAKEHF_NO_PROGRESS   when "1", suppress the tqdm-style stderr lines
//	FAKEHF_PROGRESS_SIZES
//	                     comma-separated byte sizes, one per FAKEHF_FILES
//	                     entry, driving the totals the tqdm bars count
//	                     towards. The snapshot files stay small either
//	                     way — this knob exists so a test can play a
//	                     multi-gigabyte download without writing one.
//	FAKEHF_ATTEMPT_FILE  path to a counter file; combined with
//	                     FAKEHF_FAIL_TIMES, the first N invocations fail
//	                     with a network error and later ones succeed
//	FAKEHF_FAIL_TIMES    number of leading invocations that fail (needs
//	                     FAKEHF_ATTEMPT_FILE)
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

func main() {
	args := os.Args[1:]
	if len(args) == 0 || args[0] != "download" {
		fmt.Fprintln(os.Stderr, "fakehf: expected `download` subcommand")
		os.Exit(20)
	}

	repo, file, cacheDir, revision := parseArgs(args[1:])

	if sleep := os.Getenv("FAKEHF_SLEEP"); sleep != "" {
		if d, err := time.ParseDuration(sleep); err == nil {
			time.Sleep(d)
		}
	}

	maybeTransientFailure() // os.Exit(1) on a simulated transient failure

	if msg := os.Getenv("FAKEHF_STDERR"); msg != "" {
		fmt.Fprintln(os.Stderr, msg)
	}
	if code := envInt("FAKEHF_EXIT", 0); code != 0 {
		os.Exit(code)
	}

	if repo == "" || cacheDir == "" {
		fmt.Fprintln(os.Stderr, "fakehf: missing repo or --cache-dir")
		os.Exit(20)
	}

	commit := os.Getenv("FAKEHF_COMMIT")
	if commit == "" {
		commit = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
	}
	if err := writeSnapshot(cacheDir, repo, revision, commit, file); err != nil {
		fmt.Fprintln(os.Stderr, "fakehf: "+err.Error())
		os.Exit(60)
	}
	emitProgress(file)
}

// parseArgs extracts the repo, optional positional file, --cache-dir,
// and --revision from the `hf download` argv (everything after the
// `download` token).
func parseArgs(a []string) (repo, file, cacheDir, revision string) {
	var positional []string
	for i := 0; i < len(a); i++ {
		switch {
		case a[i] == "--cache-dir" && i+1 < len(a):
			cacheDir = a[i+1]
			i++
		case a[i] == "--revision" && i+1 < len(a):
			revision = a[i+1]
			i++
		case a[i] == "--include" && i+1 < len(a):
			i++ // pattern; ignored by the fake
		case a[i] == "--max-workers" && i+1 < len(a):
			i++
		case a[i] == "--force-download":
			// no-op for the fake
		case strings.HasPrefix(a[i], "--"):
			// unknown flag with possible value; skip value if present
			if i+1 < len(a) && !strings.HasPrefix(a[i+1], "--") {
				i++
			}
		default:
			positional = append(positional, a[i])
		}
	}
	if len(positional) > 0 {
		repo = positional[0]
	}
	if len(positional) > 1 {
		file = positional[1]
	}
	return repo, file, cacheDir, revision
}

// maybeTransientFailure implements the FAKEHF_ATTEMPT_FILE / FAKEHF_FAIL_TIMES
// "fail the first N invocations, then succeed" contract. It calls os.Exit(1)
// when this invocation should fail; otherwise it returns normally.
func maybeTransientFailure() {
	counter := os.Getenv("FAKEHF_ATTEMPT_FILE")
	failTimes := envInt("FAKEHF_FAIL_TIMES", 0)
	if counter == "" || failTimes <= 0 {
		return
	}
	n := 0
	if b, err := os.ReadFile(counter); err == nil {
		n, _ = strconv.Atoi(strings.TrimSpace(string(b)))
	}
	n++
	_ = os.WriteFile(counter, []byte(strconv.Itoa(n)), 0o644)
	if n <= failTimes {
		fmt.Fprintln(os.Stderr, "ConnectionError: simulated transient network failure")
		os.Exit(1)
	}
}

// writeSnapshot creates the HF cache layout the runner reads back.
func writeSnapshot(cacheRoot, repo, revision, commit, file string) error {
	repoDir := filepath.Join(cacheRoot, "models--"+strings.ReplaceAll(repo, "/", "--"))
	snapDir := filepath.Join(repoDir, "snapshots", commit)
	if err := os.MkdirAll(snapDir, 0o755); err != nil {
		return err
	}

	files := snapshotFiles(file)
	for _, f := range files {
		dst := filepath.Join(snapDir, filepath.FromSlash(f))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(dst, []byte("fake weights for "+repo+"/"+f+"\n"), 0o644); err != nil {
			return err
		}
	}

	// Only HF writes refs/<ref> when the revision is not already a SHA;
	// mirror that so ReadRefCommit resolves the commit.
	ref := revision
	if ref == "" {
		ref = "main"
	}
	if !isHexSHA(ref) {
		refDir := filepath.Join(repoDir, "refs")
		if err := os.MkdirAll(refDir, 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(refDir, ref), []byte(commit), 0o644); err != nil {
			return err
		}
	}
	return nil
}

func snapshotFiles(positional string) []string {
	if v := os.Getenv("FAKEHF_FILES"); v != "" {
		var out []string
		for _, f := range strings.Split(v, ",") {
			if f = strings.TrimSpace(f); f != "" {
				out = append(out, f)
			}
		}
		if len(out) > 0 {
			return out
		}
	}
	if positional != "" {
		return []string{positional}
	}
	return []string{"model.safetensors"}
}

// emitProgress writes tqdm-style lines so the scanner / aggregator path
// is exercised end to end, one bar per file in the download.
//
// The label is the file's base name, not its path in the repository:
// that is what huggingface_hub puts in a bar's description, and it is
// why two files with one name in different directories are
// indistinguishable from this stream alone.
func emitProgress(file string) {
	if os.Getenv("FAKEHF_NO_PROGRESS") == "1" {
		return
	}
	files := snapshotFiles(file)
	sizes := progressSizes(len(files))

	fmt.Fprintf(os.Stderr, "Fetching %d files:   0%%|          | 0/%d\n", len(files), len(files))
	for i, f := range files {
		name := filepath.Base(f)
		total := sizes[i]
		half := total / 2
		fmt.Fprintf(os.Stderr, "%s:  50%%|#####     | %s/%s [00:00<00:00, 1.00MB/s]\n",
			name, fmtSizeof(half), fmtSizeof(total))
		fmt.Fprintf(os.Stderr, "%s: 100%%|##########| %s/%s [00:01<00:00, 1.00MB/s]\n",
			name, fmtSizeof(total), fmtSizeof(total))
	}
	fmt.Fprintf(os.Stderr, "Fetching %d files: 100%%|##########| %d/%d\n", len(files), len(files), len(files))
}

// progressSizes returns one bar total per file, defaulting to 1 MB.
func progressSizes(n int) []int64 {
	out := make([]int64, n)
	for i := range out {
		out[i] = 1_000_000
	}
	for i, s := range strings.Split(os.Getenv("FAKEHF_PROGRESS_SIZES"), ",") {
		if i >= n {
			break
		}
		if v, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64); err == nil && v > 0 {
			out[i] = v
		}
	}
	return out
}

// fmtSizeof reproduces tqdm.format_sizeof: three significant digits,
// SI units, no divisor override. 8475259708 prints as "8.48G", which is
// what makes a bar's total an approximation of the file's real size.
func fmtSizeof(n int64) string {
	v := float64(n)
	for _, unit := range []string{"", "k", "M", "G", "T"} {
		if v < 999.5 {
			switch {
			case unit == "":
				return strconv.FormatInt(int64(v), 10)
			case v < 9.995:
				return fmt.Sprintf("%.2f%s", v, unit)
			case v < 99.95:
				return fmt.Sprintf("%.1f%s", v, unit)
			default:
				return fmt.Sprintf("%.0f%s", v, unit)
			}
		}
		v /= 1000
	}
	return fmt.Sprintf("%.1fP", v)
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func isHexSHA(s string) bool {
	if len(s) != 40 {
		return false
	}
	for _, r := range s {
		switch {
		case r >= '0' && r <= '9':
		case r >= 'a' && r <= 'f':
		default:
			return false
		}
	}
	return true
}
