package lifecycle

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/llm-init/llm-init/internal/adapter/hfwrap"
	"github.com/llm-init/llm-init/internal/config"
)

// resolveHFEnginePath converts hfwrap.Result.Path (whatever hf download
// resolved) into the path engine wrappers should load.
//
// Unified HF repos (e.g. beclab/embeddinggemma-300m with onnx/ and openvino/)
// land files under .../snapshots/<sha>/... while Embed expects
// EMBED_MODEL_DIR at .../snapshots/<sha>/onnx. --include/--exclude control
// which blobs hf download fetches; --subdir (llm-init-only, not passed to hf
// CLI) selects the subfolder written to /run/llm-init/model_path.
//
// Return cases:
//   - No --subdir, res.Path non-empty → res.Path unchanged (legacy behaviour).
//   - No --subdir, res.Path empty     → "" (writeSentinelHF becomes a no-op).
//   - --subdir set                    → Join(snapshotRoot, subdir), must exist
//     and be a directory or we fail ensure so the operator sees a clear error
//     instead of the engine crashing later.
func resolveHFEnginePath(main config.ModelSource, res hfwrap.Result) (string, error) {
	if res.Path == "" {
		// hfwrap failed to resolve a path (some tests/mocks omit it). Without
		// --subdir we preserve the old sentinel no-op; with --subdir we must
		// fail because there is nowhere to append the subdirectory.
		if main.HFSubdir != "" {
			return "", fmt.Errorf("hf engine path: empty download path")
		}
		return "", nil
	}
	if main.HFSubdir == "" {
		return res.Path, nil
	}

	// hfSnapshotRoot normalises to the snapshot directory before joining
	// subdir. See hfSnapshotRoot for the single --include file case.
	snap := hfSnapshotRoot(main, res.Path)
	out := filepath.Join(snap, filepath.FromSlash(main.HFSubdir))

	// Fail fast at ensure time: embed.sh exports EMBED_MODEL_DIR from
	// model_path and embed_server validates assets under that directory.
	st, err := os.Stat(out)
	if err != nil {
		return "", fmt.Errorf("hf subdir %q not found under snapshot: %w", main.HFSubdir, err)
	}
	if !st.IsDir() {
		return "", fmt.Errorf("hf subdir %q is not a directory under snapshot", main.HFSubdir)
	}
	return out, nil
}

// hfSnapshotRoot returns the HF snapshot *directory* used as the base for
// --subdir joins.
//
// hfwrap.Result.Path depends on how many --include patterns MODEL_SOURCE had:
//
//   - 0 includes (whole repo) or ≥2 includes (pattern mode):
//     res.Path is already the snapshot root → return as-is.
//   - exactly 1 --include (single-file mode, used by llama.cpp GGUF):
//     res.Path is .../snapshots/<sha>/model.gguf → return Dir(...) so
//     --subdir (rare for this mode) would still be relative to the snapshot,
//     not the file basename.
func hfSnapshotRoot(main config.ModelSource, downloadPath string) string {
	if len(main.HFInclude) == 1 {
		return filepath.Dir(downloadPath)
	}
	return downloadPath
}
