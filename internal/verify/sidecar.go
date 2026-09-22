// Package verify persists what llm-init downloaded, next to the bytes it
// downloaded, so a later boot can answer "are these the files the current
// MODEL_SOURCE asks for, and are they intact?" without reaching the
// network.
//
// The record is a sidecar JSON file rather than a central index because
// the bytes it describes live on volumes that outlive any one container
// and are shared between applications: an HF cache root mounted into
// several model apps, or a per-app model directory. Keeping the record
// beside the data means moving, copying or deleting the data does the
// right thing to the record for free, and two applications sharing one
// cache read the same answer.
//
// What it is not: a security boundary. A sidecar proves the bytes match
// what a previous pass of this process wrote, which is a defence against
// a truncated download and a reused directory, not against someone who
// can write to the volume.
package verify

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"time"
)

// SchemaVersion is the sidecar format version. A file written by a
// different version is treated as absent rather than migrated: the cost
// of being wrong is re-downloading, and the cost of misreading an old
// record is serving bytes nobody checked.
const SchemaVersion = 1

// Sidecar is the on-disk record of one completed download.
type Sidecar struct {
	Version int `json:"version"`
	// Source identifies what was asked for. A pass whose configured
	// source differs from this one must not reuse the bytes.
	Source Source `json:"source"`
	// Files are the artefacts that were written, relative to the
	// directory the sidecar is anchored on.
	Files []File `json:"files"`
	// EnginePath is what the pass that wrote this record resolved as the
	// artefact the engine loads, relative to the same anchor directory.
	// Empty means the anchor directory itself. It is recorded rather
	// than recomputed so a reused download hands the engine the exact
	// path the download itself produced.
	EnginePath  string    `json:"engine_path,omitempty"`
	CompletedAt time.Time `json:"completed_at"`
}

// Source identifies the upstream a set of bytes came from. Only the
// fields belonging to Kind are populated; the rest stay empty so the
// comparison in Equal is exact.
type Source struct {
	Kind string `json:"kind"`

	// HF fields. Revision is what the operator configured and may be a
	// moving branch; Commit is the immutable sha it resolved to when the
	// bytes were fetched. Both are recorded because they answer different
	// questions: Revision whether the request changed, Commit which bytes
	// are on disk.
	Repo     string   `json:"repo,omitempty"`
	Revision string   `json:"revision,omitempty"`
	Commit   string   `json:"commit,omitempty"`
	Endpoint string   `json:"endpoint,omitempty"`
	Include  []string `json:"include,omitempty"`
	Exclude  []string `json:"exclude,omitempty"`
	Subdir   string   `json:"subdir,omitempty"`

	// URL fields. The URL itself is never stored: a MODEL_SOURCE URL can
	// carry userinfo or a presigned query signature, and this file sits
	// on a volume other applications mount. URLSHA256 gives an exact
	// identity comparison without holding anything replayable, and
	// URLRedacted gives an operator reading the file something to
	// recognise -- scheme, host and path only, since the query is
	// exactly where a presigned signature lives.
	URLSHA256      string `json:"url_sha256,omitempty"`
	URLRedacted    string `json:"url_redacted,omitempty"`
	DeclaredSHA256 string `json:"declared_sha256,omitempty"`
}

// Source kinds. These are the channels that own their bytes on disk;
// ollama://<tag> is absent because the daemon owns those bytes and
// llm-init never sees them.
const (
	KindHF        = "hf"
	KindURL       = "url"
	KindOllamaURL = "ollama-url"
)

// File is one downloaded artefact. Path is relative to the anchor
// directory and always uses forward slashes so a record written on one
// OS reads correctly on another.
//
// SHA256 is optional: it is recorded when it was already computed (a
// #sha256= fragment that was checked anyway, or an HF LFS blob whose
// cache filename is its digest) or when VERIFY_LEVEL asked for it. ETag
// is what the upstream last advertised, kept for the remote drift check.
type File struct {
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256,omitempty"`
	ETag   string `json:"etag,omitempty"`
}

// ErrVersionMismatch reports a sidecar written by a different schema
// version. Callers treat it like a missing file.
var ErrVersionMismatch = errors.New("verify: sidecar schema version mismatch")

// Read parses the sidecar at path. A missing file returns an error
// wrapping os.ErrNotExist so callers can branch on "never downloaded
// here" without inspecting the message; a malformed or foreign-version
// file returns a plain error, which callers also treat as a miss.
func Read(path string) (Sidecar, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return Sidecar{}, fmt.Errorf("verify: %s: %w", path, os.ErrNotExist)
		}
		return Sidecar{}, fmt.Errorf("verify: read %s: %w", path, err)
	}
	var s Sidecar
	if err := json.Unmarshal(data, &s); err != nil {
		return Sidecar{}, fmt.Errorf("verify: parse %s: %w", path, err)
	}
	if s.Version != SchemaVersion {
		return Sidecar{}, fmt.Errorf("verify: %s: %w (got %d, want %d)",
			path, ErrVersionMismatch, s.Version, SchemaVersion)
	}
	return s, nil
}

// Write atomically and durably persists s at path, creating the parent
// directory if needed.
//
// The durability sequence is the one the retired manifest package used,
// and each step earns its place:
//
//  1. marshal and write to ${path}.tmp
//  2. fsync(tmp), so the bytes -- not just the metadata -- reach the
//     device before a rename names them
//  3. rename(tmp, path), atomic at the inode-pointer level
//  4. fsync(parent), so the directory entry survives a power loss right
//     after the rename
//
// Without step 4 a crash can leave the sidecar missing while the bytes
// it describes are present, which costs a re-download, or leave a stale
// record pointing at bytes that were replaced, which is worse. On
// Windows step 4 is best-effort: opening a directory fails there, and
// NTFS's journalled metadata makes the rename durable anyway.
func Write(path string, s Sidecar) error {
	s.Version = SchemaVersion
	if s.CompletedAt.IsZero() {
		s.CompletedAt = time.Now().UTC()
	} else {
		s.CompletedAt = s.CompletedAt.UTC()
	}

	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return fmt.Errorf("verify: marshal: %w", err)
	}

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("verify: mkdir %s: %w", dir, err)
	}

	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("verify: write tmp: %w", err)
	}
	if _, err := f.Write(data); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return fmt.Errorf("verify: write tmp: %w", err)
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return fmt.Errorf("verify: fsync tmp: %w", err)
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("verify: close tmp: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("verify: rename: %w", err)
	}
	if d, err := os.Open(dir); err == nil {
		_ = d.Sync()
		_ = d.Close()
	}
	return nil
}

// Remove deletes the sidecar at path, ignoring a missing file. It is
// called when the bytes it describes are about to be replaced, so a
// crash mid-download cannot leave a record claiming the old bytes are
// still there.
func Remove(path string) error {
	if err := os.Remove(path); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("verify: remove %s: %w", path, err)
	}
	return nil
}
