package verify

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
)

// Level is how hard a pass tries to confirm the local bytes.
type Level string

const (
	// LevelSize compares each file's length. It costs a stat per file
	// and catches the failure that actually happens: a download that
	// stopped early.
	LevelSize Level = "size"
	// LevelSHA256 additionally re-hashes every file with a recorded
	// digest. It costs a full read of the model.
	LevelSHA256 Level = "sha256"
	// LevelRemote does everything LevelSHA256 does and then asks the
	// upstream whether the bytes it serves today are still the ones on
	// disk. Only this level touches the network.
	LevelRemote Level = "remote"
)

// HashesLocally reports whether the level re-reads file contents. It
// also decides whether a fresh download bothers to compute a digest to
// record: hashing bytes nobody will check back is a full read for
// nothing.
func (l Level) HashesLocally() bool {
	return l == LevelSHA256 || l == LevelRemote
}

// Verifier checks one file's size and digest. fetch.RangeDownloader
// satisfies it, which keeps the streaming hash and the size/digest
// mismatch sentinels in one place instead of growing a second copy here.
type Verifier interface {
	Verify(ctx context.Context, path string, expectedSize int64, expectedSHA string) error
}

// Report says what a local check was actually able to confirm.
//
// It exists because "verified" is not one thing. An HF snapshot mixes
// LFS weights, whose digest the cache hands us for free, with small
// files like config.json that are stored under a git object id and have
// no recorded sha256. Asking for LevelSHA256 and getting back "ok"
// would otherwise imply every file was hashed when most were not.
type Report struct {
	// Files is how many entries the sidecar described.
	Files int
	// Digest is how many of them were confirmed by re-hashing.
	Digest int
	// SizeOnly is how many could only be confirmed by length, either
	// because the level did not ask for more or because no digest was
	// ever recorded for them.
	SizeOnly int
}

// ErrEmpty reports a sidecar that lists no files. It is not a valid
// record of a download and must not be read as "nothing to check, so
// everything is fine".
var ErrEmpty = errors.New("verify: sidecar lists no files")

// CheckLocal confirms the files the sidecar describes are still on disk
// under root, at the depth level asks for. root is the directory the
// relative paths in Files are anchored on.
//
// A mismatch is returned as an error naming the file; the Report is
// still meaningful for logging on both the success and failure paths.
func (s Sidecar) CheckLocal(ctx context.Context, v Verifier, root string, level Level) (Report, error) {
	rep := Report{Files: len(s.Files)}
	if len(s.Files) == 0 {
		return rep, ErrEmpty
	}
	for _, f := range s.Files {
		path := filepath.Join(root, filepath.FromSlash(f.Path))
		wantSHA := ""
		if level.HashesLocally() && f.SHA256 != "" {
			wantSHA = f.SHA256
		}
		if err := v.Verify(ctx, path, f.Size, wantSHA); err != nil {
			return rep, fmt.Errorf("verify: %s: %w", f.Path, err)
		}
		if wantSHA != "" {
			rep.Digest++
		} else {
			rep.SizeOnly++
		}
	}
	return rep, nil
}

// Equal reports whether two sources name the same upstream request.
//
// Every field is compared, including Endpoint: pointing at a mirror is
// the operator changing their mind about where bytes come from, and
// reusing a cache filled from somewhere else would hide that. Commit is
// deliberately excluded — it records which bytes landed, not what was
// asked for, so a sidecar whose branch has since moved upstream still
// matches the same configured source. Noticing that move is the remote
// check's job, not this one's.
func (s Source) Equal(o Source) bool {
	return s.Kind == o.Kind &&
		s.Repo == o.Repo &&
		s.Revision == o.Revision &&
		s.Endpoint == o.Endpoint &&
		s.Subdir == o.Subdir &&
		s.URLSHA256 == o.URLSHA256 &&
		s.DeclaredSHA256 == o.DeclaredSHA256 &&
		slices.Equal(s.Include, o.Include) &&
		slices.Equal(s.Exclude, o.Exclude)
}

// URLIdentity is the stable identity of a URL source: the hex sha256 of
// the URL. See Source.URLSHA256 for why the URL is hashed rather than
// stored.
func URLIdentity(rawURL string) string {
	sum := sha256.Sum256([]byte(rawURL))
	return hex.EncodeToString(sum[:])
}

// Fingerprint is a short stable digest of everything Equal compares. It
// exists so a record can be filed under the request it describes when
// one directory holds several: a main GGUF and its mmproj sibling are
// two sources against the same repo and revision, differing only in
// --include, and filing both under the snapshot alone would have them
// overwrite each other and re-download on every pass.
//
// Any field added to Equal has to be added here too, which is what
// TestFingerprintAgreesWithEqual checks.
func (s Source) Fingerprint() string {
	h := sha256.New()
	for _, part := range []string{
		s.Kind, s.Repo, s.Revision, s.Endpoint, s.Subdir,
		s.URLSHA256, s.DeclaredSHA256,
	} {
		_, _ = h.Write([]byte(part))
		_, _ = h.Write([]byte{0})
	}
	for _, list := range [][]string{s.Include, s.Exclude} {
		for _, part := range list {
			_, _ = h.Write([]byte(part))
			_, _ = h.Write([]byte{0})
		}
		_, _ = h.Write([]byte{1})
	}
	return hex.EncodeToString(h.Sum(nil))[:12]
}
