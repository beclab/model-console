package fetch

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

// Verify checks that path exists, has expectedSize bytes, and (if
// expectedSHA is non-empty) that its sha256 matches expectedSHA.
//
// The hash is computed in a streaming fashion (defaultBufSize chunks) so
// large files do not blow up memory. expectedSHA may be the bare hex digest
// or include a "sha256:" prefix; either is accepted.
func (d *defaultDownloader) Verify(ctx context.Context, path string, expectedSize int64, expectedSHA string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("fetch: stat %s: %w", path, err)
	}
	if expectedSize > 0 && info.Size() != expectedSize {
		return fmt.Errorf("%w: got %d want %d", ErrSizeMismatch, info.Size(), expectedSize)
	}
	if expectedSHA == "" {
		return nil
	}
	want := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(expectedSHA)), "sha256:")
	got, err := hashFile(ctx, path)
	if err != nil {
		return err
	}
	if got != want {
		return fmt.Errorf("%w: got %s want %s", ErrSHA256Mismatch, got, want)
	}
	return nil
}

// HashFile returns the hex SHA-256 of path, read in chunks so a
// multi-gigabyte model never lands in memory. Callers that need a digest
// to record rather than to compare use this instead of growing a second
// hashing loop of their own.
func HashFile(ctx context.Context, path string) (string, error) {
	return hashFile(ctx, path)
}

// hashFile computes the SHA-256 of path. ctx is checked between chunks so a
// shutting-down process aborts promptly.
func hashFile(ctx context.Context, path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("fetch: open %s: %w", path, err)
	}
	defer f.Close()
	h := sha256.New()
	buf := make([]byte, defaultBufSize)
	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
		}
		n, rerr := f.Read(buf)
		if n > 0 {
			_, _ = h.Write(buf[:n])
		}
		if errors.Is(rerr, io.EOF) {
			break
		}
		if rerr != nil {
			return "", fmt.Errorf("fetch: read %s: %w", path, rerr)
		}
	}
	if errors.Is(ctx.Err(), context.Canceled) {
		return "", ctx.Err()
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
