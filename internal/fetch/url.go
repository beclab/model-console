package fetch

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/llm-init/llm-init/internal/progress"
)

// URLClient drives a single-file download from an arbitrary HTTPS source.
// It composes RangeDownloader + Verify and adds the EXPECTED_SHA256 hook
// for URL sources. No HF-specific redirect behavior here;
// direct URLs do not need to strip Authorization on cross-domain hops
// because they should not have been carrying one in the first place.
type URLClient struct {
	Downloader RangeDownloader
}

// NewURLClient returns a URLClient backed by the default RangeDownloader.
func NewURLClient() *URLClient {
	return &URLClient{Downloader: New(nil)}
}

// Fetch downloads url to dest, optionally verifying SHA256.
//
// expectedSize <= 0 disables size pre-check (HEAD-derived size is still
// used internally). expectedSHA == "" disables hash verification, falling
// back to size-only validation. When SHA verification fails, the resulting
// dest file is removed so a retry has a clean slate.
func (c *URLClient) Fetch(ctx context.Context, url, dest string, expectedSize int64, expectedSHA string, sink progress.Sink) error {
	if c.Downloader == nil {
		c.Downloader = New(nil)
	}
	if url == "" {
		return errors.New("fetch: url is required")
	}
	if dest == "" {
		return errors.New("fetch: dest is required")
	}
	if err := c.Downloader.Download(ctx, RangeOptions{
		URL:          url,
		DestPath:     dest,
		ExpectedSize: expectedSize,
	}, sink); err != nil {
		return err
	}
	if err := c.Downloader.Verify(ctx, dest, expectedSize, expectedSHA); err != nil {
		// On hash mismatch the file is corrupt and must be removed so
		// the lifecycle-level retry sees a clean state. Wrap to keep
		// errors.Is(err, ErrSHA256Mismatch) intact.
		if errors.Is(err, ErrSHA256Mismatch) {
			_ = removeIfExists(dest)
			return fmt.Errorf("fetch: url verify failed: %w", err)
		}
		return err
	}
	return nil
}

// FetchWithHeaders mirrors Fetch but lets callers attach extra request
// headers (User-Agent, custom auth, etc.). Used by HFSource.
func (c *URLClient) FetchWithHeaders(ctx context.Context, url, dest string, expectedSize int64, expectedSHA string, headers http.Header, sink progress.Sink) error {
	if c.Downloader == nil {
		c.Downloader = New(nil)
	}
	if err := c.Downloader.Download(ctx, RangeOptions{
		URL:          url,
		DestPath:     dest,
		ExpectedSize: expectedSize,
		Headers:      headers,
	}, sink); err != nil {
		return err
	}
	if err := c.Downloader.Verify(ctx, dest, expectedSize, expectedSHA); err != nil {
		if errors.Is(err, ErrSHA256Mismatch) {
			_ = removeIfExists(dest)
		}
		return err
	}
	return nil
}
