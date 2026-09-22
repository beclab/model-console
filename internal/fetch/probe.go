package fetch

import (
	"context"
	"net/http"
	"strconv"
)

// ProbeContentLength issues HEAD against url and returns Content-Length when
// the server reports it. size <= 0 means unknown (HEAD refused, missing
// header, or retriable transport error treated as unknown).
func ProbeContentLength(ctx context.Context, client *http.Client, url string) (int64, error) {
	size, _, err := Probe(ctx, client, url)
	if err != nil {
		return 0, err
	}
	return size, nil
}

// Probe issues HEAD against url and reports what the upstream says it
// would serve today: the object's length and its ETag, either of which
// may be absent (0 / ""). It is the same call Download makes before it
// decides whether to resume, exposed so a caller can ask the question
// without transferring anything.
func Probe(ctx context.Context, client *http.Client, url string) (size int64, etag string, err error) {
	if client == nil {
		client = newDefaultClient()
	}
	d := New(client).(*defaultDownloader)
	return d.probe(ctx, RangeOptions{URL: url})
}

// SumProbeContentLength HEADs each URL and sums positive sizes. ok is true
// when at least one URL returned a positive size. allKnown is true only when
// every non-empty URL returned a positive size (none skipped or unknown).
func SumProbeContentLength(ctx context.Context, client *http.Client, urls []string) (total int64, ok bool, allKnown bool) {
	need := 0
	known := 0
	for _, u := range urls {
		if u == "" {
			continue
		}
		need++
		size, err := ProbeContentLength(ctx, client, u)
		if err != nil {
			continue
		}
		if size > 0 {
			total += size
			ok = true
			known++
		}
	}
	allKnown = need > 0 && known == need
	return total, ok, allKnown
}

// ParseContentLength parses a Content-Length header value.
func ParseContentLength(cl string) int64 {
	if cl == "" {
		return -1
	}
	v, err := strconv.ParseInt(cl, 10, 64)
	if err != nil || v < 0 {
		return -1
	}
	return v
}
