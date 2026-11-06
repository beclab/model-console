package comfyconsole

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

// Hugging Face's CDN throttles each connection (about 1.6 MiB/s measured
// on a home line) while several connections add up, so large models are
// fetched as N byte ranges written in place into <dest>.part. Progress is
// checkpointed to <dest>.part.json so a restart resumes every range.

// errNoRanges means the upstream ignored Range; the caller falls back to
// the single-stream downloader.
var errNoRanges = errors.New("upstream does not serve byte ranges")

type segment struct {
	Start int64 `json:"start"`
	End   int64 `json:"end"` // inclusive
	Next  int64 `json:"next"`
}

type partState struct {
	Size     int64      `json:"size"`
	Segments []*segment `json:"segments"`
}

func newPartState(size int64, conns int) *partState {
	st := &partState{Size: size}
	step := (size + int64(conns) - 1) / int64(conns)
	for start := int64(0); start < size; start += step {
		end := min(start+step, size) - 1
		st.Segments = append(st.Segments, &segment{Start: start, End: end, Next: start})
	}
	return st
}

func loadPartState(file string, size int64, conns int) *partState {
	raw, err := os.ReadFile(file)
	if err == nil {
		var st partState
		if json.Unmarshal(raw, &st) == nil && st.Size == size && len(st.Segments) > 0 {
			return &st
		}
	}
	return newPartState(size, conns)
}

// downloadParallel fetches m into dest with conns concurrent range requests.
// onBytes receives the absolute byte count so far.
func (c *Console) downloadParallel(ctx context.Context, m Model, dest string, onBytes func(int64)) error {
	part := dest + ".part"
	stateFile := part + ".json"
	conns := c.opts.Connections

	st := loadPartState(stateFile, m.Size, conns)
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(part, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	if info, err := f.Stat(); err != nil {
		return err
	} else if info.Size() != m.Size {
		if err := f.Truncate(m.Size); err != nil {
			return fmt.Errorf("preallocate: %w", err)
		}
	}

	var mu sync.Mutex // guards segment Next values while checkpointing
	var have atomic.Int64
	for _, s := range st.Segments {
		have.Add(s.Next - s.Start)
	}
	onBytes(have.Load())

	save := func() {
		mu.Lock()
		raw, _ := json.Marshal(st)
		mu.Unlock()
		tmp := stateFile + ".tmp"
		if os.WriteFile(tmp, raw, 0o644) == nil {
			_ = os.Rename(tmp, stateFile)
		}
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	errs := make(chan error, len(st.Segments))
	var wg sync.WaitGroup
	for _, s := range st.Segments {
		if s.Next > s.End {
			continue
		}
		wg.Add(1)
		go func(s *segment) {
			defer wg.Done()
			if err := c.fetchSegment(ctx, m, f, s, &mu, func(n int64) { onBytes(have.Add(n)) }); err != nil {
				errs <- err
				cancel()
			}
		}(s)
	}

	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	tick := time.NewTicker(5 * time.Second)
	defer tick.Stop()
loop:
	for {
		select {
		case <-done:
			break loop
		case <-tick.C:
			save()
		}
	}
	save()
	close(errs)
	if err := <-errs; err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(part, dest); err != nil {
		return err
	}
	_ = os.Remove(stateFile)
	return nil
}

// fetchSegment streams one range, retrying from where it stopped.
func (c *Console) fetchSegment(ctx context.Context, m Model, f *os.File, s *segment, mu *sync.Mutex, onBytes func(int64)) error {
	buf := make([]byte, 1<<20)
	failures := 0
	for {
		mu.Lock()
		next := s.Next
		mu.Unlock()
		if next > s.End {
			return nil
		}
		err := c.fetchRange(ctx, m, f, next, s, mu, buf, onBytes)
		if err == nil || errors.Is(err, errNoRanges) || ctx.Err() != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return err
		}
		failures++
		if failures > c.opts.SegmentRetries {
			return fmt.Errorf("range %d-%d: %w", s.Start, s.End, err)
		}
		wait := min(time.Second<<min(failures, 6), time.Minute)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(wait):
		}
	}
}

func (c *Console) fetchRange(ctx context.Context, m Model, f *os.File, from int64, s *segment, mu *sync.Mutex, buf []byte, onBytes func(int64)) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, m.URL, nil)
	if err != nil {
		return err
	}
	for k, v := range c.opts.Headers {
		req.Header[k] = v
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", from, s.End))
	resp, err := c.opts.DownloadClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusPartialContent:
	case http.StatusOK:
		return errNoRanges
	default:
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("GET %s: HTTP %d: %s", m.URL, resp.StatusCode, body)
	}
	off := from
	for off <= s.End {
		n, rerr := resp.Body.Read(buf[:min(int64(len(buf)), s.End-off+1)])
		if n > 0 {
			if _, werr := f.WriteAt(buf[:n], off); werr != nil {
				return werr
			}
			off += int64(n)
			mu.Lock()
			s.Next = off
			mu.Unlock()
			onBytes(int64(n))
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return rerr
		}
	}
	if off <= s.End {
		return io.ErrUnexpectedEOF
	}
	return nil
}
