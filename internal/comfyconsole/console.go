package comfyconsole

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/llm-init/llm-init/internal/fetch"
	"github.com/llm-init/llm-init/internal/progress"
)

// Model states reported by /api/status.
const (
	StatePending     = "pending"
	StateDownloading = "downloading"
	StateVerifying   = "verifying"
	StateDone        = "done"
	StateError       = "error"
)

// Backend states reported by /api/status.
const (
	BackendDown = "down"
	BackendUp   = "up"
)

// markerDir holds one file per verified model under the models root, so a
// restart does not re-hash tens of GB. A marker only counts while the
// model's size and mtime still match what was hashed.
const markerDir = ".llm-init-verified"

// Options wires a Console. Zero intervals take the defaults below.
type Options struct {
	Config     *AppConfig
	ModelsDir  string
	ComfyURL   string // e.g. http://127.0.0.1:8188
	Headers    http.Header
	Downloader fetch.RangeDownloader
	Client     *http.Client // for the /system_stats probe

	// Models of at least ParallelMin bytes are fetched over Connections
	// concurrent byte ranges through DownloadClient; smaller ones, or an
	// upstream without range support, use Downloader.
	Connections    int
	ParallelMin    int64
	SegmentRetries int
	DownloadClient *http.Client

	ProbeInterval  time.Duration
	RecheckEvery   time.Duration // how often finished models are re-stat'ed
	RetryBackoff   time.Duration
	RetryBackoffMx time.Duration
	Logger         *slog.Logger
}

// Console tracks the models and the ComfyUI backend.
type Console struct {
	opts    Options
	mu      sync.Mutex
	items   []*itemState
	backend atomic.Bool
}

type itemState struct {
	model      Model
	state      string
	downloaded int64
	err        string
}

// New returns a Console with defaults applied.
func New(opts Options) *Console {
	if opts.Downloader == nil {
		opts.Downloader = fetch.New(nil)
	}
	if opts.Client == nil {
		opts.Client = &http.Client{Timeout: 3 * time.Second}
	}
	if opts.Connections <= 0 {
		opts.Connections = 8
	}
	if opts.ParallelMin <= 0 {
		opts.ParallelMin = 64 << 20
	}
	if opts.SegmentRetries <= 0 {
		opts.SegmentRetries = 30
	}
	if opts.DownloadClient == nil {
		// No overall timeout: a range of a 20 GB file streams for minutes.
		opts.DownloadClient = &http.Client{Transport: &http.Transport{
			Proxy:                 http.ProxyFromEnvironment,
			ResponseHeaderTimeout: 60 * time.Second,
			IdleConnTimeout:       90 * time.Second,
			MaxIdleConnsPerHost:   32,
		}}
	}
	if opts.ProbeInterval <= 0 {
		opts.ProbeInterval = 2 * time.Second
	}
	if opts.RecheckEvery <= 0 {
		opts.RecheckEvery = time.Minute
	}
	if opts.RetryBackoff <= 0 {
		opts.RetryBackoff = 10 * time.Second
	}
	if opts.RetryBackoffMx <= 0 {
		opts.RetryBackoffMx = 5 * time.Minute
	}
	if opts.Logger == nil {
		opts.Logger = slog.Default()
	}
	c := &Console{opts: opts}
	for _, m := range opts.Config.Models {
		c.items = append(c.items, &itemState{model: m, state: StatePending})
	}
	return c
}

// Run drives downloads and the backend probe until ctx ends.
func (c *Console) Run(ctx context.Context) {
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); c.probeLoop(ctx) }()
	go func() { defer wg.Done(); c.modelLoop(ctx) }()
	wg.Wait()
}

func (c *Console) probeLoop(ctx context.Context) {
	t := time.NewTicker(c.opts.ProbeInterval)
	defer t.Stop()
	for {
		up := c.probe(ctx)
		if prev := c.backend.Swap(up); prev != up {
			c.opts.Logger.Info("comfyui backend state changed", "up", up)
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

func (c *Console) probe(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.opts.ComfyURL+"/system_stats", nil)
	if err != nil {
		return false
	}
	resp, err := c.opts.Client.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// modelLoop ensures every model in order, then keeps re-checking so a model
// deleted or moved from the ComfyUI side is fetched again.
func (c *Console) modelLoop(ctx context.Context) {
	failures := 0
	for {
		failed := false
		for _, it := range c.items {
			if ctx.Err() != nil {
				return
			}
			if err := c.ensure(ctx, it); err != nil {
				if ctx.Err() != nil {
					return
				}
				failed = true
				c.opts.Logger.Error("model ensure failed", "model", it.model.RelPath(), "err", err)
				c.set(it, StateError, -1, err.Error())
			}
		}
		wait := c.opts.RecheckEvery
		if failed {
			wait = min(c.opts.RetryBackoff<<min(failures, 8), c.opts.RetryBackoffMx)
			failures++
		} else {
			failures = 0
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(wait):
		}
	}
}

func (c *Console) ensure(ctx context.Context, it *itemState) error {
	m := it.model
	dest := filepath.Join(c.opts.ModelsDir, filepath.FromSlash(m.RelPath()))
	marker := filepath.Join(c.opts.ModelsDir, markerDir, filepath.FromSlash(m.RelPath())+".json")

	if info, err := os.Stat(dest); err == nil && info.Size() == m.Size {
		if markerMatches(marker, m, info) {
			c.set(it, StateDone, m.Size, "")
			return nil
		}
		if err := c.verify(ctx, it, dest, marker); err == nil {
			return nil
		} else if !errors.Is(err, fetch.ErrSHA256Mismatch) {
			return err
		}
		// Corrupt file: drop it and download again.
		c.opts.Logger.Warn("model failed verification, re-downloading", "model", m.RelPath())
		_ = os.Remove(dest)
	}
	_ = os.Remove(marker)

	if err := c.download(ctx, it, dest); err != nil {
		return fmt.Errorf("download: %w", err)
	}
	if err := c.verify(ctx, it, dest, marker); err != nil {
		if errors.Is(err, fetch.ErrSHA256Mismatch) {
			_ = os.Remove(dest)
		}
		return err
	}
	return nil
}

func (c *Console) download(ctx context.Context, it *itemState, dest string) error {
	m := it.model
	if c.opts.Connections > 1 && m.Size >= c.opts.ParallelMin {
		c.set(it, StateDownloading, 0, "")
		err := c.downloadParallel(ctx, m, dest, func(n int64) { c.set(it, StateDownloading, n, "") })
		if !errors.Is(err, errNoRanges) {
			return err
		}
		c.opts.Logger.Warn("upstream ignored Range, downloading in one stream", "model", m.RelPath())
		_ = os.Remove(dest + ".part.json")
		_ = os.Remove(dest + ".part")
	}
	var have int64
	if info, err := os.Stat(dest + ".part"); err == nil {
		have = info.Size()
	}
	c.set(it, StateDownloading, have, "")
	return c.opts.Downloader.Download(ctx, fetch.RangeOptions{
		URL:          m.URL,
		DestPath:     dest,
		ExpectedSize: m.Size,
		Headers:      c.opts.Headers,
	}, &bytesSink{c: c, it: it})
}

func (c *Console) verify(ctx context.Context, it *itemState, dest, marker string) error {
	m := it.model
	c.set(it, StateVerifying, m.Size, "")
	if err := c.opts.Downloader.Verify(ctx, dest, m.Size, m.SHA256); err != nil {
		return fmt.Errorf("verify: %w", err)
	}
	info, err := os.Stat(dest)
	if err != nil {
		return fmt.Errorf("verify: %w", err)
	}
	if err := writeMarker(marker, m, info); err != nil {
		// Only costs a re-hash on the next start.
		c.opts.Logger.Warn("write verify marker", "model", m.RelPath(), "err", err)
	}
	c.set(it, StateDone, m.Size, "")
	c.opts.Logger.Info("model ready", "model", m.RelPath())
	return nil
}

// set updates an item. downloaded < 0 keeps the current byte count.
func (c *Console) set(it *itemState, state string, downloaded int64, errMsg string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	it.state = state
	if downloaded >= 0 {
		it.downloaded = downloaded
	}
	it.err = errMsg
}

type bytesSink struct {
	progress.NopSink
	c  *Console
	it *itemState
}

func (s *bytesSink) OnBytes(delta int64) {
	s.c.mu.Lock()
	s.it.downloaded += delta
	s.c.mu.Unlock()
}

type markerFile struct {
	SHA256  string `json:"sha256"`
	Size    int64  `json:"size"`
	ModTime int64  `json:"mtime_unix_nano"`
}

func markerMatches(file string, m Model, info os.FileInfo) bool {
	raw, err := os.ReadFile(file)
	if err != nil {
		return false
	}
	var mk markerFile
	if json.Unmarshal(raw, &mk) != nil {
		return false
	}
	return mk.SHA256 == m.SHA256 && mk.Size == info.Size() && mk.ModTime == info.ModTime().UnixNano()
}

func writeMarker(file string, m Model, info os.FileInfo) error {
	if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
		return err
	}
	raw, err := json.Marshal(markerFile{SHA256: m.SHA256, Size: info.Size(), ModTime: info.ModTime().UnixNano()})
	if err != nil {
		return err
	}
	tmp := file + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, file)
}
