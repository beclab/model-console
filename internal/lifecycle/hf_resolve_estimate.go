package lifecycle

import (
	"context"
	"net/url"
	"path"
	"strings"

	"github.com/llm-init/llm-init/internal/adapter/hfwrap"
	"github.com/llm-init/llm-init/internal/config"
)

type hfResolveGroupKey struct {
	repo     string
	revision string
}

// estimateHFResolveURLList sums file sizes for Hugging Face "resolve" URLs
// (https://huggingface.co/<owner>/<repo>/resolve/<rev>/<file>) via the Hub
// tree API. complete is true when every URL in urls parsed as HF resolve and
// every requested basename was found in the tree.
func estimateHFResolveURLList(ctx context.Context, cfg config.Config, urls []string) (total int64, complete bool) {
	type group struct {
		files map[string]struct{}
	}
	groups := make(map[hfResolveGroupKey]*group)
	var nonHF []string

	for _, raw := range urls {
		if raw == "" {
			continue
		}
		repo, rev, base, ok := parseHFResolveURL(raw, cfg.HFEndpoint)
		if !ok {
			nonHF = append(nonHF, raw)
			continue
		}
		key := hfResolveGroupKey{repo: repo, revision: rev}
		g, exists := groups[key]
		if !exists {
			g = &group{files: make(map[string]struct{})}
			groups[key] = g
		}
		g.files[base] = struct{}{}
	}
	if len(groups) == 0 || len(nonHF) > 0 {
		return 0, false
	}

	var sum int64
	for key, g := range groups {
		patterns := make([]string, 0, len(g.files))
		for base := range g.files {
			patterns = append(patterns, base)
		}
		t, all, err := hfwrap.EstimateRepoFiles(ctx, hfwrap.Config{
			Repo:     key.repo,
			Revision: key.revision,
			Token:    cfg.HFToken,
			Endpoint: cfg.HFEndpoint,
		}, patterns)
		if err != nil || !all || t <= 0 {
			return 0, false
		}
		sum += t
	}
	return sum, true
}

// parseHFResolveURL extracts owner/repo, revision, and basename from a HF
// resolve/blob download URL on a Hub-compatible host.
func parseHFResolveURL(rawURL, hfEndpoint string) (repo, revision, basename string, ok bool) {
	u, err := url.Parse(rawURL)
	if err != nil || u.Host == "" {
		return "", "", "", false
	}
	if !hfResolveHostAllowed(u.Host, hfEndpoint) {
		return "", "", "", false
	}
	parts := strings.Split(strings.Trim(u.Path, "/"), "/")
	// /<owner>/<repo>/resolve/<rev>/<file>
	resolveAt := -1
	for i, p := range parts {
		if p == "resolve" {
			resolveAt = i
			break
		}
	}
	if resolveAt < 2 || resolveAt+2 >= len(parts) {
		return "", "", "", false
	}
	repo = parts[0] + "/" + parts[1]
	revision = parts[resolveAt+1]
	basename = path.Base(strings.Join(parts[resolveAt+2:], "/"))
	if basename == "" || basename == "." || basename == ".." {
		return "", "", "", false
	}
	return repo, revision, basename, true
}

func hfResolveHostAllowed(host, hfEndpoint string) bool {
	host = strings.ToLower(strings.TrimSuffix(host, ":443"))
	allowed := map[string]struct{}{
		"huggingface.co":     {},
		"www.huggingface.co": {},
		"hf-mirror.com":      {},
	}
	if ep, err := url.Parse(strings.TrimSpace(hfEndpoint)); err == nil && ep.Host != "" {
		allowed[strings.ToLower(ep.Host)] = struct{}{}
	}
	_, ok := allowed[host]
	return ok
}
