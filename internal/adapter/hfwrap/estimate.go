package hfwrap

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// RepoEstimate is the Hub-tree size of one hf:// source: every file
// this request will fetch, after --include / --exclude / --subdir.
type RepoEstimate struct {
	Bytes int64
	Files int
	// Complete is true when the tree answered and every exact
	// (non-glob) include matched at least one file. A glob-only or
	// whole-repo request is complete when at least one file matched.
	// Callers that pin a progress denominator require Complete && Bytes > 0.
	Complete bool
	// Entries is each matched file, in tree order. Used to seed the
	// dashboard file list before the first transfer starts.
	Entries []TreeEntry
}

// EstimateRepo sums remote sizes for the files an hf:// source will
// fetch. An empty include list is the whole repository (minus exclude
// and outside --subdir). The tree walk is recursive so files in
// subdirectories participate.
func EstimateRepo(ctx context.Context, cfg Config, include, exclude []string, subdir string) (RepoEstimate, error) {
	var zero RepoEstimate
	if cfg.Repo == "" {
		return zero, nil
	}
	entries, _, err := fetchTree(ctx, cfg, true)
	if err != nil {
		return zero, err
	}

	var matched []TreeEntry
	for _, e := range entries {
		if e.Path == "" || !UnderSubdir(e.Path, subdir) {
			continue
		}
		if !MatchPatterns(e.Path, include, exclude) {
			continue
		}
		matched = append(matched, e)
	}
	if len(matched) == 0 {
		return zero, nil
	}

	var bytes int64
	for _, e := range matched {
		bytes += e.Size
	}
	return RepoEstimate{
		Bytes:    bytes,
		Files:    len(matched),
		Complete: bytes > 0 && everyExactIncludeMatched(include, matched),
		Entries:  matched,
	}, nil
}

func everyExactIncludeMatched(include []string, matched []TreeEntry) bool {
	for _, p := range include {
		p = strings.TrimSpace(p)
		if p == "" || looksLikeGlob(p) {
			continue
		}
		found := false
		for _, e := range matched {
			if matchPattern(p, e.Path) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// EstimateRepoFiles sums remote sizes for exact filenames in patterns via
// the HF Hub tree API. patterns are typically --include values (exact
// basenames). allMatched is true when every non-empty pattern matched a
// file with size > 0; total is the sum of those sizes.
//
// Empty patterns return (0, false, nil): this helper names specific
// files (HF resolve URLs). Whole-repo estimates use EstimateRepo.
func EstimateRepoFiles(ctx context.Context, cfg Config, patterns []string) (total int64, allMatched bool, err error) {
	if len(patterns) == 0 {
		return 0, false, nil
	}
	est, err := EstimateRepo(ctx, cfg, patterns, nil, "")
	if err != nil {
		return 0, false, err
	}
	return est.Bytes, est.Complete, nil
}

// httpTimeout is the client timeout for Hub metadata calls.
const httpTimeout = 60 * time.Second

// TreeEntry is one file the Hub currently serves for a revision.
type TreeEntry struct {
	// Path is relative to the repository root, with forward slashes.
	Path string
	// Size is the file's length in bytes.
	Size int64
	// SHA256 is the LFS object id, which for an LFS-tracked file is the
	// sha256 of its contents and is therefore directly comparable to
	// what the local cache records. Small files are not LFS-tracked and
	// leave this empty; the Hub only offers their git blob id, which is
	// a sha1 over different bytes and no use as a content digest.
	SHA256 string
}

// RepoTree lists what the Hub serves for a revision, and the commit that
// revision currently resolves to.
//
// This is the network half of drift detection: the local cache knows
// which commit it fetched and what each file hashed to, and comparing
// the two is the only way to notice that a branch has moved since. The
// commit alone is not enough to act on -- a revision whose README
// changed resolves differently while every weight file is untouched --
// so callers compare the per-file digests and use the commit for the
// log line.
func RepoTree(ctx context.Context, cfg Config) (entries []TreeEntry, commit string, err error) {
	if cfg.Repo == "" {
		return nil, "", fmt.Errorf("hfwrap tree: no repo")
	}
	return fetchTree(ctx, cfg, true)
}

// fetchTree reads a revision's file listing, following the Hub's
// pagination to the end, and returns the commit the first page reported.
//
// Following it is not optional for a caller that acts on absence. The
// Hub caps a page at a few hundred entries and hands back a `Link:
// rel="next"`, so a single request against a large repository returns a
// prefix of the tree -- and a drift check reading that prefix concludes
// the files it cannot see were deleted upstream, which under
// VERIFY_ON_DRIFT=follow re-downloads the whole model to land the same
// bytes. The budget estimate that shares this code is unharmed either
// way, since a pattern it fails to match only costs it the progress
// pin; the cost is asymmetric, not the correctness.
//
// maxTreePages bounds the loop rather than trusting the server to stop
// offering a next page.
func fetchTree(ctx context.Context, cfg Config, recursive bool) (entries []TreeEntry, commit string, err error) {
	endpoint := strings.TrimSuffix(cfg.Endpoint, "/")
	if endpoint == "" {
		endpoint = "https://huggingface.co"
	}
	rev := cfg.Revision
	if rev == "" {
		rev = defaultHFRef
	}
	next := fmt.Sprintf("%s/api/models/%s/tree/%s", endpoint, cfg.Repo, url.PathEscape(rev))
	if recursive {
		next += "?recursive=true"
	}

	client := &http.Client{Timeout: httpTimeout}
	for page := 0; next != "" && page < maxTreePages; page++ {
		var raw []treeItem
		header, err := getTreePage(ctx, client, cfg.Token, next, &raw)
		if err != nil {
			return nil, "", err
		}
		for _, e := range raw {
			if e.Type != "" && e.Type != "file" {
				continue
			}
			te := TreeEntry{Path: e.Path, Size: e.Size}
			if e.LFS != nil {
				te.SHA256 = e.LFS.OID
			}
			entries = append(entries, te)
		}
		if commit == "" {
			// The commit is a response header rather than a body field,
			// and an endpoint that omits it leaves it empty rather than
			// failing: it only says which revision a drift was against.
			commit = header.Get("X-Repo-Commit")
		}
		next = nextTreePage(header)
	}
	return entries, commit, nil
}

// maxTreePages stops the pagination loop. At the Hub's page size this
// is far more files than any model repository holds, so reaching it
// means the server is repeating itself.
const maxTreePages = 100

type treeItem struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
	Type string `json:"type"`
	LFS  *struct {
		OID string `json:"oid"`
	} `json:"lfs"`
}

// getTreePage performs one authenticated GET and decodes the body into
// dst, returning the response headers the caller needs for the commit
// and the next link.
func getTreePage(ctx context.Context, client *http.Client, token, apiURL string, dst any) (http.Header, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, apiURL, http.NoBody)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("hfwrap tree: GET %s: status %d: %s",
			apiURL, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	if err := json.NewDecoder(resp.Body).Decode(dst); err != nil {
		return nil, fmt.Errorf("hfwrap tree: decode: %w", err)
	}
	return resp.Header, nil
}

// nextTreePage extracts the absolute URL of the next page from a Link
// header of the form `<https://...>; rel="next"`. Anything it cannot
// parse ends the walk, which under-reads rather than looping.
func nextTreePage(header http.Header) string {
	for _, h := range header.Values("Link") {
		for _, part := range strings.Split(h, ",") {
			segs := strings.Split(strings.TrimSpace(part), ";")
			if len(segs) < 2 {
				continue
			}
			target := strings.TrimSpace(segs[0])
			if !strings.HasPrefix(target, "<") || !strings.HasSuffix(target, ">") {
				continue
			}
			for _, attr := range segs[1:] {
				v := strings.ToLower(strings.TrimSpace(attr))
				v = strings.ReplaceAll(v, `"`, "")
				v = strings.ReplaceAll(v, "'", "")
				if v == "rel=next" {
					return strings.TrimSuffix(strings.TrimPrefix(target, "<"), ">")
				}
			}
		}
	}
	return ""
}
