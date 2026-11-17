package hfwrap

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
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
	// Commit is the commit Entries were listed at: cfg.Revision when it
	// is one, "" for a branch or tag. Neither the Hub nor the Olares
	// mirror puts X-Repo-Commit on tree responses, so a caller that needs
	// it resolves the revision first (ResolveCommit) and lists at that.
	Commit string
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
	var commit string
	if isHexSHA(cfg.Revision) {
		commit = cfg.Revision
	}
	return RepoEstimate{
		Bytes:    bytes,
		Files:    len(matched),
		Complete: bytes > 0 && everyExactIncludeMatched(include, matched),
		Entries:  matched,
		Commit:   commit,
	}, nil
}

// ResolveCommit returns the commit cfg.Revision names on the Hub now. A
// revision that already is a full commit is returned without a request.
//
// When the revision API does not answer and cfg.HFFile names one file,
// the X-Repo-Commit of a HEAD on that file decides instead. That is the
// request `hf download` itself makes, so a lookup fails here only when
// hf cannot reach the revision either -- and hf then quietly serves
// whatever snapshot refs/<revision> points at.
func ResolveCommit(ctx context.Context, cfg Config) (string, error) {
	if isHexSHA(cfg.Revision) {
		return cfg.Revision, nil
	}
	endpoint := hubEndpoint(cfg)
	rev := cfg.Revision
	if rev == "" {
		rev = defaultHFRef
	}
	client := &http.Client{Timeout: resolveTimeout}
	sha, err := revisionCommit(ctx, client, cfg.Token, endpoint, cfg.Repo, rev)
	if err == nil {
		return sha, nil
	}
	if cfg.HFFile == "" || looksLikeGlob(cfg.HFFile) {
		return "", err
	}
	head, headErr := headFile(ctx, client, cfg.Token, endpoint, cfg.Repo, rev, cfg.HFFile)
	if headErr != nil {
		return "", errors.Join(err, headErr)
	}
	return head.Commit, nil
}

// FileHead is what one HEAD on a file's resolve URL says about it.
type FileHead struct {
	// Commit is the X-Repo-Commit the revision resolved to.
	Commit string
	// Entry is the file as a tree listing would describe it. Size is 0
	// when the response carried no usable length, and BlobID is empty
	// unless the ETag is one the cache names blobs by.
	Entry TreeEntry
}

// HeadFile asks about cfg.HFFile at cfg.Revision with a single HEAD:
// the commit, the size, and the blob id. For a source that names one
// file this is everything the budget needs, in the one request `hf
// download` itself makes, and it does not touch /api/, which the
// Olares mirror has been seen answering with a 401 after fifteen
// seconds while serving this HEAD in one.
func HeadFile(ctx context.Context, cfg Config) (FileHead, error) {
	if cfg.HFFile == "" || looksLikeGlob(cfg.HFFile) {
		return FileHead{}, fmt.Errorf("hfwrap head: %q names no single file", cfg.HFFile)
	}
	rev := cfg.Revision
	if rev == "" {
		rev = defaultHFRef
	}
	return headFile(ctx, &http.Client{Timeout: resolveTimeout}, cfg.Token, hubEndpoint(cfg), cfg.Repo, rev, cfg.HFFile)
}

// hubEndpoint is cfg.Endpoint without a trailing slash, or the Hub.
func hubEndpoint(cfg Config) string {
	if e := strings.TrimSuffix(cfg.Endpoint, "/"); e != "" {
		return e
	}
	return "https://huggingface.co"
}

// resolveTimeout bounds one lookup. It is short next to httpTimeout
// because a lookup is retried, and a hung one holds the pass in
// phase=download.
const resolveTimeout = 20 * time.Second

func revisionCommit(ctx context.Context, client *http.Client, token, endpoint, repo, rev string) (string, error) {
	apiURL := fmt.Sprintf("%s/api/models/%s/revision/%s", endpoint, repo, url.PathEscape(rev))
	var info struct {
		SHA string `json:"sha"`
	}
	if _, err := getTreePage(ctx, client, token, apiURL, &info); err != nil {
		return "", err
	}
	if !isHexSHA(info.SHA) {
		return "", fmt.Errorf("hfwrap revision: %s resolved to %q", rev, info.SHA)
	}
	return info.SHA, nil
}

// headFile reads a HEAD on the file's resolve URL. The Hub answers it
// with a redirect to the CDN, which carries X-Repo-Commit and the
// X-Linked-* headers and is not followed; the Olares mirror answers
// 200 with a Content-Length and its own ETag.
func headFile(ctx context.Context, client *http.Client, token, endpoint, repo, rev, file string) (FileHead, error) {
	segs := strings.Split(file, "/")
	for i := range segs {
		segs[i] = url.PathEscape(segs[i])
	}
	fileURL := fmt.Sprintf("%s/%s/resolve/%s/%s", endpoint, repo, url.PathEscape(rev), strings.Join(segs, "/"))
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, fileURL, http.NoBody)
	if err != nil {
		return FileHead{}, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("Accept-Encoding", "identity")
	noFollow := *client
	noFollow.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	resp, err := noFollow.Do(req)
	if err != nil {
		return FileHead{}, err
	}
	_ = resp.Body.Close()
	if resp.StatusCode >= http.StatusBadRequest {
		return FileHead{}, fmt.Errorf("hfwrap revision: HEAD %s: status %d", fileURL, resp.StatusCode)
	}
	sha := resp.Header.Get("X-Repo-Commit")
	if !isHexSHA(sha) {
		return FileHead{}, fmt.Errorf("hfwrap revision: HEAD %s: X-Repo-Commit %q", fileURL, sha)
	}
	entry := TreeEntry{Path: file}
	if n, err := strconv.ParseInt(resp.Header.Get("X-Linked-Size"), 10, 64); err == nil && n > 0 {
		entry.Size = n
	} else if resp.StatusCode == http.StatusOK && resp.ContentLength > 0 {
		// A redirect's Content-Length is its own body's, not the file's.
		entry.Size = resp.ContentLength
	}
	etag := resp.Header.Get("X-Linked-ETag")
	if etag == "" {
		etag = resp.Header.Get("ETag")
	}
	if id := strings.Trim(strings.TrimPrefix(etag, "W/"), `"`); isSHA256(id) || isHexSHA(id) {
		entry.BlobID = id
		if isSHA256(id) {
			entry.SHA256 = id
		}
	}
	return FileHead{Commit: sha, Entry: entry}, nil
}

func isSHA256(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, r := range s {
		if (r < '0' || r > '9') && (r < 'a' || r > 'f') {
			return false
		}
	}
	return true
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
	// BlobID is the name the local HF cache files this content under in
	// blobs/: the LFS sha256 when there is one, the git blob id
	// otherwise. Two revisions share a blob exactly when they share
	// this id, so it says whether a file is already on disk without
	// reading it.
	BlobID string
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
	endpoint := hubEndpoint(cfg)
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
			te := TreeEntry{Path: e.Path, Size: e.Size, BlobID: e.OID}
			if e.LFS != nil {
				te.SHA256 = e.LFS.OID
				te.BlobID = e.LFS.OID
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
	OID  string `json:"oid"`
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
