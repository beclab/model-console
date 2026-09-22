package config

import (
	"errors"
	"fmt"
	"net/url"
	"path/filepath"
	"strings"
	"unicode"
)

// ModelSource kind values exposed via /api/config and used by lifecycle
// to dispatch the correct fetch channel. The four values are the
// closed enum used by ModelSource.kind.
const (
	KindHF        SourceKind = "hf"
	KindOllama    SourceKind = "ollama"
	KindOllamaURL SourceKind = "ollama-url"
	KindURL       SourceKind = "url"
)

// URL scheme literals accepted in MODEL_SOURCE and validated against
// parsed URLs. KindURL ("url") is the resulting kind, distinct from the
// "http"/"https" wire schemes, so these need their own constants.
const (
	schemeHTTP  = "http"
	schemeHTTPS = "https"
)

// Role values. main / extra follow the MODEL_SOURCE position (first
// segment is main, later segments are extra); mmproj is opted into with
// the inline --role mmproj flag on an hf:// segment.
const (
	RoleMain   = "main"
	RoleMmproj = "mmproj"
	RoleExtra  = "extra"
)

// modelSourceSchemePrefixes are accepted at the start of each comma-separated
// MODEL_SOURCE segment (after optional whitespace).
var modelSourceSchemePrefixes = []string{
	"hf://",
	"ollama://",
	"http://",
	"https://",
}

// ModelSource is one resolved entry from MODEL_SOURCE.
//
// Single-source deploys produce a 1-element slice with Index == 1.
// Multi-source deploys preserve the 1-based comma-separated position.
//
// RawValue carries the original env value (including inline flags) so
// callers needing a faithful debug echo can use it. RedactedSource is
// the same value with URL userinfo masked - that's what we expose on
// /api/config.
type ModelSource struct {
	Index          int
	Kind           SourceKind
	Role           string
	RawValue       string
	RedactedSource string
	LocalPath      string

	HFRepo     string
	HFRevision string
	HFInclude  []string
	HFExclude  []string
	// HFSubdir names a subdirectory *inside* the HF snapshot that engine
	// wrappers should load. Parsed from MODEL_SOURCE --subdir (llm-init
	// only; not forwarded to hf download).
	//
	// Motivation: unified HF repos (e.g. beclab/embeddinggemma-300m) store
	// onnx/ and openvino/ under one snapshot root. --include/--exclude
	// control which files hf download fetches; HFSubdir tells lifecycle
	// which subfolder to write into /run/llm-init/model_path so
	// embed.sh sets EMBED_MODEL_DIR correctly.
	//
	// Example: HFSubdir="onnx" → model_path=.../snapshots/<sha>/onnx
	HFSubdir string

	OllamaTag string

	URL       string
	URLSha256 string
}

// ViaOllama is the convenience flag we expose on /api/config: true iff
// fetch goes through the ollama daemon (Library tag pull or URL-overload
// blob upload).
func (m ModelSource) ViaOllama() bool {
	return m.Kind == KindOllama || m.Kind == KindOllamaURL
}

// LoadModelSources reads MODEL_SOURCE and its MODEL_SOURCE_LOCAL companion.
func LoadModelSources(g Getenv) ([]ModelSource, error) {
	single := strings.TrimSpace(g("MODEL_SOURCE"))
	if strings.TrimSpace(g(removedModelSourceNumEnv)) != "" {
		return nil, errors.New("MODEL_SOURCE_NUM: removed; use comma-separated MODEL_SOURCE")
	}
	if single == "" {
		return nil, errors.New("MODEL_SOURCE: required")
	}
	return loadCommaSeparatedSources(single, strings.TrimSpace(g("MODEL_SOURCE_LOCAL")))
}

func loadCommaSeparatedSources(single, localRaw string) ([]ModelSource, error) {
	segments := splitModelSourceList(single)
	if len(segments) == 0 {
		return nil, errors.New("MODEL_SOURCE: required")
	}

	locals, err := splitModelSourceLocals(localRaw, len(segments))
	if err != nil {
		return nil, err
	}

	out := make([]ModelSource, 0, len(segments))
	mmprojCount := 0
	for i, seg := range segments {
		role := RoleExtra
		if i == 0 {
			role = RoleMain
		}
		ms, perr := ParseOneSource(i+1, seg, locals[i], role)
		if perr != nil {
			return nil, perr
		}
		if ms.Role == RoleMmproj {
			if i == 0 {
				return nil, errors.New("MODEL_SOURCE: the first source is the main source and cannot carry --role mmproj")
			}
			// lifecycle downloads the projector as a single file and hands
			// that path to the engine; anything else resolves to the whole
			// snapshot and would be ignored.
			if len(ms.HFInclude) != 1 {
				return nil, fmt.Errorf("MODEL_SOURCE: --role mmproj needs exactly one --include naming the projector file (source %d has %d)", i+1, len(ms.HFInclude))
			}
			mmprojCount++
		}
		out = append(out, ms)
	}
	if mmprojCount > 1 {
		return nil, fmt.Errorf("MODEL_SOURCE: --role mmproj is allowed on at most one source (got %d)", mmprojCount)
	}
	// Only the hf:// branch of lifecycle loads a projector, so a projector
	// alongside any other main source would download and never be used.
	if mmprojCount == 1 && out[0].Kind != KindHF {
		return nil, fmt.Errorf("MODEL_SOURCE: --role mmproj needs an hf:// main source (the first source is %s)", out[0].Kind)
	}
	return out, nil
}

// splitModelSourceList splits MODEL_SOURCE at commas followed by an accepted
// source scheme.
func splitModelSourceList(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	var out []string
	start := 0
	for i := 0; i < len(raw); i++ {
		if raw[i] != ',' {
			continue
		}
		rest := raw[i+1:]
		trimmed := strings.TrimLeftFunc(rest, unicode.IsSpace)
		if !hasModelSourceSchemePrefix(trimmed) {
			continue
		}
		j := i + 1 + len(rest) - len(trimmed)
		if seg := strings.TrimSpace(raw[start:i]); seg != "" {
			out = append(out, seg)
		}
		start = j
		i = j - 1
	}
	if seg := strings.TrimSpace(raw[start:]); seg != "" {
		out = append(out, seg)
	}
	return out
}

func hasModelSourceSchemePrefix(s string) bool {
	for _, p := range modelSourceSchemePrefixes {
		if strings.HasPrefix(s, p) {
			return true
		}
	}
	return false
}

// splitModelSourceLocals aligns MODEL_SOURCE_LOCAL with comma-separated
// MODEL_SOURCE segments (1:1). Empty segments are allowed for non-URL kinds.
func splitModelSourceLocals(raw string, n int) ([]string, error) {
	if n < 1 {
		return nil, errors.New("MODEL_SOURCE: internal error: zero segments")
	}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return make([]string, n), nil
	}
	parts := strings.Split(raw, ",")
	locals := make([]string, 0, len(parts))
	for _, p := range parts {
		locals = append(locals, strings.TrimSpace(p))
	}
	if len(locals) != n {
		return nil, fmt.Errorf(
			"MODEL_SOURCE_LOCAL: got %d path(s) but MODEL_SOURCE has %d comma-separated source(s); provide one LOCAL per source (empty for hf:// and ollama://)",
			len(locals), n)
	}
	return locals, nil
}

// ParseOneSource parses one MODEL_SOURCE segment with its aligned local path
// and positional role. role defaults to "main" when empty; an hf:// segment
// carrying --role mmproj overrides it.
//
// raw must be of the form `<scheme>://<body> [<inline-flags>...]`.
// Inline flags are shlex-tokenised, so quoting/escaping rules from
// ENGINE_ARGS apply.
func ParseOneSource(index int, raw, local, role string) (ModelSource, error) {
	if role == "" {
		role = RoleMain
	}
	if role != RoleMain && role != RoleExtra {
		return ModelSource{}, fmt.Errorf("MODEL_SOURCE: positional role %q is not allowed (allowed: %s, %s; mmproj is opted into with --role mmproj)", role, RoleMain, RoleExtra)
	}
	tokens, err := shlex(raw)
	if err != nil {
		return ModelSource{}, fmt.Errorf("MODEL_SOURCE: %w", err)
	}
	if len(tokens) == 0 {
		return ModelSource{}, errors.New("MODEL_SOURCE: empty after tokenisation")
	}
	head := tokens[0]
	flags := tokens[1:]

	scheme, body, ok := strings.Cut(head, "://")
	if !ok || scheme == "" || body == "" {
		return ModelSource{}, fmt.Errorf("MODEL_SOURCE: %q must be of the form <scheme>://<body> (scheme in {hf, ollama, http, https})", head)
	}

	ms := ModelSource{
		Index:    index,
		Role:     role,
		RawValue: raw,
	}
	switch strings.ToLower(scheme) {
	case string(KindHF):
		if err := parseHFSource(&ms, body, flags); err != nil {
			return ModelSource{}, err
		}
	case string(KindOllama):
		if err := parseOllamaSource(&ms, body, flags); err != nil {
			return ModelSource{}, err
		}
	case schemeHTTP, schemeHTTPS:
		if err := parseURLSource(&ms, head, flags, local); err != nil {
			return ModelSource{}, err
		}
	default:
		return ModelSource{}, fmt.Errorf("MODEL_SOURCE: unsupported scheme %q (allowed: hf, ollama, http, https)", scheme)
	}

	// http(s):// must have _LOCAL; other kinds: ignore _LOCAL silently
	// (HF cache layout / ollama daemon decide the on-disk path).
	if ms.Kind == KindURL && local == "" {
		return ModelSource{}, fmt.Errorf("MODEL_SOURCE_LOCAL: required for https?:// sources (got MODEL_SOURCE=%s)", redactURL(head))
	}
	if ms.Kind == KindURL {
		ms.LocalPath = local
	}

	ms.RedactedSource = redactRaw(raw)
	return ms, nil
}

func parseHFSource(ms *ModelSource, body string, flags []string) error {
	ms.Kind = KindHF
	if _, err := parseHFRepo("MODEL_SOURCE", body); err != nil {
		return fmt.Errorf("MODEL_SOURCE: hf body %q must be owner/repo", body)
	}
	ms.HFRepo = body
	for i := 0; i < len(flags); i++ {
		f := flags[i]
		switch {
		case f == "--include":
			if i+1 >= len(flags) {
				return errors.New("MODEL_SOURCE: --include needs a value")
			}
			ms.HFInclude = append(ms.HFInclude, flags[i+1])
			i++
		case strings.HasPrefix(f, "--include="):
			ms.HFInclude = append(ms.HFInclude, strings.TrimPrefix(f, "--include="))
		case f == "--exclude":
			if i+1 >= len(flags) {
				return errors.New("MODEL_SOURCE: --exclude needs a value")
			}
			ms.HFExclude = append(ms.HFExclude, flags[i+1])
			i++
		case strings.HasPrefix(f, "--exclude="):
			ms.HFExclude = append(ms.HFExclude, strings.TrimPrefix(f, "--exclude="))
		case f == "--revision":
			if i+1 >= len(flags) {
				return errors.New("MODEL_SOURCE: --revision needs a value")
			}
			ms.HFRevision = flags[i+1]
			i++
		case strings.HasPrefix(f, "--revision="):
			ms.HFRevision = strings.TrimPrefix(f, "--revision=")
		// --subdir: llm-init-only. After hf download, lifecycle joins
		// snapshot/<subdir> and writes that into model_path (see
		// lifecycle.resolveHFEnginePath). Not passed to hf download.
		case f == "--subdir":
			if i+1 >= len(flags) {
				return errors.New("MODEL_SOURCE: --subdir needs a value")
			}
			if err := setHFSubdir(ms, flags[i+1]); err != nil {
				return err
			}
			i++
		case strings.HasPrefix(f, "--subdir="):
			if err := setHFSubdir(ms, strings.TrimPrefix(f, "--subdir=")); err != nil {
				return err
			}
		// --role: llm-init-only, mmproj is the only accepted value. It
		// marks the llama.cpp vision projector, which ensureHF downloads
		// inline and which never lands in extra_model_path. Not passed to
		// hf download.
		case f == "--role":
			if i+1 >= len(flags) {
				return errors.New("MODEL_SOURCE: --role needs a value")
			}
			if err := setHFRole(ms, flags[i+1]); err != nil {
				return err
			}
			i++
		case strings.HasPrefix(f, "--role="):
			if err := setHFRole(ms, strings.TrimPrefix(f, "--role=")); err != nil {
				return err
			}
		case f == "--endpoint" || strings.HasPrefix(f, "--endpoint="):
			return errors.New("MODEL_SOURCE: --endpoint is not allowed inline; set the deployment-level HF_ENDPOINT env so all hf:// sources share one mirror")
		default:
			return fmt.Errorf("MODEL_SOURCE: unsupported hf:// flag %q (allowed: --include, --exclude, --revision, --subdir, --role)", f)
		}
	}
	return nil
}

// setHFRole promotes an hf:// segment to ROLE=mmproj. Positional roles
// (main / extra) cannot be requested inline: main is the first segment by
// definition and extra is what every later segment already is.
func setHFRole(ms *ModelSource, raw string) error {
	v := strings.TrimSpace(raw)
	if v != RoleMmproj {
		return fmt.Errorf("MODEL_SOURCE: --role %q is not allowed (only %q; main is the first source and later sources are %q)", raw, RoleMmproj, RoleExtra)
	}
	if ms.Role == RoleMmproj {
		return errors.New("MODEL_SOURCE: --role specified more than once")
	}
	ms.Role = RoleMmproj
	return nil
}

// setHFSubdir stores at most one --subdir per hf:// MODEL_SOURCE entry.
// Validation rejects path traversal and absolute paths so model_path
// cannot escape the HF snapshot tree via ../ segments.
func setHFSubdir(ms *ModelSource, raw string) error {
	if ms.HFSubdir != "" {
		return errors.New("MODEL_SOURCE: --subdir specified more than once")
	}
	clean, err := validateHFSubdir(raw)
	if err != nil {
		return err
	}
	ms.HFSubdir = clean
	return nil
}

// validateHFSubdir normalises a --subdir value to a safe relative path.
// Allowed: "onnx", "openvino", "nested/dir". Rejected: "", ".", "..",
// "/abs", "foo/../bar", duplicate slashes that collapse to empty segments.
func validateHFSubdir(raw string) (string, error) {
	v := strings.TrimSpace(raw)
	if v == "" {
		return "", errors.New("MODEL_SOURCE: --subdir needs a value")
	}
	// Must be relative to the snapshot root; absolute paths would bypass
	// the intended sandbox under .../snapshots/<sha>/.
	if filepath.IsAbs(v) || strings.HasPrefix(v, "/") || strings.HasPrefix(v, `\`) {
		return "", fmt.Errorf("MODEL_SOURCE: --subdir %q must be relative to the snapshot root", raw)
	}
	clean := filepath.ToSlash(filepath.Clean(v))
	if clean == "." || clean == ".." {
		return "", fmt.Errorf("MODEL_SOURCE: --subdir %q is not allowed", raw)
	}
	// Reject each path segment so "foo/../bar" cannot escape the snapshot.
	for _, seg := range strings.Split(clean, "/") {
		if seg == ".." {
			return "", fmt.Errorf("MODEL_SOURCE: --subdir %q is not allowed", raw)
		}
		if seg == "" {
			return "", fmt.Errorf("MODEL_SOURCE: --subdir %q is not allowed", raw)
		}
	}
	return filepath.FromSlash(clean), nil
}

// parseOllamaSource decides between Library tag (KindOllama) and URL
// overload (KindOllamaURL) by the heuristic in
// ollama overload rules:
//  1. body contains "://" -> URL form
//  2. body contains both "." and "/" -> URL form
//  3. otherwise -> Library tag.
func parseOllamaSource(ms *ModelSource, body string, flags []string) error {
	if len(flags) > 0 {
		return fmt.Errorf("MODEL_SOURCE: ollama:// does not accept inline flags; got %v", flags)
	}
	if strings.Contains(body, "://") {
		return setOllamaURL(ms, body)
	}
	if strings.Contains(body, ".") && strings.Contains(body, "/") {
		return setOllamaURL(ms, "https://"+body)
	}
	ms.Kind = KindOllama
	ms.OllamaTag = body
	return nil
}

func setOllamaURL(ms *ModelSource, urlStr string) error {
	u, err := url.Parse(urlStr)
	if err != nil {
		return fmt.Errorf("MODEL_SOURCE: ollama:// URL %q invalid: %w", urlStr, err)
	}
	if u.Scheme != schemeHTTP && u.Scheme != schemeHTTPS {
		return fmt.Errorf("MODEL_SOURCE: ollama:// URL must be http or https, got %q", u.Scheme)
	}
	if u.Host == "" {
		return fmt.Errorf("MODEL_SOURCE: ollama:// URL %q has empty host", urlStr)
	}
	ms.Kind = KindOllamaURL
	ms.URL = urlStr
	if frag := u.Fragment; frag != "" {
		if v, ok := strings.CutPrefix(frag, "sha256="); ok {
			if _, err := parseSHA256("#sha256=", v); err != nil {
				return fmt.Errorf("MODEL_SOURCE: %w", err)
			}
			ms.URLSha256 = v
		}
	}
	return nil
}

func parseURLSource(ms *ModelSource, head string, flags []string, local string) error {
	_ = local
	if len(flags) > 0 {
		return fmt.Errorf("MODEL_SOURCE: https?:// does not accept inline flags; got %v (use #sha256= URL fragment instead)", flags)
	}
	u, err := url.Parse(head)
	if err != nil {
		return fmt.Errorf("MODEL_SOURCE: %q invalid: %w", head, err)
	}
	if u.Scheme != schemeHTTP && u.Scheme != schemeHTTPS {
		return fmt.Errorf("MODEL_SOURCE: scheme must be http or https, got %q", u.Scheme)
	}
	if u.Host == "" {
		return fmt.Errorf("MODEL_SOURCE: %q has empty host", head)
	}
	ms.Kind = KindURL
	ms.URL = head
	if frag := u.Fragment; frag != "" {
		v, ok := strings.CutPrefix(frag, "sha256=")
		if !ok {
			return fmt.Errorf("MODEL_SOURCE: only #sha256= URL fragment is supported, got %q", frag)
		}
		if _, err := parseSHA256("#sha256=", v); err != nil {
			return fmt.Errorf("MODEL_SOURCE: %w", err)
		}
		ms.URLSha256 = v
	}
	return nil
}

// redactURL masks userinfo in an http(s) URL. Non-URLs and URLs without
// embedded credentials are returned unchanged.
func redactURL(s string) string {
	u, err := url.Parse(s)
	if err != nil || u.User == nil {
		return s
	}
	u.User = url.User("REDACTED")
	return u.String()
}

// redactRaw redacts every shlex token that looks like an http(s) URL in
// raw, preserving the rest verbatim. We don't redact HF_TOKEN because
// MODEL_SOURCE never carries it (HF_TOKEN is a deployment-level env);
// the only secret-bearing path is a URL with userinfo.
func redactRaw(raw string) string {
	tokens, err := shlex(raw)
	if err != nil {
		return raw
	}
	out := make([]string, 0, len(tokens))
	for _, t := range tokens {
		switch {
		case strings.HasPrefix(t, "http://") || strings.HasPrefix(t, "https://"):
			out = append(out, redactURL(t))
		case strings.HasPrefix(t, "ollama://http://") || strings.HasPrefix(t, "ollama://https://"):
			scheme := "ollama://https://"
			if strings.HasPrefix(t, "ollama://http://") {
				scheme = "ollama://http://"
			}
			inner := strings.TrimPrefix(t, scheme)
			out = append(out, scheme+redactURLBody(inner))
		default:
			out = append(out, t)
		}
	}
	return strings.Join(out, " ")
}

func redactURLBody(body string) string {
	if at := strings.IndexByte(body, '@'); at != -1 {
		return "REDACTED@" + body[at+1:]
	}
	return body
}
