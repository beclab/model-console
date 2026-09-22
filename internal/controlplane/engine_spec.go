package controlplane

import (
	"context"
	"encoding/json"
	"log/slog"
	"maps"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
)

// pathEngineSpec is the optional engine self-report of its own capability endpoints; engines without one 404 and this relay stays silent.
const pathEngineSpec = "/api/engine-spec"

// engineSpecTTL keeps dashboard polling off the engine's back — the report only changes when the engine restarts.
const engineSpecTTL = 5 * time.Second

const engineSpecTimeout = 2 * time.Second

const groupEngine = "Engine-reported"

type engineSpec struct {
	SchemaVersion *int               `json:"schema_version"`
	Model         string             `json:"model"`
	Base          string             `json:"base"`
	Implements    *[]string          `json:"implements"`
	Declares      *[]string          `json:"declares"`
	Serves        *[]string          `json:"serves"`
	Endpoints     *[]json.RawMessage `json:"endpoints"`
	Extensions    map[string]any     `json:"extensions"`
}

type engineSpecRow struct {
	Capability     string `json:"capability"`
	Method         string `json:"method"`
	Path           string `json:"path"`
	Description    string `json:"description"`
	Available      *bool  `json:"available"`
	Reason         string `json:"reason"`
	AsyncSupported *bool  `json:"async_supported"`
	// Deprecated marks a path the engine still answers but has superseded — the async task
	// aliases (/v1/audio/tasks/*) are the current case. Without relaying it, an alias reads
	// exactly like a current route on the dashboard.
	Deprecated       bool                 `json:"deprecated"`
	OperationID      string               `json:"operation_id"`
	Protocol         string               `json:"protocol"`
	Transport        string               `json:"transport"`
	SyncSupported    *bool                `json:"sync_supported"`
	Streaming        bool                 `json:"streaming"`
	RequiredSupports []string             `json:"required_supports"`
	InputModalities  []string             `json:"input_modalities"`
	OutputModalities []string             `json:"output_modalities"`
	OutputFormats    []string             `json:"output_formats"`
	SampleRates      []int                `json:"sample_rates"`
	Parameters       []OperationParameter `json:"parameters"`
	Limits           map[string]any       `json:"limits"`
	ResourceScope    string               `json:"resource_scope"`
	Extensions       map[string]any       `json:"extensions"`
}

// engineSpecCache is the per-Server memo of the last successful report.
type engineSpecCache struct {
	mu      sync.Mutex
	report  engineSpecReport
	fetched time.Time

	// lastLogged is the signature of the outcome the log already carries.
	// The catalog is polled by a dashboard, so an engine that stays
	// unreachable would otherwise emit a line every engineSpecTTL forever;
	// logging transitions says the same thing once.
	lastLogged string
}

// Outcomes of one read of the engine's own spec. The relay answers callers
// with an empty report for every one of the failures, which is what keeps a
// dashboard working while an engine restarts — and is also why an operator
// could not previously tell an engine that declares nothing from one that
// could not be asked.
const (
	specFetchOK          = "ok"
	specFetchBadURL      = "bad_engine_url"
	specFetchUnreachable = "unreachable"
	specFetchHTTPStatus  = "http_status"
	specFetchDecodeError = "decode_error"
	specFetchInvalidSpec = "invalid_spec"
	specFetchNoEndpoints = "no_endpoints"
)

// engineSpecOutcome is why a fetch produced the report it did, carried
// separately so the report itself stays the plain data the catalog merges.
type engineSpecOutcome struct {
	result string
	// detail is the human half of result: a status code, a decode error.
	detail string
	// dropped counts refused rows by the field that failed. A capability
	// that lands here vanishes from the catalog, and the gateway downstream
	// then refuses the route as unsupported.
	dropped map[string]int
	rows    int
}

// signature is what the transition-logging in engineEndpoints compares.
func (o engineSpecOutcome) signature() string {
	parts := make([]string, 0, len(o.dropped)+2)
	parts = append(parts, o.result, o.detail)
	for _, reason := range slices.Sorted(maps.Keys(o.dropped)) {
		parts = append(parts, reason+"="+strconv.Itoa(o.dropped[reason]))
	}
	return strings.Join(parts, "|")
}

func (o *engineSpecOutcome) drop(reason string) {
	if o.dropped == nil {
		o.dropped = map[string]int{}
	}
	o.dropped[reason]++
}

// engineSpecState is what the relay knows about the sibling engine's own
// route list, as opposed to what that list says.
//
// The two failures used to be one: a fetch that never happened and an engine
// that has nothing to declare both produced an empty report, so the catalog
// fell back to the static rows for the mode. For an audio application that
// fallback advertises chat and embeddings — routes it will never serve —
// until the first spec arrives and dropUndeclaredProxyRows removes them.
// A dashboard rendered during that window, or a Router model sync landing in
// it, reads a capability set nothing will honour.
type engineSpecState int

const (
	// engineSpecUnknown: there is an engine to ask and no answer from it
	// yet. The zero value, so a report nobody filled in says so.
	engineSpecUnknown engineSpecState = iota
	// engineSpecSilent: asked and answered, with no versioned spec. Engines
	// that predate the contract 404 here, and llm-init describes their data
	// plane from its own static catalog — the behaviour this whole relay is
	// an optional upgrade to.
	engineSpecSilent
	// engineSpecDeclared: a valid versioned spec. Authoritative over the
	// static rows, including the power to remove them.
	engineSpecDeclared
)

type engineSpecReport struct {
	rows          []EndpointInfo
	state         engineSpecState
	authoritative bool
	modelMatches  bool
	extensions    map[string]any
	diagnostics   []string
}

// engineEndpoints returns an empty report if the engine is down or serves no valid spec; the catalog must not fail on the engine.
func (s *Server) engineEndpoints() engineSpecReport {
	cfg := s.config()
	if cfg.Engine.URL == "" {
		// No sibling to ask, so nothing is pending: this deployment's data
		// plane is whatever llm-init's own catalog says it is.
		return engineSpecReport{state: engineSpecSilent}
	}
	// Liveness, not readiness: this only asks whether it is worth
	// spending a round trip asking the engine for its own route list.
	// An engine that is up without its model still answers that.
	if !s.readiness().EngineAlive {
		return s.cachedEngineSpec()
	}
	s.engineSpec.mu.Lock()
	defer s.engineSpec.mu.Unlock()
	now := s.opts.NowFunc()
	if !s.engineSpec.fetched.IsZero() && now.Sub(s.engineSpec.fetched) < engineSpecTTL {
		return s.engineSpec.report
	}
	report, outcome := fetchEngineSpec(cfg.Engine.URL, cfg.Model.Name)
	// A fetch that could not reach a verdict does not overwrite one that
	// did. Otherwise a single reset connection re-opens the routes an
	// authoritative spec had removed, for as long as it takes the next
	// poll to land -- and it is the second read, not the first, that a
	// dashboard or a model sync is likely to catch. The failure itself is
	// not hidden: recordEngineSpecOutcome counts and logs it either way.
	if report.state != engineSpecUnknown || s.engineSpec.report.state == engineSpecUnknown {
		s.engineSpec.report = report
	}
	s.engineSpec.fetched = now
	s.recordEngineSpecOutcome(cfg.Engine.URL, outcome)
	return s.engineSpec.report
}

func (s *Server) cachedEngineSpec() engineSpecReport {
	s.engineSpec.mu.Lock()
	defer s.engineSpec.mu.Unlock()
	return s.engineSpec.report
}

// recordEngineSpecOutcome counts every fetch and logs the ones that changed
// something. Callers hold engineSpec.mu.
func (s *Server) recordEngineSpecOutcome(engineURL string, o engineSpecOutcome) {
	if m := s.opts.Metrics; m != nil {
		m.EngineSpecFetchTotal.WithLabelValues(o.result).Inc()
		for reason, n := range o.dropped {
			m.EngineSpecRowsDroppedTotal.WithLabelValues(reason).Add(float64(n))
		}
	}
	signature := o.signature()
	if signature == s.engineSpec.lastLogged {
		return
	}
	s.engineSpec.lastLogged = signature
	switch {
	case o.result != specFetchOK:
		slog.Warn("engine spec unavailable; serving the catalog without engine-declared capabilities",
			"url", strings.TrimRight(engineURL, "/")+pathEngineSpec,
			"result", o.result, "detail", o.detail)
	case len(o.dropped) > 0:
		slog.Warn("engine spec rows refused; the capabilities they declare will not reach the gateway",
			"url", strings.TrimRight(engineURL, "/")+pathEngineSpec,
			"accepted", o.rows, "dropped", formatDropped(o.dropped))
	default:
		slog.Info("engine spec accepted", "endpoints", o.rows)
	}
}

func formatDropped(dropped map[string]int) string {
	parts := make([]string, 0, len(dropped))
	for _, reason := range slices.Sorted(maps.Keys(dropped)) {
		parts = append(parts, reason+"="+strconv.Itoa(dropped[reason]))
	}
	return strings.Join(parts, ",")
}

func fetchEngineSpec(engineURL, configuredModel string) (engineSpecReport, engineSpecOutcome) {
	ctx, cancel := context.WithTimeout(context.Background(), engineSpecTimeout)
	defer cancel()
	url := strings.TrimRight(engineURL, "/") + pathEngineSpec
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return engineSpecReport{state: engineSpecUnknown},
			engineSpecOutcome{result: specFetchBadURL, detail: err.Error()}
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return engineSpecReport{state: engineSpecUnknown},
			engineSpecOutcome{result: specFetchUnreachable, detail: err.Error()}
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		// "This route does not exist here" is an answer, and the one every
		// engine written before this contract gives forever; llm-init then
		// describes their data plane from its own catalog, as it always
		// has. A 5xx or a 429 is the same engine failing to answer, which
		// settles nothing.
		state := engineSpecUnknown
		switch resp.StatusCode {
		case http.StatusNotFound, http.StatusMethodNotAllowed, http.StatusNotImplemented:
			state = engineSpecSilent
		}
		return engineSpecReport{state: state}, engineSpecOutcome{
			result: specFetchHTTPStatus, detail: strconv.Itoa(resp.StatusCode),
		}
	}
	var spec engineSpec
	if err := json.NewDecoder(resp.Body).Decode(&spec); err != nil {
		// An engine that serves this path meant to declare something, so a
		// body that will not parse is a broken report rather than an absent
		// one, and the static rows are not a safe stand-in for it.
		return engineSpecReport{state: engineSpecUnknown},
			engineSpecOutcome{result: specFetchDecodeError, detail: err.Error()}
	}
	authoritative := spec.SchemaVersion != nil && (*spec.SchemaVersion == 1 || *spec.SchemaVersion == 2)
	if authoritative && !validEngineSpecVersioned(spec) {
		return engineSpecReport{state: engineSpecUnknown}, engineSpecOutcome{
			result: specFetchInvalidSpec, detail: invalidEngineSpecDetail(spec),
		}
	}
	if spec.Endpoints == nil {
		return engineSpecReport{state: engineSpecSilent},
			engineSpecOutcome{result: specFetchNoEndpoints}
	}
	outcome := engineSpecOutcome{result: specFetchOK}
	diagnostics := engineSpecDiagnostics(spec.Model, configuredModel)
	out := make([]EndpointInfo, 0, len(*spec.Endpoints))
	for _, raw := range *spec.Endpoints {
		var r engineSpecRow
		if err := json.Unmarshal(raw, &r); err != nil {
			outcome.drop(dropRowDecode)
			continue
		}
		if spec.SchemaVersion != nil && *spec.SchemaVersion == 2 {
			r.Extensions = mergeUnknownEngineFields(raw, r.Extensions)
		}
		if reason := engineSpecRowDropReason(r, spec.SchemaVersion); reason != "" {
			outcome.drop(reason)
			continue
		}
		out = append(out, EndpointInfo{
			Method:         r.Method,
			Path:           r.Path,
			Category:       categoryOpenAI,
			Group:          groupEngine,
			Description:    engineRowDescription(r),
			Available:      *r.Available,
			Reasons:        engineRowReasons(r, diagnostics),
			AsyncSupported: r.AsyncSupported,
			OperationID:    r.OperationID, Protocol: r.Protocol, Transport: r.Transport,
			SyncSupported: r.SyncSupported, RequiredSupports: r.RequiredSupports,
			Streaming:       r.Streaming,
			InputModalities: r.InputModalities, OutputModalities: r.OutputModalities,
			OutputFormats: r.OutputFormats, SampleRates: r.SampleRates,
			Parameters: r.Parameters, Limits: r.Limits, ResourceScope: r.ResourceScope, Extensions: r.Extensions,
		})
	}
	outcome.rows = len(out)
	state := engineSpecSilent
	if authoritative && len(out) > 0 {
		state = engineSpecDeclared
	}
	return engineSpecReport{
		rows:          out,
		state:         state,
		authoritative: authoritative && len(out) > 0,
		modelMatches:  spec.Model == configuredModel,
		extensions:    engineModelExtensions(spec.Extensions),
		diagnostics:   diagnostics,
	}, outcome
}

// invalidEngineSpecDetail names the envelope fields a versioned spec left
// out. Without it the log says only that the whole spec was refused, and
// the engine author has to diff against the contract to find out why.
func invalidEngineSpecDetail(spec engineSpec) string {
	var missing []string
	if strings.TrimSpace(spec.Model) == "" {
		missing = append(missing, "model")
	}
	if spec.Implements == nil {
		missing = append(missing, "implements")
	}
	if spec.Declares == nil {
		missing = append(missing, "declares")
	}
	if spec.Serves == nil {
		missing = append(missing, "serves")
	}
	if spec.Endpoints == nil || len(*spec.Endpoints) == 0 {
		missing = append(missing, "endpoints")
	}
	return "missing or empty: " + strings.Join(missing, ", ")
}

func mergeUnknownEngineFields(raw json.RawMessage, extensions map[string]any) map[string]any {
	var fields map[string]json.RawMessage
	if json.Unmarshal(raw, &fields) != nil {
		return extensions
	}
	known := map[string]bool{
		"capability": true, "method": true, "path": true, "description": true,
		"available": true, jsonKeyReason: true, "async_supported": true, "deprecated": true,
		"operation_id": true, "protocol": true, "transport": true, "sync_supported": true, "streaming": true,
		"required_supports": true, "input_modalities": true, "output_modalities": true,
		"output_formats": true, "sample_rates": true, "parameters": true, "limits": true,
		"resource_scope": true, "extensions": true,
	}
	out := map[string]any{}
	for key, value := range extensions {
		out[key] = value
	}
	for key, value := range fields {
		if known[key] {
			continue
		}
		var decoded any
		if json.Unmarshal(value, &decoded) == nil {
			out[key] = decoded
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// engineModelExtensions keeps the engine-owned capability namespaces while
// reserving capacity for Model Console's measured runtime reporter.
func engineModelExtensions(source map[string]any) map[string]any {
	if len(source) == 0 {
		return nil
	}
	out := make(map[string]any, len(source))
	for key, value := range source {
		if key != "capacity" {
			out[key] = value
		}
	}
	return out
}

// Why a declared endpoint was refused. These are metric label values and
// appear in the operator log, so they are named for the engine author who
// has to go fix the row, not for the branch that rejected it.
const (
	dropRowDecode            = "row_decode"
	dropMissingAvailable     = "missing_available"
	dropBadMethod            = "bad_method"
	dropBadPath              = "bad_path"
	dropReservedPath         = "reserved_path"
	dropMissingOperationID   = "missing_operation_id"
	dropMissingProtocol      = "missing_protocol"
	dropBadTransport         = "bad_transport"
	dropMissingSyncSupported = "missing_sync_supported"
)

// engineSpecRowDropReason names the field that disqualified a row, or "" if
// the row is good. A refused row leaves no trace in the catalog it was meant
// to join, so the reason is the only thing that can explain the absence.
func engineSpecRowDropReason(r engineSpecRow, version *int) string {
	switch {
	case r.Available == nil:
		return dropMissingAvailable
	case !engineSpecMethods[r.Method]:
		return dropBadMethod
	case r.Path == "" ||
		r.Path != strings.TrimSpace(r.Path) ||
		!strings.HasPrefix(r.Path, "/") ||
		strings.IndexFunc(r.Path, unicode.IsSpace) != -1 ||
		strings.ContainsAny(r.Path, "?#"):
		return dropBadPath
	case strings.HasPrefix(r.Path, "/api/"):
		return dropReservedPath
	}
	if version == nil || *version != 2 {
		return ""
	}
	switch {
	case r.OperationID == "":
		return dropMissingOperationID
	case r.Protocol == "":
		return dropMissingProtocol
	case r.Transport != "http" && r.Transport != "websocket":
		return dropBadTransport
	case r.SyncSupported == nil:
		return dropMissingSyncSupported
	}
	return ""
}

var engineSpecMethods = map[string]bool{
	mGET:     true,
	mPOST:    true,
	mDELETE:  true,
	mWS:      true,
	mPUT:     true,
	mPATCH:   true,
	mHEAD:    true,
	mOPTIONS: true,
}

func validEngineSpecVersioned(spec engineSpec) bool {
	return strings.TrimSpace(spec.Model) != "" &&
		spec.Implements != nil &&
		spec.Declares != nil &&
		spec.Serves != nil &&
		spec.Endpoints != nil &&
		len(*spec.Endpoints) > 0
}

// engineRowDescription names the row's capability, so a reader can tell which MODEL_SUPPORTS key turns it on, and flags a superseded path the same way llm-init's own rows are flagged.
func engineRowDescription(r engineSpecRow) string {
	d := r.Description
	if r.Capability != "" {
		d = r.Capability + ": " + d
	}
	if r.Deprecated {
		// Parenthetical rather than a sentence: an engine's description ends however it
		// likes, and this has to compose with all of them.
		return d + " (deprecated)"
	}
	return d
}

// newEngineRows drops engine rows llm-init documents itself, so a path the binary owns keeps its own description and curl hint.
func newEngineRows(own, reported []EndpointInfo) []EndpointInfo {
	if len(reported) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(own))
	for _, e := range own {
		seen[e.Method+" "+e.Path] = struct{}{}
	}
	out := make([]EndpointInfo, 0, len(reported))
	for _, e := range reported {
		if _, dup := seen[e.Method+" "+e.Path]; !dup {
			out = append(out, e)
		}
	}
	return out
}

func engineSpecDiagnostics(reportedModel, configuredModel string) []string {
	if reportedModel == "" || configuredModel == "" || reportedModel == configuredModel {
		return nil
	}
	return []string{
		`engine spec model "` + reportedModel + `" does not match configured MODEL_NAME "` + configuredModel + `"`,
	}
}

func engineRowReasons(r engineSpecRow, diagnostics []string) []string {
	var reasons []string
	if !*r.Available && r.Reason != "" {
		reasons = append(reasons, r.Reason)
	}
	reasons = append(reasons, diagnostics...)
	return reasons
}
