package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/config"
)

// ladderCard is a card declaring the one rule shape this check acts on: a
// string parameter with a closed list of values.
func ladderCard() config.ModelSpec {
	return config.ModelSpec{
		Name: "m",
		ParameterRules: []config.ParameterRule{{
			Name:    "reasoning_effort",
			Type:    config.ParamRuleTypeString,
			Options: []string{"low", "medium", "high"},
		}},
	}
}

func adapterWithCard(t *testing.T, spec config.ModelSpec, upstreamURL string) *Adapter {
	t.Helper()
	a, err := NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineLlamaCpp, URL: upstreamURL},
		Model:  config.Model{Name: "m"},
		Spec:   spec,
	}, config.EngineLlamaCpp)
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}
	return a
}

// countingUpstream answers 200 and says whether it was reached, which is the
// assertion that matters: a refused request must not have been forwarded.
func countingUpstream(t *testing.T, hits *int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		*hits++
		w.Header().Set(headerContentType, contentTypeJSON)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func postChat(t *testing.T, a *Adapter, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.ContentLength = int64(len(body))
	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)
	return rec
}

// The whole point of the check: llama.cpp's chat template raises inside Jinja
// for a level it does not know, so without this the caller is handed a 502
// with a Python traceback in it and no mention of reasoning_effort.
func TestAValueOutsideTheLadderIsRefusedBeforeTheEngineSeesIt(t *testing.T) {
	t.Parallel()
	hits := 0
	srv := countingUpstream(t, &hits)
	a := adapterWithCard(t, ladderCard(), srv.URL)

	rec := postChat(t, a, `{"model":"m","messages":[],"reasoning_effort":"minimal"}`)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body = %s", rec.Code, rec.Body.String())
	}
	if hits != 0 {
		t.Errorf("upstream was reached %d times; a refused request must not be forwarded", hits)
	}

	var envelope struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("body is not the Error envelope clients parse: %v (%s)", err, rec.Body.String())
	}
	if envelope.Error.Code != errCodeInvalidParameterValue {
		t.Errorf("code = %q, want %q", envelope.Error.Code, errCodeInvalidParameterValue)
	}
	// The accepted values have to be in the message: a data-plane client
	// may have no route to the control plane that serves the card.
	for _, level := range []string{"low", "medium", "high"} {
		if !strings.Contains(envelope.Error.Message, level) {
			t.Errorf("message %q does not name the accepted level %q", envelope.Error.Message, level)
		}
	}
	if !strings.Contains(envelope.Error.Message, "reasoning_effort") {
		t.Errorf("message %q does not name the parameter at fault", envelope.Error.Message)
	}
}

func TestADeclaredValueIsForwarded(t *testing.T) {
	t.Parallel()
	hits := 0
	srv := countingUpstream(t, &hits)
	a := adapterWithCard(t, ladderCard(), srv.URL)

	rec := postChat(t, a, `{"model":"m","messages":[],"reasoning_effort":"high"}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", rec.Code, rec.Body.String())
	}
	if hits != 1 {
		t.Errorf("upstream hits = %d, want 1", hits)
	}
}

// A card that declares no rule leaves the data plane exactly as it was.
func TestACardWithoutRulesChecksNothing(t *testing.T) {
	t.Parallel()
	hits := 0
	srv := countingUpstream(t, &hits)
	a := adapterWithCard(t, config.ModelSpec{Name: "m"}, srv.URL)

	rec := postChat(t, a, `{"model":"m","messages":[],"reasoning_effort":"minimal"}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", rec.Code, rec.Body.String())
	}
	if hits != 1 {
		t.Errorf("upstream hits = %d, want 1", hits)
	}
}

// Only the enum half is checkable. A numeric rule's range is the engine's to
// enforce on its own terms, and refusing here would invent a domain the card
// never claimed.
func TestOnlyEnumRulesAreChecked(t *testing.T) {
	t.Parallel()
	hits := 0
	srv := countingUpstream(t, &hits)
	max := 2.0
	a := adapterWithCard(t, config.ModelSpec{
		Name: "m",
		ParameterRules: []config.ParameterRule{
			{Name: "temperature", Type: "float", Max: &max},
			{Name: "quantization", Type: config.ParamRuleTypeString},
		},
	}, srv.URL)

	rec := postChat(t, a, `{"model":"m","messages":[],"temperature":9,"quantization":"anything"}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", rec.Code, rec.Body.String())
	}
	if hits != 1 {
		t.Errorf("upstream hits = %d, want 1", hits)
	}
}

// A rule of type string whose value arrives as a number is a differently
// malformed request, and the engine says so legibly. It is the unknown member
// of a known set that arrives as a template crash.
func TestANonStringValueIsLeftToTheEngine(t *testing.T) {
	t.Parallel()
	hits := 0
	srv := countingUpstream(t, &hits)
	a := adapterWithCard(t, ladderCard(), srv.URL)

	rec := postChat(t, a, `{"model":"m","messages":[],"reasoning_effort":3}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", rec.Code, rec.Body.String())
	}
	if hits != 1 {
		t.Errorf("upstream hits = %d, want 1", hits)
	}
}

// A request wrong about two parameters must name the same one every time: a
// client retrying against a moving answer learns nothing from the second.
func TestTwoBadValuesReportTheSameOneEveryTime(t *testing.T) {
	t.Parallel()
	rules := enumRules{
		"reasoning_effort": {"low", "high"},
		"audio_format":     {"wav"},
	}
	body := []byte(`{"reasoning_effort":"minimal","audio_format":"flac"}`)

	for i := 0; i < 20; i++ {
		rec := httptest.NewRecorder()
		if !checkEnumParams(rec, rules, "/v1/chat/completions", body) {
			t.Fatal("checkEnumParams reported nothing wrong")
		}
		if !strings.Contains(rec.Body.String(), "audio_format") {
			t.Fatalf("run %d named %s; the first name in sorted order is audio_format", i, rec.Body.String())
		}
	}
}

// A body this side cannot read is not a body it can find a bad parameter in.
// rewriteJSONModel reaches the same verdict and forwards it untouched.
func TestAnUnreadableBodyIsNotJudged(t *testing.T) {
	t.Parallel()
	rec := httptest.NewRecorder()
	if checkEnumParams(rec, enumRules{"x": {"a"}}, "/v1/chat/completions", []byte("not json")) {
		t.Error("an unparseable body was refused as a bad parameter value")
	}
}

func TestBuildEnumRulesTakesOnlyWhatItCanCheck(t *testing.T) {
	t.Parallel()
	got := buildEnumRules(config.ModelSpec{ParameterRules: []config.ParameterRule{
		{Name: "reasoning_effort", Type: config.ParamRuleTypeString, Options: []string{"low"}},
		{Name: "no_options", Type: config.ParamRuleTypeString},
		{Name: "numeric", Type: "int", Options: []string{"1"}},
		{Name: "", Type: config.ParamRuleTypeString, Options: []string{"x"}},
	}})

	if len(got) != 1 {
		t.Fatalf("rules = %v, want only reasoning_effort", got)
	}
	if len(got["reasoning_effort"]) != 1 {
		t.Errorf("options = %v", got["reasoning_effort"])
	}
}

func TestNoRulesMeansNoWork(t *testing.T) {
	t.Parallel()
	if got := buildEnumRules(config.ModelSpec{}); got != nil {
		t.Errorf("rules = %v, want nil so the hot path can skip the parse", got)
	}
}
