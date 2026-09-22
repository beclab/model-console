package controlplane

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/handoff"
)

func post(t *testing.T, s *Server, path string) *httpResp {
	t.Helper()
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)
	resp, err := http.Post(srv.URL+path, contentTypeJSON, http.NoBody)
	if err != nil {
		t.Fatalf("POST %s: %v", path, err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	return &httpResp{Status: resp.StatusCode, Header: resp.Header.Clone(), Body: body}
}

func TestEngineRestart_POSTReceiptSeparatesSignalFromRestart(t *testing.T) {
	t.Parallel()
	s := fixtureWithSpec(t, "")

	r := post(t, s, "/api/engine/restart")
	if r.Status != 200 {
		t.Fatalf("status = %d; body=%s", r.Status, r.Body)
	}
	var got struct {
		OK      bool            `json:"ok"`
		RunDir  string          `json:"run_dir"`
		Restart handoff.Receipt `json:"restart"`
	}
	if err := json.Unmarshal(r.Body, &got); err != nil {
		t.Fatalf("decode: %v; body=%s", err, r.Body)
	}
	if !got.Restart.Signaled || got.Restart.Generation == "" {
		t.Errorf("receipt = %+v, want a signaled generation", got.Restart)
	}
	// 200 says the signal is on disk. Nothing here has watched an engine,
	// so the receipt must not imply one relaunched.
	if got.Restart.Supervision != handoff.SupervisionUnknown {
		t.Errorf("Supervision = %q, want unknown", got.Restart.Supervision)
	}
	if h := r.Header.Get(headerRestarted); h != "" {
		t.Errorf("%s = %q, want absent", headerRestarted, h)
	}
}

func TestEngineRestart_GETReportsTheOutcome(t *testing.T) {
	t.Parallel()
	s := fixtureWithSpec(t, "")

	before := get(t, s, "/api/engine/restart")
	var idle handoff.Status
	if err := json.Unmarshal(before.Body, &idle); err != nil {
		t.Fatalf("decode: %v; body=%s", err, before.Body)
	}
	if idle.State != handoff.StateIdle {
		t.Errorf("State = %q before any request, want idle", idle.State)
	}

	if r := post(t, s, "/api/engine/restart"); r.Status != 200 {
		t.Fatalf("POST status = %d; body=%s", r.Status, r.Body)
	}

	after := get(t, s, "/api/engine/restart")
	var signaled handoff.Status
	if err := json.Unmarshal(after.Body, &signaled); err != nil {
		t.Fatalf("decode: %v; body=%s", err, after.Body)
	}
	// No probe is wired in this fixture, so there is nothing to observe
	// the dip — which the status says outright instead of implying success.
	if signaled.State != handoff.StateUnverified {
		t.Errorf("State = %q, want unverified", signaled.State)
	}
	if signaled.Generation == "" || signaled.RequestedAt == nil {
		t.Errorf("status = %+v, want the request recorded", signaled)
	}
}

func TestModelSpec_PutLegacyHeaderNeedsPriorEvidence(t *testing.T) {
	t.Parallel()
	// A run dir that already carries evidence of a supervisor acting on
	// an earlier signal — what a redeployed llm-init finds beside an
	// engine whose wrapper has been relaunching all along.
	s := fixtureWithSpecSeeded(t, "", func(cfg *config.Config) {
		evidence := `{"supervised":true,"last_confirmed_generation":"1"}`
		path := filepath.Join(cfg.Runtime.RunDir, handoff.StateName)
		if err := os.WriteFile(path, []byte(evidence), 0o644); err != nil {
			t.Fatal(err)
		}
	})

	body := `{"name":"qwen2.5-7b","mode":"chat","supports":{},"engine_args":"--max-model-len 8192"}`
	r := putSpec(t, s, body)
	if r.Status != 200 {
		t.Fatalf("PUT status = %d; body=%s", r.Status, r.Body)
	}
	if got := r.Header.Get(headerRestartSupervision); got != string(handoff.SupervisionConfirmed) {
		t.Errorf("%s = %q, want confirmed", headerRestartSupervision, got)
	}
	if got := r.Header.Get(headerRestarted); got != "true" {
		t.Errorf("%s = %q; a supervisor observed before is expected to act again",
			headerRestarted, got)
	}
}

func TestModelSpec_PutUnchangedEngineArgsSignalsNothing(t *testing.T) {
	t.Parallel()
	s := fixtureWithSpec(t, "")

	body := `{"name":"qwen2.5-7b","mode":"chat","supports":{}}`
	r := putSpec(t, s, body)
	if r.Status != 200 {
		t.Fatalf("PUT status = %d; body=%s", r.Status, r.Body)
	}
	if got := r.Header.Get(headerRestart); got != restartNotNeeded {
		t.Errorf("%s = %q, want %q", headerRestart, got, restartNotNeeded)
	}
	if _, err := os.Stat(filepath.Join(s.config().Runtime.RunDir, handoff.RestartName)); !os.IsNotExist(err) {
		t.Errorf("nothing the engine launches with changed, so no signal is due (stat err=%v)", err)
	}
}
