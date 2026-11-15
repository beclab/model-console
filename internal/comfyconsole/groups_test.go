package comfyconsole

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// groupFixture serves /f/<name> from payloads; /f/slow writes half of its
// payload and then hangs until the client goes away.
type groupFixture struct {
	srv      *httptest.Server
	payloads map[string][]byte
}

func newGroupFixture(t *testing.T) *groupFixture {
	t.Helper()
	g := &groupFixture{payloads: map[string][]byte{}}
	for _, n := range []string{"shared", "a", "b", "both", "slow"} {
		g.payloads[n] = bytes.Repeat([]byte(n+"-weights-"), 2048)
	}
	g.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/system_stats" {
			_, _ = w.Write([]byte(`{}`))
			return
		}
		name := strings.TrimPrefix(r.URL.Path, "/f/")
		p, ok := g.payloads[name]
		if !ok {
			http.NotFound(w, r)
			return
		}
		if name == "slow" {
			w.Header().Set("Content-Length", strconv.Itoa(len(p)))
			if r.Method == http.MethodHead {
				return
			}
			_, _ = w.Write(p[:len(p)/2])
			w.(http.Flusher).Flush()
			<-r.Context().Done()
			return
		}
		http.ServeContent(w, r, name, time.Unix(0, 0), bytes.NewReader(p))
	}))
	t.Cleanup(g.srv.Close)
	return g
}

func (g *groupFixture) model(name string) Model {
	p := g.payloads[name]
	return Model{Name: name + ".safetensors", Dir: "diffusion_models", URL: g.srv.URL + "/f/" + name, SHA256: sum(p), Size: int64(len(p))}
}

func (g *groupFixture) config() *AppConfig {
	return &AppConfig{
		Models: []Model{g.model("shared")},
		Groups: []Group{
			{ID: "a", Label: "A", Models: []Model{g.model("a"), g.model("both")}},
			{ID: "b", Label: "B", Models: []Model{g.model("b"), g.model("both")}},
			{ID: "slow", Label: "Slow", Models: []Model{g.model("slow")}},
		},
	}
}

func groupState(st Status, id string) GroupStatus {
	for _, g := range st.Groups {
		if g.ID == id {
			return g
		}
	}
	return GroupStatus{}
}

func exists(dir, name string) bool {
	_, err := os.Stat(filepath.Join(dir, "diffusion_models", name+".safetensors"))
	return err == nil
}

func TestLoadSelection(t *testing.T) {
	t.Parallel()
	cfg := newGroupFixture(t).config()
	dir := t.TempDir()
	for initial, want := range map[string]string{"": "a", "all": "a,b,slow", "b, a": "b,a"} {
		got, err := LoadSelection(cfg, "", initial)
		if err != nil || strings.Join(got, ",") != want {
			t.Errorf("LoadSelection(%q) = %v, %v; want %s", initial, got, err, want)
		}
	}
	if _, err := LoadSelection(cfg, "", "fp8"); err == nil {
		t.Error("unknown COMFY_GROUPS accepted")
	}

	file := filepath.Join(dir, "sel", "groups.json")
	if got, _ := LoadSelection(cfg, file, "b"); strings.Join(got, ",") != "b" {
		t.Fatalf("first start = %v", got)
	}
	if got, _ := LoadSelection(cfg, file, "a"); strings.Join(got, ",") != "b" {
		t.Fatalf("the selection file must win over COMFY_GROUPS, got %v", got)
	}
	_ = os.WriteFile(file, []byte(`{"groups":["gone"]}`), 0o644)
	if got, _ := LoadSelection(cfg, file, "a"); strings.Join(got, ",") != "a" {
		t.Fatalf("stale selection should fall back to COMFY_GROUPS, got %v", got)
	}
	if got, _ := LoadSelection(&AppConfig{Models: cfg.Models}, file, "a"); got != nil {
		t.Fatalf("no groups: %v", got)
	}
}

func TestGroupsValidation(t *testing.T) {
	t.Parallel()
	g := newGroupFixture(t)
	bad := g.config()
	bad.Groups[0].Models = append(bad.Groups[0].Models, g.model("shared"))
	if err := bad.validate(); err == nil || !strings.Contains(err.Error(), "also in models") {
		t.Errorf("shared file in a group: %v", err)
	}
	bad = g.config()
	bad.Groups[1].Models[1].SHA256 = sum([]byte("other"))
	if err := bad.validate(); err == nil || !strings.Contains(err.Error(), "differs") {
		t.Errorf("same path, other file: %v", err)
	}
	bad = g.config()
	bad.Groups[1].ID = "a"
	if err := bad.validate(); err == nil {
		t.Error("duplicate group id accepted")
	}
	ok := g.config()
	ok.Models = nil
	if err := ok.validate(); err != nil {
		t.Errorf("groups without shared models: %v", err)
	}
}

func TestGroups_AddAndRemove(t *testing.T) {
	t.Parallel()
	g := newGroupFixture(t)
	dir := t.TempDir()
	selFile := filepath.Join(dir, "groups.json")
	cfg := g.config()
	sel, err := LoadSelection(cfg, selFile, "a")
	if err != nil {
		t.Fatal(err)
	}
	c := New(Options{Config: cfg, ModelsDir: dir, ComfyURL: g.srv.URL, Selected: sel, SelectionFile: selFile,
		ProbeInterval: 10 * time.Millisecond, RecheckEvery: time.Hour, RetryBackoff: 20 * time.Millisecond, RetryBackoffMx: 20 * time.Millisecond})
	run(t, c)
	h := c.Handler()
	call := func(method, id string) int {
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, httptest.NewRequest(method, "/api/groups/"+id, nil))
		return rr.Code
	}

	st := waitFor(t, c, "ready with group a", func(s Status) bool { return s.Ready })
	if a, b := groupState(st, "a"), groupState(st, "b"); a.State != StateDone || !a.Selected || b.State != StateAbsent || b.Selected {
		t.Fatalf("groups a=%+v b=%+v", a, b)
	}
	if len(st.Models.Items) != 3 || !exists(dir, "shared") || !exists(dir, "both") || exists(dir, "b") {
		t.Fatalf("only shared and group a should be on disk; items %+v", st.Models.Items)
	}

	if code := call(http.MethodPut, "b"); code != http.StatusOK {
		t.Fatalf("PUT b = %d", code)
	}
	waitFor(t, c, "group b installed", func(s Status) bool { return groupState(s, "b").State == StateDone })
	if code := call(http.MethodDelete, "a"); code != http.StatusOK {
		t.Fatalf("DELETE a = %d", code)
	}
	if exists(dir, "a") || !exists(dir, "both") || !exists(dir, "shared") {
		t.Fatal("deleting a must remove a.safetensors and keep files group b or every group uses")
	}
	st = waitFor(t, c, "a absent, still ready", func(s Status) bool { return groupState(s, "a").State == StateAbsent && s.Ready })
	if len(st.Models.Items) != 3 {
		t.Fatalf("items after delete: %+v", st.Models.Items)
	}
	if code := call(http.MethodDelete, "b"); code != http.StatusConflict {
		t.Fatalf("deleting the last group = %d, want 409", code)
	}
	if code := call(http.MethodPut, "fp8"); code != http.StatusNotFound {
		t.Fatalf("PUT unknown = %d, want 404", code)
	}
	var saved selectionFile
	raw, _ := os.ReadFile(selFile)
	if json.Unmarshal(raw, &saved) != nil || strings.Join(saved.Groups, ",") != "b" {
		t.Fatalf("selection file = %s", raw)
	}
}

func TestGroups_RemoveStopsDownload(t *testing.T) {
	t.Parallel()
	g := newGroupFixture(t)
	dir := t.TempDir()
	c := New(Options{Config: g.config(), ModelsDir: dir, ComfyURL: g.srv.URL, Selected: []string{"a"},
		ProbeInterval: 10 * time.Millisecond, RecheckEvery: time.Hour, RetryBackoff: 20 * time.Millisecond, RetryBackoffMx: 20 * time.Millisecond})
	run(t, c)
	waitFor(t, c, "group a", func(s Status) bool { return groupState(s, "a").State == StateDone })
	if err := c.AddGroup("slow"); err != nil {
		t.Fatal(err)
	}
	waitFor(t, c, "slow downloading", func(s Status) bool {
		gs := groupState(s, "slow")
		return gs.State == StateDownloading && gs.Downloaded > 0
	})
	done := make(chan error, 1)
	go func() { done <- c.RemoveGroup("slow") }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("RemoveGroup did not stop the download")
	}
	matches, _ := filepath.Glob(filepath.Join(dir, "diffusion_models", "slow.safetensors*"))
	if len(matches) != 0 {
		t.Fatalf("left behind: %v", matches)
	}
	if st := c.Status(); groupState(st, "slow").State != StateAbsent || !st.Ready {
		t.Fatalf("after removing slow: %+v ready=%v", groupState(st, "slow"), st.Ready)
	}
}
