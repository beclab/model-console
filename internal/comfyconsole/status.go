package comfyconsole

import (
	"embed"
	"encoding/json"
	"io/fs"
	"net/http"
)

// Status is the body of GET /api/status.
type Status struct {
	Ready   bool          `json:"ready"`
	Models  ModelsStatus  `json:"models"`
	Backend BackendStatus `json:"backend"`
}

// ModelsStatus aggregates every model.
type ModelsStatus struct {
	State    string       `json:"state"`
	Progress float64      `json:"progress"`
	Items    []ItemStatus `json:"items"`
}

// ItemStatus is one model.
type ItemStatus struct {
	Name       string `json:"name"`
	Dir        string `json:"dir"`
	State      string `json:"state"`
	Downloaded int64  `json:"downloaded"`
	Size       int64  `json:"size"`
	Error      string `json:"error,omitempty"`
}

// BackendStatus is the ComfyUI probe result.
type BackendStatus struct {
	State string `json:"state"`
}

// Status snapshots the console.
func (c *Console) Status() Status {
	c.mu.Lock()
	var st Status
	var have, total int64
	counts := map[string]int{}
	for _, it := range c.items {
		st.Models.Items = append(st.Models.Items, ItemStatus{
			Name:       it.model.Name,
			Dir:        it.model.Dir,
			State:      it.state,
			Downloaded: min(it.downloaded, it.model.Size),
			Size:       it.model.Size,
			Error:      it.err,
		})
		counts[it.state]++
		have += min(it.downloaded, it.model.Size)
		total += it.model.Size
	}
	c.mu.Unlock()

	switch {
	case counts[StateDone] == len(st.Models.Items):
		st.Models.State = StateDone
	case counts[StateError] > 0:
		st.Models.State = StateError
	case counts[StateDownloading] > 0:
		st.Models.State = StateDownloading
	case counts[StateVerifying] > 0:
		st.Models.State = StateVerifying
	default:
		st.Models.State = StatePending
	}
	if total > 0 {
		st.Models.Progress = float64(have) / float64(total)
	} else if st.Models.State == StateDone {
		st.Models.Progress = 1
	}
	st.Backend.State = BackendDown
	if c.backend.Load() {
		st.Backend.State = BackendUp
	}
	st.Ready = st.Models.State == StateDone && st.Backend.State == BackendUp
	return st
}

//go:embed static
var staticFS embed.FS

// Handler serves the console. Paths are relative so the app's nginx can
// mount it under any prefix (the design uses /console/).
func (c *Console) Handler() http.Handler {
	m := http.NewServeMux()
	m.HandleFunc("GET /api/status", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, c.Status())
	})
	m.HandleFunc("GET /livez", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	m.HandleFunc("GET /readyz", func(w http.ResponseWriter, _ *http.Request) {
		st := c.Status()
		code := http.StatusOK
		if !st.Ready {
			code = http.StatusServiceUnavailable
		}
		writeJSON(w, code, map[string]bool{"ready": st.Ready})
	})
	sub, _ := fs.Sub(staticFS, "static")
	m.Handle("GET /", http.FileServer(http.FS(sub)))
	return m
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
