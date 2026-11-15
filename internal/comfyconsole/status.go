package comfyconsole

import (
	"embed"
	"encoding/json"
	"errors"
	"io/fs"
	"net/http"
)

// Status is the body of GET /api/status.
type Status struct {
	Ready   bool          `json:"ready"`
	Models  ModelsStatus  `json:"models"`
	Groups  []GroupStatus `json:"groups,omitempty"`
	Backend BackendStatus `json:"backend"`
}

// GroupStatus is one model group. State is StateAbsent while unselected,
// else the aggregate of its own models.
type GroupStatus struct {
	ID         string  `json:"id"`
	Label      string  `json:"label"`
	Selected   bool    `json:"selected"`
	State      string  `json:"state"`
	Progress   float64 `json:"progress"`
	Downloaded int64   `json:"downloaded"`
	Size       int64   `json:"size"`
}

// StateAbsent is an unselected group.
const StateAbsent = "absent"

// ModelsStatus aggregates the models kept: shared and selected groups.
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
	sharedDone := true
	byGroup := map[string][]*itemState{}
	for _, it := range c.items {
		for _, g := range it.groups {
			byGroup[g] = append(byGroup[g], it)
		}
		if !c.wanted(it) {
			continue
		}
		if len(it.groups) == 0 && it.state != StateDone {
			sharedDone = false
		}
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
	groupDone := len(c.opts.Config.Groups) == 0
	for _, g := range c.opts.Config.Groups {
		gs := GroupStatus{ID: g.ID, Label: g.Label, Selected: c.selected[g.ID], State: StateAbsent}
		gc := map[string]int{}
		for _, it := range byGroup[g.ID] {
			gs.Size += it.model.Size
			if gs.Selected {
				gs.Downloaded += min(it.downloaded, it.model.Size)
				gc[it.state]++
			}
		}
		if gs.Selected {
			gs.State = aggregate(gc, len(byGroup[g.ID]))
			groupDone = groupDone || gs.State == StateDone
		}
		if gs.Size > 0 {
			gs.Progress = float64(gs.Downloaded) / float64(gs.Size)
		}
		st.Groups = append(st.Groups, gs)
	}
	c.mu.Unlock()

	st.Models.State = aggregate(counts, len(st.Models.Items))
	if total > 0 {
		st.Models.Progress = float64(have) / float64(total)
	}
	st.Backend.State = BackendDown
	if c.backend.Load() {
		st.Backend.State = BackendUp
	}
	// With groups, one installed group is enough: another may still be
	// downloading after the user added it.
	st.Ready = sharedDone && groupDone && st.Backend.State == BackendUp
	return st
}

func aggregate(counts map[string]int, n int) string {
	switch {
	case counts[StateDone] == n:
		return StateDone
	case counts[StateError] > 0:
		return StateError
	case counts[StateDownloading] > 0:
		return StateDownloading
	case counts[StateVerifying] > 0:
		return StateVerifying
	default:
		return StatePending
	}
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
	m.HandleFunc("PUT /api/groups/{id}", func(w http.ResponseWriter, r *http.Request) {
		c.groupChange(w, c.AddGroup(r.PathValue("id")))
	})
	m.HandleFunc("DELETE /api/groups/{id}", func(w http.ResponseWriter, r *http.Request) {
		c.groupChange(w, c.RemoveGroup(r.PathValue("id")))
	})
	sub, _ := fs.Sub(staticFS, "static")
	m.Handle("GET /", http.FileServer(http.FS(sub)))
	return m
}

func (c *Console) groupChange(w http.ResponseWriter, err error) {
	if err == nil {
		writeJSON(w, http.StatusOK, c.Status())
		return
	}
	status := http.StatusInternalServerError
	switch {
	case errors.Is(err, ErrUnknownGroup), errors.Is(err, ErrNoGroups):
		status = http.StatusNotFound
	case errors.Is(err, ErrLastGroup):
		status = http.StatusConflict
	}
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
