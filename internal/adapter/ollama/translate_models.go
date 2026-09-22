package ollama

import (
	"encoding/json"
	"net/http"
)

// handleModels translates GET /v1/models. We only ever advertise the one
// model llm-init was configured with (cfg.Model.Name, the alias clients
// address), regardless of what /api/tags lists: this stack is
// single-model. Other models that may be
// present in the daemon are not made visible to OpenAI clients.
//
// We still call /api/tags to populate the "created" timestamp, because
// OpenAI clients use it for cache-busting; if the daemon errors, we fall
// back to a stub. The lookup matches against upstreamModel() (the
// daemon-side tag) — that is what /api/tags actually contains.
func (a *Adapter) handleModels(w http.ResponseWriter, r *http.Request) {
	created := int64(0)
	want := a.upstreamModel()
	tags, err := a.client.Tags(r.Context())
	if err == nil {
		for _, m := range tags.Models {
			if m.Name == want || matchesModelName(m.Name, want) {
				created = m.ModifiedAt.Unix()
				break
			}
		}
	}
	resp := OpenAIModelsResponse{
		Object: objectList,
		Data: []OpenAIModelDatum{
			{
				ID:      a.cfg.Model.Name,
				Object:  objectModel,
				Created: created,
				OwnedBy: ownerLLMInit,
			},
		},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}
