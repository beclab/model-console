package proxy

import (
	"context"
	"fmt"
	"net/http"

	dto "github.com/prometheus/client_model/go"

	"github.com/llm-init/llm-init/internal/enginemetrics"
)

// metrics.go is this package's transport half of the engines'
// Prometheus /metrics endpoint; the parsing and the gauge readers live
// in internal/enginemetrics, shared with the control plane.

// fetchMetricFamilies GETs /metrics and parses it. Returns an error on
// transport failure or non-200 so callers that treat /metrics as their
// primary signal (vLLM) can surface a 503.
func (a *Adapter) fetchMetricFamilies(ctx context.Context) (map[string]*dto.MetricFamily, error) {
	body, status, err := a.getJSON(ctx, "/metrics")
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("GET /metrics status %d", status)
	}
	return enginemetrics.Parse(body)
}
