// Package enginemetrics reads the Prometheus text exposition the
// inference engines serve on /metrics.
//
// vLLM exposes it by default; SGLang and llama.cpp require
// --enable-metrics / --metrics respectively, which the deploy wrappers
// bake in. Only a handful of gauges are ever pulled, so the cost is one
// small GET parsed with expfmt.
//
// The parsing lives here rather than beside either caller because there
// are two of them now — the data-plane adapter's diag path and the
// control plane's engine-load report — and a second copy of "which
// scheme does the parser need" is the kind of detail that is only
// discovered wrong at runtime, on the engine whose metric names carry
// colons.
package enginemetrics

import (
	"bytes"
	"fmt"
	"strconv"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
)

// Parse turns a /metrics body into a name->family map.
func Parse(body []byte) (map[string]*dto.MetricFamily, error) {
	// UTF8Validation accepts the colon-bearing engine metric names
	// (vllm:..., sglang:..., llamacpp:...); the parser's default
	// scheme is unset and panics on validation.
	parser := expfmt.NewTextParser(model.UTF8Validation)
	fams, err := parser.TextToMetricFamilies(bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("parse /metrics: %w", err)
	}
	return fams, nil
}

// Gauge returns the first sample value of a gauge family, or nil when
// the family is absent / has no samples.
func Gauge(fams map[string]*dto.MetricFamily, name string) *float64 {
	mf := fams[name]
	if mf == nil || len(mf.Metric) == 0 {
		return nil
	}
	m := mf.Metric[0]
	if m.Gauge == nil || m.Gauge.Value == nil {
		return nil
	}
	v := m.GetGauge().GetValue()
	return &v
}

// Label reads a label value off the first sample of an info-style gauge
// family (e.g. vllm:cache_config_info{gpu_memory_utilization=...}).
// Returns "" when the family or label is absent.
func Label(fams map[string]*dto.MetricFamily, name, label string) string {
	mf := fams[name]
	if mf == nil || len(mf.Metric) == 0 {
		return ""
	}
	for _, lp := range mf.Metric[0].Label {
		if lp.GetName() == label {
			return lp.GetValue()
		}
	}
	return ""
}

// FloatLabel parses an info label as a float; nil when the label is
// empty or unparseable.
func FloatLabel(fams map[string]*dto.MetricFamily, name, label string) *float64 {
	s := Label(fams, name, label)
	if s == "" {
		return nil
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return nil
	}
	return &v
}

// IntLabel parses an info label as an int; nil when the label is empty
// or unparseable. Used for cpu_offload_gb, which vLLM reports as an
// integer-valued string that may render as "0" or "0.0" depending on
// version.
func IntLabel(fams map[string]*dto.MetricFamily, name, label string) *int {
	s := Label(fams, name, label)
	if s == "" {
		return nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return nil
	}
	n := int(f)
	return &n
}
