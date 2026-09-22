package proxy

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sort"
	"strings"

	"github.com/llm-init/llm-init/internal/config"
)

// errCodeInvalidParameterValue is returned when a request names a value the
// model card says the model does not have. Operator-visible, so changing it
// is a wire-shape change.
const errCodeInvalidParameterValue = "invalid_parameter_value"

// enumRules is the subset of a card's parameter_rules that can be checked
// here: the ones whose type is `string` and that list the values they accept.
//
// Nothing else is checkable without guessing. A numeric rule's min and max
// describe a range the engine will clamp or reject on its own terms, and a
// rule with no options is a description of a parameter rather than a
// statement about its domain — the card is written by the deployment, and
// inventing a domain for it here would refuse requests the engine accepts.
//
// The gain from the enum half is not symmetry with the rest, it is that an
// enum is the one case where the engine's own refusal is unreadable. A
// llama.cpp chat template raises inside Jinja for a level it does not know,
// so the caller gets a 500 with a Python-looking stack in it and no mention
// of which parameter was at fault — while the card sitting one hop away has
// the list of levels that would have worked.
type enumRules map[string][]string

// buildEnumRules reads the checkable rules off a card. Returns nil when the
// card declares none, which is the ordinary case and is what lets the check
// cost nothing on the hot path.
func buildEnumRules(spec config.ModelSpec) enumRules {
	var out enumRules
	for _, rule := range spec.ParameterRules {
		if rule.Type != config.ParamRuleTypeString || len(rule.Options) == 0 || rule.Name == "" {
			continue
		}
		if out == nil {
			out = make(enumRules, len(spec.ParameterRules))
		}
		out[rule.Name] = rule.Options
	}
	return out
}

// checkEnumParams reports whether body names a value outside what the card
// declares, having already written the 400 when it does.
//
// Only a JSON string is judged. A rule of type `string` whose value arrives
// as a number or an object is a differently malformed request, and the engine
// answers that one legibly — it is the unknown-member-of-a-known-set case
// that arrives as a template crash.
//
// The rules come from the card this process was assembled with, not from the
// store. An edit through PUT /api/model-spec is stored and served but not in
// force here, which is what the `parameter_rules` entry in the response's
// pending-app-restart list says.
func checkEnumParams(w http.ResponseWriter, rules enumRules, path string, body []byte) bool {
	if len(rules) == 0 || len(body) == 0 {
		return false
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		// Not a JSON object. rewriteJSONModel reaches the same verdict
		// and forwards it untouched; a body this side cannot read is
		// not a body it can find a bad parameter in.
		return false
	}
	// Sorted so a request that is wrong about two parameters names the same
	// one every time; a client retrying against a moving target learns
	// nothing from the second answer.
	names := make([]string, 0, len(fields))
	for name := range fields {
		if _, checked := rules[name]; checked {
			names = append(names, name)
		}
	}
	sort.Strings(names)

	for _, name := range names {
		var value string
		if err := json.Unmarshal(fields[name], &value); err != nil {
			continue
		}
		if allowed(rules[name], value) {
			continue
		}
		writeInvalidParameterValue(w, path, name, value, rules[name])
		return true
	}
	return false
}

func allowed(options []string, value string) bool {
	for _, option := range options {
		if option == value {
			return true
		}
	}
	return false
}

// writeInvalidParameterValue emits the 400 envelope. The message carries the
// accepted values because the caller has no other way to discover them from
// here — the card is served by the control plane, which a data-plane client
// may not be able to reach at all.
func writeInvalidParameterValue(w http.ResponseWriter, path, name, value string, options []string) {
	w.Header().Set(headerContentType, contentTypeJSON)
	w.WriteHeader(http.StatusBadRequest)
	_ = json.NewEncoder(w).Encode(map[string]any{
		jsonKeyError: map[string]string{
			jsonKeyCode: errCodeInvalidParameterValue,
			jsonKeyMessage: fmt.Sprintf(
				"%s=%q is not one of the values this model accepts (%s)",
				name, value, strings.Join(options, ", ")),
		},
	})
	slog.Warn("proxy: rejected a parameter value the model card does not list",
		slog.String("path", path),
		slog.String("param", name),
		slog.String("value", value),
		slog.String("options", strings.Join(options, ",")))
}
