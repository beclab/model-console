package config

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// knownFlagPresent is the value recorded in EngineArgs.Known for a bare
// boolean flag (one with no peeked value token).
const knownFlagPresent = "true"

// MarshalJSON ensures EngineArgs round-trips with stable field names
// matching the public API. Empty Known / Unknown serialise as
// {} / [] (not null) so the dashboard can render a row count without
// a JS null check.
func (a EngineArgs) MarshalJSON() ([]byte, error) {
	known := a.Known
	if known == nil {
		known = map[string]string{}
	}
	unknown := a.Unknown
	if unknown == nil {
		unknown = []string{}
	}
	return json.Marshal(struct {
		Raw     string            `json:"raw"`
		Known   map[string]string `json:"known"`
		Unknown []string          `json:"unknown"`
	}{
		Raw:     a.Raw,
		Known:   known,
		Unknown: unknown,
	})
}

// EngineArgs is the parsed view of the single ENGINE_ARGS env var.
//
// Raw is the byte-for-byte original; callers needing the operator's
// exact spelling (debug logs, audit echo) read Raw. Known is the
// per-engine recognised subset, normalised so the dashboard can render
// uniformly. Unknown carries every token that did not match the
// per-engine known table - llm-init does NOT error on unknowns; the
// engine container's wrapper script forwards them verbatim.
type EngineArgs struct {
	Raw     string
	Known   map[string]string
	Unknown []string
}

// ParseEngineArgs is the per-Kind dispatcher. raw == "" returns a zero
// EngineArgs with empty maps (legal). Unbalanced quotes in raw are
// surfaced as fail-fast config_invalid errors.
func ParseEngineArgs(kind EngineKind, raw string) (EngineArgs, error) {
	switch kind {
	case EngineOllama:
		return parseOllamaArgs(raw)
	case EngineVLLM:
		return parseCmdlineArgs(raw, vllmKnownFlags)
	case EngineSGLang:
		return parseCmdlineArgs(raw, sglangKnownFlags)
	case EngineLlamaCpp:
		return parseLlamacppArgs(raw)
	case EngineEmbed, EngineClipEmbed, EngineAudio, EngineOCR, EngineRerank, EngineMusic:
		// ENGINE_ARGS are set on the engine container (llamacpp for ocr), not llm-init.
		args := EngineArgs{Raw: raw, Known: map[string]string{}}
		if strings.TrimSpace(raw) == "" {
			return args, nil
		}
		return EngineArgs{}, fmt.Errorf("ParseEngineArgs: ENGINE_ARGS is not used with engine kind %q; configure the engine container directly instead", kind)
	default:
		return EngineArgs{}, fmt.Errorf("ParseEngineArgs: unsupported engine kind %q", kind)
	}
}

// GetString returns Known[key] and whether it was set. The empty-key
// case returns ("", false) so callers can use the bool to drive
// "fallback to engine default" logic without checking len.
func (a EngineArgs) GetString(key string) (string, bool) {
	if a.Known == nil {
		return "", false
	}
	v, ok := a.Known[key]
	return v, ok
}

// GetInt parses Known[key] as base-10 int. Missing key returns
// (0, false); present-but-unparseable returns (0, false) too -
// callers wanting to surface parse errors should use GetString and
// strconv.Atoi themselves.
func (a EngineArgs) GetInt(key string) (int, bool) {
	v, ok := a.GetString(key)
	if !ok {
		return 0, false
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, false
	}
	return n, true
}

// GetFloat parses Known[key] as float64. Same fallback semantics as
// GetInt.
func (a EngineArgs) GetFloat(key string) (float64, bool) {
	v, ok := a.GetString(key)
	if !ok {
		return 0, false
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return 0, false
	}
	return f, true
}

// parseOllamaArgs handles the daemon env list form: tokens are
// KEY=VALUE pairs separated by whitespace. Any token without "=" goes
// to Unknown verbatim; any KEY not in ollamaKnownEnvKeys also goes to
// Unknown. We do NOT enforce a leading "OLLAMA_" prefix on Unknown
// tokens because operators occasionally set bare envs the daemon picks
// up via libc (rare; still pass through).
func parseOllamaArgs(raw string) (EngineArgs, error) {
	args := EngineArgs{Raw: raw, Known: map[string]string{}}
	if strings.TrimSpace(raw) == "" {
		return args, nil
	}
	tokens, err := shlex(raw)
	if err != nil {
		return EngineArgs{}, fmt.Errorf("ENGINE_ARGS: %w", err)
	}
	for _, tok := range tokens {
		eq := strings.IndexByte(tok, '=')
		if eq <= 0 {
			args.Unknown = append(args.Unknown, tok)
			continue
		}
		key := tok[:eq]
		val := tok[eq+1:]
		if norm, ok := ollamaKnownEnvKeys[key]; ok {
			args.Known[norm] = val
		} else {
			args.Unknown = append(args.Unknown, tok)
		}
	}
	return args, nil
}

// parseCmdlineArgs handles the cmdline-argv form (vLLM and SGLang).
// Recognised tokens match table by exact flag string ("--max-model-len"
// or "--max-model-len=8192"); unknown flags drop into Unknown along
// with their value (if peeked).
//
// Boolean detection is heuristic: a known flag with no peeked value
// (next token is empty / starts with '-' / is a KEY=VALUE env-style
// token) is recorded as "true". An unknown flag whose next token
// looks like a value gets BOTH tokens preserved in Unknown so the
// downstream wrapper can still pass --unknown-flag value through.
func parseCmdlineArgs(raw string, table map[string]string) (EngineArgs, error) {
	args := EngineArgs{Raw: raw, Known: map[string]string{}}
	if strings.TrimSpace(raw) == "" {
		return args, nil
	}
	tokens, err := shlex(raw)
	if err != nil {
		return EngineArgs{}, fmt.Errorf("ENGINE_ARGS: %w", err)
	}
	i := 0
	for i < len(tokens) {
		tok := tokens[i]
		if !strings.HasPrefix(tok, "-") {
			args.Unknown = append(args.Unknown, tok)
			i++
			continue
		}
		// --flag=value form
		if eq := strings.IndexByte(tok, '='); eq != -1 {
			flag := tok[:eq]
			val := tok[eq+1:]
			if norm, ok := table[flag]; ok {
				args.Known[norm] = val
			} else {
				args.Unknown = append(args.Unknown, tok)
			}
			i++
			continue
		}
		// Bare flag; peek next.
		flag := tok
		hasNext := i+1 < len(tokens)
		nextLooksLikeVal := hasNext &&
			!strings.HasPrefix(tokens[i+1], "-") &&
			!strings.Contains(tokens[i+1], "=")
		if norm, ok := table[flag]; ok {
			if nextLooksLikeVal {
				args.Known[norm] = tokens[i+1]
				i += 2
			} else {
				args.Known[norm] = knownFlagPresent
				i++
			}
			continue
		}
		args.Unknown = append(args.Unknown, tok)
		if nextLooksLikeVal {
			args.Unknown = append(args.Unknown, tokens[i+1])
			i += 2
		} else {
			i++
		}
	}
	return args, nil
}

// parseLlamacppArgs handles llama.cpp's dual-track (cmdline OR
// LLAMA_ARG_*=val env). Per-token detection: tokens that don't start
// with '-' AND contain '=' are treated as env form (LLAMA_ARG_<NAME>=v),
// everything else flows through the cmdline path with llamacppKnownFlags.
//
// Mixing forms in one ENGINE_ARGS is supported but discouraged in
// the engine-native argument syntax.
func parseLlamacppArgs(raw string) (EngineArgs, error) {
	args := EngineArgs{Raw: raw, Known: map[string]string{}}
	if strings.TrimSpace(raw) == "" {
		return args, nil
	}
	tokens, err := shlex(raw)
	if err != nil {
		return EngineArgs{}, fmt.Errorf("ENGINE_ARGS: %w", err)
	}
	i := 0
	for i < len(tokens) {
		tok := tokens[i]
		// env form
		if !strings.HasPrefix(tok, "-") && strings.Contains(tok, "=") {
			eq := strings.IndexByte(tok, '=')
			key := tok[:eq]
			val := tok[eq+1:]
			if norm, ok := llamacppKnownEnvs[key]; ok {
				args.Known[norm] = val
			} else {
				args.Unknown = append(args.Unknown, tok)
			}
			i++
			continue
		}
		if !strings.HasPrefix(tok, "-") {
			args.Unknown = append(args.Unknown, tok)
			i++
			continue
		}
		if eq := strings.IndexByte(tok, '='); eq != -1 {
			flag := tok[:eq]
			val := tok[eq+1:]
			if norm, ok := llamacppKnownFlags[flag]; ok {
				args.Known[norm] = val
			} else {
				args.Unknown = append(args.Unknown, tok)
			}
			i++
			continue
		}
		flag := tok
		hasNext := i+1 < len(tokens)
		nextLooksLikeVal := hasNext &&
			!strings.HasPrefix(tokens[i+1], "-") &&
			!strings.Contains(tokens[i+1], "=")
		if norm, ok := llamacppKnownFlags[flag]; ok {
			if nextLooksLikeVal {
				args.Known[norm] = tokens[i+1]
				i += 2
			} else {
				args.Known[norm] = knownFlagPresent
				i++
			}
			continue
		}
		args.Unknown = append(args.Unknown, tok)
		if nextLooksLikeVal {
			args.Unknown = append(args.Unknown, tokens[i+1])
			i += 2
		} else {
			i++
		}
	}
	return args, nil
}
