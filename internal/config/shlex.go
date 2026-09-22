package config

import (
	"fmt"
	"strings"
)

// shlex splits s into tokens using a small subset of POSIX shell word
// splitting that is sufficient for both MODEL_SOURCE inline flags and
// ENGINE_ARGS values:
//
//   - Whitespace (space/tab/CR/LF) separates tokens.
//   - Single-quoted runs ('...') preserve their content literally.
//   - Double-quoted runs ("...") preserve their content literally too;
//     no variable expansion or backslash escaping is applied because we
//     run after env interpolation has already happened.
//   - Outside quotes, '\' escapes the next byte (so operators can write
//     KEY=value\ with\ space if they prefer to avoid quotes).
//   - Adjacent quoted/unquoted runs concatenate into one token (so
//     OLLAMA_DEBUG="info verbose" becomes the single token
//     `OLLAMA_DEBUG=info verbose`).
//
// An unbalanced quote or a trailing unescaped backslash is reported as
// "shlex: unbalanced quote" / "shlex: dangling backslash" - callers
// surface these as config_invalid errors.
func shlex(s string) ([]string, error) {
	var tokens []string
	var cur strings.Builder
	inToken := false
	quote := byte(0)
	escaped := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if escaped {
			cur.WriteByte(c)
			escaped = false
			inToken = true
			continue
		}
		if quote == 0 && c == '\\' {
			escaped = true
			inToken = true
			continue
		}
		if quote == 0 && (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
			if inToken {
				tokens = append(tokens, cur.String())
				cur.Reset()
				inToken = false
			}
			continue
		}
		if quote == 0 && (c == '"' || c == '\'') {
			quote = c
			inToken = true
			continue
		}
		if quote != 0 && c == quote {
			quote = 0
			continue
		}
		cur.WriteByte(c)
		inToken = true
	}
	if quote != 0 {
		return nil, fmt.Errorf("shlex: unbalanced quote (%c)", quote)
	}
	if escaped {
		return nil, fmt.Errorf("shlex: dangling backslash")
	}
	if inToken {
		tokens = append(tokens, cur.String())
	}
	return tokens, nil
}
