package hfwrap

import (
	"regexp"
	"strconv"
	"strings"
)

// scanLinesOrCR is a bufio.SplitFunc that yields tokens delimited by
// either '\r' (tqdm in-place updates) or '\n' (real line break).
// huggingface_hub's CLI emits per-tick tqdm output as lines terminated
// only by '\r'; default bufio.ScanLines never returns until '\n', so
// progress updates would be invisible until the bar rolls over.
//
// Trailing '\n' or '\r' is stripped from the returned token. EOF
// flushes whatever is in the buffer as the final token.
func scanLinesOrCR(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if atEOF && len(data) == 0 {
		return 0, nil, nil
	}
	for i, b := range data {
		if b == '\n' || b == '\r' {
			tok := data[:i]
			// Collapse a CR immediately followed by LF (\r\n) into
			// one boundary so the next call doesn't see an empty
			// line. The reverse (\n\r) does not occur in tqdm
			// output but the symmetric handling keeps the splitter
			// safe against future producers.
			next := i + 1
			if b == '\r' && next < len(data) && data[next] == '\n' {
				next++
			}
			return next, tok, nil
		}
	}
	if atEOF {
		return len(data), data, nil
	}
	return 0, nil, nil
}

// hfANSICSIRE matches the ANSI Control-Sequence-Introducer escapes
// huggingface_hub uses for cursor moves (e.g. "\x1b[2K", "\x1b[A").
// We strip them before regex matching so the parser is robust against
// terminal-style updates.
var hfANSICSIRE = regexp.MustCompile(`\x1b\[[0-9;]*[A-Za-z]`)

// stripANSIEscapes removes ANSI CSI escapes from s.
func stripANSIEscapes(s string) string {
	return hfANSICSIRE.ReplaceAllString(s, "")
}

// hfProgressLineRE matches the tqdm bar shape huggingface_hub's CLI
// emits, e.g.:
//
//	model.safetensors:  37%|███▋      | 1.46G/3.95G [00:21<00:30, 82.4MB/s]
//
// Captures: 1=label  2=percent  3=downloaded number  4=downloaded unit
// 5=total number  6=total unit
//
// `label` may be empty when tqdm omits the desc (e.g. xet outer bar).
var hfProgressLineRE = regexp.MustCompile(
	`^\s*(?:(\S[^:]*?):\s+)?(\d+)%\|[^|]*\|\s*` +
		`([\d.]+)\s*([kKMGTP]?B?)\s*/\s*([\d.]+)\s*([kKMGTP]?B?)\b`)

// hfFetchingFilesRE matches the outer aggregate bar tqdm prints when
// huggingface_hub spawns multiple workers, e.g. "Fetching 13 files".
var hfFetchingFilesRE = regexp.MustCompile(`^Fetching\s+(\d+)\s+files?\b`)

// hfUnitMultiplier maps tqdm's n/total suffixes to bytes.
// huggingface_hub 0.36.2 builds the bar with unit_scale=True and does
// not set unit_divisor, so tqdm.format_sizeof uses SI (1000). `4.22G`
// is 4.22e9, matching the Hub tree; 1024-based GiB overshoots ~7%.
func hfUnitMultiplier(u string) float64 {
	u = strings.ToUpper(strings.TrimSuffix(u, "B"))
	switch u {
	case "":
		return 1
	case "K":
		return 1e3
	case "M":
		return 1e6
	case "G":
		return 1e9
	case "T":
		return 1e12
	case "P":
		return 1e15
	}
	return 1
}

// hfIsOuterAggregateLabel reports whether label belongs to the
// "Fetching N files" outer bar; the aggregator skips these because
// they double-count individual file progress.
func hfIsOuterAggregateLabel(label string) bool {
	return hfFetchingFilesRE.MatchString(strings.TrimSpace(label))
}

// parseFetchingFilesCount extracts N from "Fetching N files" (standalone
// or as a tqdm label). ok is false when the line is not that banner.
func parseFetchingFilesCount(raw string) (int, bool) {
	m := hfFetchingFilesRE.FindStringSubmatch(strings.TrimSpace(stripANSIEscapes(raw)))
	if m == nil {
		return 0, false
	}
	n, err := strconv.Atoi(m[1])
	if err != nil || n <= 0 {
		return 0, false
	}
	return n, true
}

// HFProgressEvent is the parsed shape of one tqdm progress line.
type HFProgressEvent struct {
	Label      string
	Downloaded int64
	Total      int64
	Percent    int
}

// parseHFProgressLine attempts to interpret raw as a tqdm progress
// line. Returns ok=false when raw is not a progress-shaped string.
func parseHFProgressLine(raw string) (HFProgressEvent, bool) {
	clean := stripANSIEscapes(raw)
	m := hfProgressLineRE.FindStringSubmatch(clean)
	if m == nil {
		return HFProgressEvent{}, false
	}
	pct, _ := strconv.Atoi(m[2])
	downNum, _ := strconv.ParseFloat(m[3], 64)
	totalNum, _ := strconv.ParseFloat(m[5], 64)
	dl := int64(downNum * hfUnitMultiplier(m[4]))
	tot := int64(totalNum * hfUnitMultiplier(m[6]))
	return HFProgressEvent{
		Label:      strings.TrimSpace(m[1]),
		Downloaded: dl,
		Total:      tot,
		Percent:    pct,
	}, true
}
