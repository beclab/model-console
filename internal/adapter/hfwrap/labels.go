package hfwrap

import "strings"

// hfLabelTruncPrefix is what huggingface_hub prepends when it shortens a
// bar's description to the last 40 characters of the file name.
const hfLabelTruncPrefix = "(…)"

// tqdmSizeTolerance is how far a parsed tqdm total may sit from the real
// file size. tqdm formats with three significant digits and SI units
// (unit_scale=True, no unit_divisor), so 3775160672 prints as "3.78G"
// and parses back as 3780000000 — close, never equal. Half a unit in the
// last place is at most 0.5%; 1% leaves room for the rounding to land on
// the wide side of a boundary.
const tqdmSizeTolerance = 0.01

// labelIndex answers "which repository file is this tqdm bar counting?".
//
// huggingface_hub names a bar after the file alone, so a repo holding
// `fireredtts3_instruct/model.safetensors` and `redae/model.safetensors`
// prints two bars that read identically on stderr. Keying progress on
// the label alone made the second bar land on the first one's state,
// where the "bytes greater than this file's last n" delta gate filtered
// every one of its ticks: 3.5 GiB transferred and nothing reported.
//
// The tree listing the pass already fetched carries each file's exact
// size, and tqdm prints the total it counts towards, so the size is what
// tells two same-named files apart.
type labelIndex struct {
	byBase map[string][]TreeEntry
}

// newLabelIndex builds the index. A nil or empty tree yields a nil
// index, which lookup treats as "cannot name anything" — the caller
// then keys on the raw label, as it did before the tree existed.
func newLabelIndex(entries []TreeEntry) *labelIndex {
	if len(entries) == 0 {
		return nil
	}
	ix := &labelIndex{byBase: make(map[string][]TreeEntry, len(entries))}
	for _, e := range entries {
		if e.Path == "" {
			continue
		}
		base := labelBase(e.Path)
		ix.byBase[base] = append(ix.byBase[base], e)
	}
	if len(ix.byBase) == 0 {
		return nil
	}
	return ix
}

// lookup names the repository path the bar belongs to.
//
// ok is false when there is no tree, when nothing in it carries that
// name, or when two files carry both that name and that size. All three
// leave the bar unnameable, and the caller falls back to the label.
func (ix *labelIndex) lookup(label string, total int64) (string, bool) {
	if ix == nil || label == "" {
		return "", false
	}
	candidates := ix.byBase[labelBase(label)]
	switch len(candidates) {
	case 0:
		return "", false
	case 1:
		return candidates[0].Path, true
	}
	var hit string
	matches := 0
	for _, e := range candidates {
		if sizeMatchesTQDM(e.Size, total) {
			hit = e.Path
			matches++
		}
	}
	if matches == 1 {
		return hit, true
	}
	return "", false
}

// labelBase reduces a tqdm label or a repository path to the file name
// both agree on, dropping the "(…)" a long description was shortened
// with.
func labelBase(label string) string {
	label = strings.TrimPrefix(strings.TrimSpace(label), hfLabelTruncPrefix)
	label = strings.ReplaceAll(label, "\\", "/")
	if i := strings.LastIndex(label, "/"); i >= 0 {
		label = label[i+1:]
	}
	return label
}

// sizeMatchesTQDM reports whether a bar counting towards total can be
// the file of size want.
func sizeMatchesTQDM(want, total int64) bool {
	if want <= 0 || total <= 0 {
		return false
	}
	diff := want - total
	if diff < 0 {
		diff = -diff
	}
	return float64(diff) <= tqdmSizeTolerance*float64(want)
}
