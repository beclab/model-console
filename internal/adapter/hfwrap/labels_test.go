package hfwrap

import "testing"

// fireRedTree is the shape that made this index necessary: one
// repository, two directories, one file name. The sizes are the ones
// the Hub reports for FireRedTeam/FireRedTTS3.
var fireRedTree = []TreeEntry{
	{Path: ".gitattributes", Size: 1585},
	{Path: "fireredtts3_instruct/config.json", Size: 403},
	{Path: "fireredtts3_instruct/model.safetensors", Size: 8475259708},
	{Path: "redae/config.json", Size: 853},
	{Path: "redae/model.safetensors", Size: 3775160672},
}

func TestLabelIndex_NilForEmptyTree(t *testing.T) {
	t.Parallel()
	if ix := newLabelIndex(nil); ix != nil {
		t.Fatalf("newLabelIndex(nil) = %v, want nil", ix)
	}
	if _, ok := (*labelIndex)(nil).lookup("a.bin", 10); ok {
		t.Error("a nil index must name nothing")
	}
}

func TestLabelIndex_UniqueNameIgnoresSize(t *testing.T) {
	t.Parallel()
	ix := newLabelIndex(fireRedTree)
	// Only one .gitattributes, so it resolves even before tqdm has
	// settled on a total.
	got, ok := ix.lookup(".gitattributes", 0)
	if !ok || got != ".gitattributes" {
		t.Errorf("lookup = %q, %v; want .gitattributes, true", got, ok)
	}
}

func TestLabelIndex_DuplicateNameResolvedBySize(t *testing.T) {
	t.Parallel()
	ix := newLabelIndex(fireRedTree)
	cases := []struct {
		total int64
		want  string
	}{
		// tqdm prints three significant digits: 8.48G and 3.78G.
		{8480000000, "fireredtts3_instruct/model.safetensors"},
		{3780000000, "redae/model.safetensors"},
		// Small files are printed exactly.
		{403, "fireredtts3_instruct/config.json"},
		{853, "redae/config.json"},
	}
	for _, c := range cases {
		got, ok := ix.lookup(labelBase(c.want), c.total)
		if !ok || got != c.want {
			t.Errorf("lookup(total=%d) = %q, %v; want %q, true", c.total, got, ok, c.want)
		}
	}
}

func TestLabelIndex_DuplicateNameAndSizeIsUnnameable(t *testing.T) {
	t.Parallel()
	ix := newLabelIndex([]TreeEntry{
		{Path: "a/weights.bin", Size: 1000},
		{Path: "b/weights.bin", Size: 1000},
	})
	if got, ok := ix.lookup("weights.bin", 1000); ok {
		t.Errorf("lookup = %q, true; want unnameable so the caller keys on the label", got)
	}
}

func TestLabelIndex_UnknownNameIsUnnameable(t *testing.T) {
	t.Parallel()
	ix := newLabelIndex(fireRedTree)
	if _, ok := ix.lookup("nowhere.bin", 10); ok {
		t.Error("a name absent from the tree must not resolve")
	}
}

func TestLabelBase(t *testing.T) {
	t.Parallel()
	cases := []struct{ in, want string }{
		{"model.safetensors", "model.safetensors"},
		{"redae/model.safetensors", "model.safetensors"},
		{"(…)redtts3_instruct/model.safetensors", "model.safetensors"},
		{"  fireredtts3_instruct/config.json  ", "config.json"},
		{`windows\style\path.bin`, "path.bin"},
	}
	for _, c := range cases {
		if got := labelBase(c.in); got != c.want {
			t.Errorf("labelBase(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestSizeMatchesTQDM(t *testing.T) {
	t.Parallel()
	cases := []struct {
		want, total int64
		match       bool
	}{
		{8475259708, 8480000000, true},  // 8.48G, rounded up
		{3775160672, 3780000000, true},  // 3.78G, rounded up
		{3775160672, 8480000000, false}, // the namesake's bar
		{853, 403, false},               // the other config.json
		{403, 403, true},
		{1000, 0, false},
		{0, 1000, false},
	}
	for _, c := range cases {
		if got := sizeMatchesTQDM(c.want, c.total); got != c.match {
			t.Errorf("sizeMatchesTQDM(%d, %d) = %v, want %v", c.want, c.total, got, c.match)
		}
	}
}
