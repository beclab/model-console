package hfwrap

import (
	"bufio"
	"strings"
	"testing"
)

func TestScanLinesOrCR_SplitsOnBoth(t *testing.T) {
	in := "line1\nline2\rline3\r\nline4"
	s := bufio.NewScanner(strings.NewReader(in))
	s.Split(scanLinesOrCR)
	var got []string
	for s.Scan() {
		got = append(got, s.Text())
	}
	want := []string{"line1", "line2", "line3", "line4"}
	if len(got) != len(want) {
		t.Fatalf("got %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d] got %q want %q", i, got[i], want[i])
		}
	}
}

func TestStripANSIEscapes(t *testing.T) {
	in := "\x1b[2K\x1b[1Aconfig.json: 100%"
	got := stripANSIEscapes(in)
	if got != "config.json: 100%" {
		t.Errorf("got %q", got)
	}
}

func TestParseHFProgressLine(t *testing.T) {
	cases := []struct {
		name           string
		raw            string
		wantOK         bool
		wantLabel      string
		wantDownloaded int64
		wantTotal      int64
	}{
		{
			name:           "kb_units",
			raw:            "config.json: 100%|##########| 2.00kB/2.00kB [00:01<00:00, 1MB/s]",
			wantOK:         true,
			wantLabel:      "config.json",
			wantDownloaded: 2000,
			wantTotal:      2000,
		},
		{
			name:           "model_safetensors_partial",
			raw:            "model.safetensors:  37%|███▋      | 1.46G/3.95G [00:21<00:30, 82.4MB/s]",
			wantOK:         true,
			wantLabel:      "model.safetensors",
			wantDownloaded: 1460000000, // tqdm SI: 1.46 * 1e9
			wantTotal:      3950000000, // 3.95 * 1e9
		},
		{
			// Hub tree size for Qwen3-ASR-1.7B shard 1 is 4220320824.
			// tqdm prints that as 4.22G (SI). 1024-based parsing would
			// claim ~4.53 GiB and the dashboard would overshoot the pin.
			name:           "si_gigabyte_matches_hub_size",
			raw:            "model-00001-of-00002.safetensors: 100%|##########| 4.22G/4.22G [01:00<00:00, 70.0MB/s]",
			wantOK:         true,
			wantLabel:      "model-00001-of-00002.safetensors",
			wantDownloaded: 4220000000,
			wantTotal:      4220000000,
		},
		{
			// hf_xet truncates a long filename desc as "<first40>(…)"
			// (file_download.py xet_get). The trailing ellipsis must not
			// break label capture; the bar is otherwise the standard
			// unit_scale=B tqdm format shared with the LFS path.
			name:           "xet_truncated_desc",
			raw:            "pytorch_model-00001-of-00002.safetensor(…):  50%|█████     | 2.50G/5.00G [00:30<00:30, 80.0MB/s]",
			wantOK:         true,
			wantLabel:      "pytorch_model-00001-of-00002.safetensor(…)",
			wantDownloaded: 2500000000, // 2.5 * 1e9
			wantTotal:      5000000000, // 5.0 * 1e9
		},
		{
			name:   "no_pipe_at_all",
			raw:    "Fetching 7 files",
			wantOK: false,
		},
		{
			name:   "blank",
			raw:    "",
			wantOK: false,
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			ev, ok := parseHFProgressLine(tc.raw)
			if ok != tc.wantOK {
				t.Fatalf("ok=%v want %v (%q)", ok, tc.wantOK, tc.raw)
			}
			if !ok {
				return
			}
			if ev.Label != tc.wantLabel {
				t.Errorf("label=%q want %q", ev.Label, tc.wantLabel)
			}
			tol := int64(1 << 24) // 16 MiB tolerance for floating-point
			if abs(ev.Downloaded-tc.wantDownloaded) > tol {
				t.Errorf("downloaded=%d want ~%d", ev.Downloaded, tc.wantDownloaded)
			}
			if abs(ev.Total-tc.wantTotal) > tol {
				t.Errorf("total=%d want ~%d", ev.Total, tc.wantTotal)
			}
		})
	}
}

func TestParseFetchingFilesCount(t *testing.T) {
	t.Parallel()
	if n, ok := parseFetchingFilesCount("Fetching 1 files"); !ok || n != 1 {
		t.Fatalf("singular wording: n=%d ok=%v", n, ok)
	}
	if n, ok := parseFetchingFilesCount("Fetching 7 files"); !ok || n != 7 {
		t.Fatalf("standalone: n=%d ok=%v", n, ok)
	}
	if n, ok := parseFetchingFilesCount("Fetching 13 files: 30%|███ | 3/13"); !ok || n != 13 {
		t.Fatalf("tqdm label: n=%d ok=%v", n, ok)
	}
	if _, ok := parseFetchingFilesCount("config.json"); ok {
		t.Fatal("regular filename must not parse as a file count")
	}
}

func TestHFIsOuterAggregateLabel(t *testing.T) {
	if !hfIsOuterAggregateLabel("Fetching 7 files") {
		t.Error("Fetching N files should match")
	}
	if hfIsOuterAggregateLabel("config.json") {
		t.Error("regular file label should not match")
	}
}

func abs(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}
