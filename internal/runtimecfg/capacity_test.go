package runtimecfg

import (
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/enginecapacity"
)

func readCard(t *testing.T, s *Store) map[string]any {
	t.Helper()
	data, err := os.ReadFile(s.Snapshot().Runtime.ModelSpecPath)
	if err != nil {
		t.Fatalf("read card: %v", err)
	}
	var card map[string]any
	if err := json.Unmarshal(data, &card); err != nil {
		t.Fatalf("decode card: %v", err)
	}
	return card
}

func TestRecordCapacity_WritesTheBlockAndTheCard(t *testing.T) {
	t.Parallel()
	s, _ := newStore(t)

	changed, err := s.RecordCapacity(enginecapacity.Capacity{
		ContextSize:    32768,
		MaxConcurrency: 33,
		PoolTokens:     1048576,
		Source:         enginecapacity.SourceSGLangServerInfo,
		ReportedAt:     time.Unix(1700000000, 0).UTC(),
		FromArgs:       []string{enginecapacity.FieldContextSize},
	})
	if err != nil {
		t.Fatalf("RecordCapacity: %v", err)
	}
	if !changed {
		t.Fatal("changed = false on the first reading")
	}

	block, ok := s.Snapshot().Spec.Extensions["capacity"].(map[string]any)
	if !ok {
		t.Fatalf("extensions.capacity = %#v, want a map", s.Snapshot().Spec.Extensions["capacity"])
	}
	if block["pool_tokens"] != float64(1048576) {
		t.Errorf("pool_tokens = %#v, want 1048576", block["pool_tokens"])
	}
	if block["source"] != string(enginecapacity.SourceSGLangServerInfo) {
		t.Errorf("source = %#v", block["source"])
	}

	// Durable, not only in memory: Router reads whatever the next boot
	// loads off the PVC.
	ext, ok := readCard(t, s)["extensions"].(map[string]any)
	if !ok {
		t.Fatal("card has no extensions object")
	}
	if _, ok := ext["capacity"]; !ok {
		t.Errorf("card extensions = %v, want a capacity block", ext)
	}
}

// A card written by a build that knows nothing about capacity has to keep
// whatever else is in extensions.
func TestRecordCapacity_KeepsSiblingExtensions(t *testing.T) {
	t.Parallel()
	s, _ := newStore(t)
	before := s.Snapshot()
	before.Spec.Extensions = map[string]any{"translate": map[string]any{"languages": []any{"en"}}}
	s.cfg = before

	if _, err := s.RecordCapacity(enginecapacity.Capacity{
		ContextSize: 4096, Source: enginecapacity.SourceEngineArgs,
	}); err != nil {
		t.Fatalf("RecordCapacity: %v", err)
	}
	ext := s.Snapshot().Spec.Extensions
	if _, ok := ext["translate"]; !ok {
		t.Errorf("extensions = %v, want translate kept", ext)
	}
	if _, ok := ext["capacity"]; !ok {
		t.Errorf("extensions = %v, want capacity added", ext)
	}
}

// The reporter re-reads on every restart and the card lives on a shared
// PVC. A reading that says the same thing must not cost a write.
func TestRecordCapacity_SameNumbersLaterIsNotAChange(t *testing.T) {
	t.Parallel()
	s, _ := newStore(t)
	first := enginecapacity.Capacity{
		ContextSize:    8192,
		MaxConcurrency: 4,
		PoolTokens:     32768,
		Source:         enginecapacity.SourceLlamacppProps,
		ReportedAt:     time.Unix(1700000000, 0).UTC(),
	}
	if _, err := s.RecordCapacity(first); err != nil {
		t.Fatalf("RecordCapacity: %v", err)
	}
	stat, err := os.Stat(s.Snapshot().Runtime.ModelSpecPath)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}

	later := first
	later.ReportedAt = time.Unix(1700009999, 0).UTC()
	changed, err := s.RecordCapacity(later)
	if err != nil {
		t.Fatalf("RecordCapacity: %v", err)
	}
	if changed {
		t.Error("changed = true for a reading that only advanced the clock")
	}
	// The stored timestamp stays the one the numbers were first seen at.
	block := s.Snapshot().Spec.Extensions["capacity"].(map[string]any)
	if got := block["reported_at"]; got != "2023-11-14T22:13:20Z" {
		t.Errorf("reported_at = %v, want the first reading's", got)
	}
	after, err := os.Stat(s.Snapshot().Runtime.ModelSpecPath)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if !after.ModTime().Equal(stat.ModTime()) {
		t.Error("the card was rewritten for a no-op reading")
	}
}

func TestRecordCapacity_ChangedNumbersRewrite(t *testing.T) {
	t.Parallel()
	s, _ := newStore(t)
	base := enginecapacity.Capacity{
		ContextSize: 8192, MaxConcurrency: 2,
		Source: enginecapacity.SourceLlamacppProps,
	}
	if _, err := s.RecordCapacity(base); err != nil {
		t.Fatalf("RecordCapacity: %v", err)
	}
	wider := base
	wider.MaxConcurrency = 4
	changed, err := s.RecordCapacity(wider)
	if err != nil {
		t.Fatalf("RecordCapacity: %v", err)
	}
	if !changed {
		t.Fatal("changed = false after the engine relaunched with four slots")
	}
	block := s.Snapshot().Spec.Extensions["capacity"].(map[string]any)
	if block["max_concurrency"] != float64(4) {
		t.Errorf("max_concurrency = %#v, want 4", block["max_concurrency"])
	}
}

// A block of some other shape -- a hand edit, or something else's write --
// is replaced rather than left in place, because a consumer cannot read it.
func TestRecordCapacity_ReplacesAnUnreadableBlock(t *testing.T) {
	t.Parallel()
	s, _ := newStore(t)
	before := s.Snapshot()
	before.Spec.Extensions = map[string]any{"capacity": "12288"}
	s.cfg = before

	changed, err := s.RecordCapacity(enginecapacity.Capacity{
		ContextSize: 4096, Source: enginecapacity.SourceEngineArgs,
	})
	if err != nil {
		t.Fatalf("RecordCapacity: %v", err)
	}
	if !changed {
		t.Fatal("changed = false over a block nothing can read")
	}
	if _, ok := s.Snapshot().Spec.Extensions["capacity"].(map[string]any); !ok {
		t.Errorf("extensions.capacity = %#v, want a map",
			s.Snapshot().Spec.Extensions["capacity"])
	}
}

// A capacity block must not survive as the only thing keeping a card that
// cannot be persisted; the error names the persist failure so a caller
// warns rather than failing a serving model.
func TestRecordCapacity_PersistFailureIsReported(t *testing.T) {
	t.Parallel()
	s, dir := newStore(t)
	next := s.Snapshot()
	next.Runtime.ModelSpecPath = dir + "/no-such-dir/nested/model-spec.json"
	s.cfg = next
	if err := os.WriteFile(dir+"/no-such-dir", []byte("not a directory"), 0o644); err != nil {
		t.Fatalf("write blocker: %v", err)
	}

	if _, err := s.RecordCapacity(enginecapacity.Capacity{
		ContextSize: 4096, Source: enginecapacity.SourceEngineArgs,
	}); !errors.Is(err, ErrPersist) {
		t.Errorf("err = %v, want ErrPersist", err)
	}
	// The in-memory card is untouched, so the next attempt has the same
	// work to do rather than believing it already published.
	if _, ok := s.Snapshot().Spec.Extensions["capacity"]; ok {
		t.Error("extensions.capacity was swapped in despite the write failing")
	}
}
