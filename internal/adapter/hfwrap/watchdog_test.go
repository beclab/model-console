package hfwrap

import (
	"context"
	"os/exec"
	"testing"
	"time"
)

func TestWatchdog_KillsAfterTimeout(t *testing.T) {
	wd := newWatchdog(context.Background(), &exec.Cmd{}, "x/y", 5*time.Millisecond, 30*time.Millisecond)
	wd.Touch()

	deadline := time.Now().Add(2 * time.Second)
	for !wd.Killed() && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	wd.Stop()
	if !wd.Killed() {
		t.Error("watchdog should have flagged kill after stalled timeout")
	}
}

func TestWatchdog_NoKillIfTouched(t *testing.T) {
	// A generous timeout (1s) relative to the touch cadence (every 20ms,
	// 8 times) so a loaded CI runner's scheduling jitter cannot push the
	// inter-touch gap past the timeout and trigger a spurious kill.
	wd := newWatchdog(context.Background(), &exec.Cmd{}, "x/y", 5*time.Millisecond, time.Second)
	for i := 0; i < 8; i++ {
		wd.Touch()
		time.Sleep(20 * time.Millisecond)
	}
	if wd.Killed() {
		t.Error("Killed() should be false while regularly touched")
	}
	wd.Stop()
}

func TestWatchdog_StopIsIdempotent(t *testing.T) {
	wd := newWatchdog(context.Background(), &exec.Cmd{}, "x/y", time.Hour, time.Hour)
	wd.Touch()
	wd.Stop()
	wd.Stop()
	wd.Stop()
}

func TestWatchdog_ContextCancelStops(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	wd := newWatchdog(ctx, &exec.Cmd{}, "x/y", 5*time.Millisecond, time.Hour)
	wd.Touch()
	cancel()
	time.Sleep(30 * time.Millisecond)
	if wd.Killed() {
		t.Error("ctx cancel should stop watchdog without flagging kill")
	}
	wd.Stop()
}
