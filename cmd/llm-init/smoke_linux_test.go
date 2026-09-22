//go:build linux

package main

import (
	"os"
	"sync"
	"testing"
	"time"
)

// TestRun_GracefulShutdown_Linux exercises the SIGINT -> server.Shutdown
// -> exit 0 contract that the rest of the smoke suite intentionally
// skips. Pre-v1.0.4 the assertion lived inside TestRun_StartsControlPlane
// and TestRun_M4ReachesReadyAndDataPlane behind a `runtime.GOOS ==
// windows ? t.Skip()` guard, so on Windows the shutdown path silently
// went unverified and the rest of the assertions were entangled with a
// leak-free goroutine wait. Splitting the shutdown into a Linux-only
// test:
//
//   - keeps the cross-platform smoke tests pure (they don't need to
//     unwind run()'s lifecycle);
//   - never silently skips on Linux, where the actual production
//     shutdown path runs;
//   - is `//go:build linux` so the compiler enforces the constraint
//     instead of a runtime branch.
//
// Windows is excluded by the build tag because os.FindProcess(self).Signal
// returns syscall.EWINDOWS — the host kernel does not deliver
// os.Interrupt to its own process. macOS is excluded purely to keep the
// test deterministic on a single CI runner; the bug we'd catch (failure
// to wire SIGINT into the controlplane.Server.Shutdown path) is identical
// on Linux and macOS.
func TestRun_GracefulShutdown_Linux(t *testing.T) {
	port, err := freePort()
	if err != nil {
		t.Fatalf("freePort: %v", err)
	}
	dir := t.TempDir()
	env := mapEnv(map[string]string{
		// Minimal config — we don't care that the engine is unreachable;
		// the lifecycle will park in PhaseInit (failed WaitAlive) and
		// the control plane will still be up to receive the signal. The
		// shutdown path doesn't depend on PhaseReady. MODEL_MODE is
		// required since the v1.1 config cutover; RUN_DIR must be a
		// writable temp dir so ensure() can seed the spec on Linux CI.
		"ENGINE_KIND":  "vllm",
		"MODEL_NAME":   "qwen2.5-7b",
		"MODEL_MODE":   "chat",
		"MODEL_SOURCE": "hf://Qwen/Qwen2.5-7B-Instruct --revision 0123456789abcdef0123456789abcdef01234567",
		"RUN_DIR":      dir + "/run-state",
		"PORT":         port,
	})
	stdout, _ := os.Create(dir + "/out")
	t.Cleanup(func() { stdout.Close() })
	stderr, _ := os.Create(dir + "/err")
	t.Cleanup(func() { stderr.Close() })

	var (
		wg   sync.WaitGroup
		code int
	)
	wg.Add(1)
	go func() {
		defer wg.Done()
		code = run(nil, stdout, stderr, env)
	}()

	if err := waitFor("http://127.0.0.1:"+port+"/livez", 30*time.Second); err != nil {
		dumpServerLogs(t, stderr.Name())
		t.Fatalf("server never came up: %v", err)
	}

	self, err := os.FindProcess(os.Getpid())
	if err != nil {
		t.Fatalf("FindProcess(self): %v", err)
	}
	if err := self.Signal(os.Interrupt); err != nil {
		t.Fatalf("Signal: %v (build tag should have excluded non-POSIX)", err)
	}

	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("run() did not exit within 10s after SIGINT (httpServer.Shutdown stuck?)")
	}
	if code != exitOK {
		t.Errorf("exit code = %d, want %d", code, exitOK)
	}
}
