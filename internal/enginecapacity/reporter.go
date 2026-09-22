package enginecapacity

import (
	"context"
	"log/slog"
	"time"
)

// Reporter keeps a consumer's copy of the capacity in step with the
// engine, by probing once per event that could have changed it.
//
// Per event, rather than on a timer, because capacity is a startup
// decision: these engines size their KV cache when they load the model
// and do not resize it afterwards. What does change it is a relaunch — a
// card edit bumps engine_restart and the engine comes back with
// different flags — so a relaunch is the event worth re-reading on.
//
// Readiness is one way to notice one, and not a reliable way. A relaunch
// in place does not show up in readiness at all: the health loop holds
// its last verdict through a grace period longer than the relaunch
// takes, so Ready never returns false and the dip never happens. Hence
// Restarted, and hence the launch flags being compared against the ones
// the last reading was taken under.
//
// The retries are for a different failure: an engine answers /readyz
// before it has finished publishing its own metrics, so the first probe
// after a restart can miss a gauge that appears a second later. They stop
// as soon as a probe answers, and after retryLimit attempts regardless,
// because an engine that has no such endpoint will never have one and a
// permanent 1-second poll against it is worse than a declared number.
type Reporter struct {
	// Target returns what to probe, read fresh on every attempt rather
	// than captured once. The launch flags are the part that moves: a
	// card edit rewrites them and restarts the engine, so a seed taken at
	// boot would describe an engine that is no longer running — and it is
	// the seed the drift warning compares the measurement against.
	Target func() Options

	// Ready is the readiness verdict this reporter follows, normally
	// lifecycle.Manager.Ready.
	Ready func() bool

	// Restarted identifies the engine's last relaunch, normally
	// handoff.Tracker.Relaunch. A change arms the next reading; nil
	// leaves readiness as the only trigger.
	//
	// Readiness does not catch a relaunch, and that is the common case
	// rather than a corner of it. The engine is relaunched in place by
	// its wrapper, and the health loop holds its last verdict through a
	// grace period longer than the relaunch takes -- so Ready never
	// returns false, the dip below never happens, and the card keeps the
	// capacity of an engine that no longer exists. That number is the
	// one Router admits requests against.
	Restarted func() string

	// Record publishes a reading and reports whether it changed
	// anything. Normally runtimecfg.Store.RecordCapacity.
	Record func(Capacity) (bool, error)

	// Interval is how often Ready is consulted. Defaults to
	// defaultReportInterval. It is not how often the engine is probed.
	Interval time.Duration
}

const (
	// defaultReportInterval is a readiness read, not a network call: one
	// snapshot and one atomic load. Short enough that the card carries a
	// measured capacity within a second of the engine coming up, which
	// matters because Router may read the card before that.
	defaultReportInterval = time.Second

	// retryLimit bounds the attempts spent chasing an endpoint that may
	// not exist. Ten seconds of trying, at the default interval.
	retryLimit = 10
)

// Run follows readiness until ctx is done. It never returns an error: a
// capacity that could not be read is a warning, not a reason to fail a
// model that is otherwise serving.
func (r *Reporter) Run(ctx context.Context) {
	interval := r.Interval
	if interval <= 0 {
		interval = defaultReportInterval
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	// attempts counts probes since the reading was last armed; settled
	// says a probe has answered and there is nothing more to chase.
	attempts := 0
	settled := false
	// probed is the launch flags the last reading was taken against, as
	// the verbatim string the engine was given.
	probed := ""
	probedConfiguredMaxConcurrency := 0

	relaunch := ""
	if r.Restarted != nil {
		relaunch = r.Restarted()
	}

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}

		if r.Ready == nil || r.Target == nil || !r.Ready() {
			// A dip is what arms the next reading. Resetting here rather
			// than on the way back up is what makes a restart re-probe
			// without needing to remember the previous state separately.
			attempts = 0
			settled = false
			continue
		}

		// A relaunch readiness did not see, for the reason on Restarted.
		if r.Restarted != nil {
			if token := r.Restarted(); token != "" && token != relaunch {
				relaunch = token
				attempts = 0
				settled = false
			}
		}

		target := r.Target()
		done := settled || attempts >= retryLimit
		// Flags that changed without a relaunch anybody saw. The reading
		// on the card may still be true of the engine that is running,
		// but it was taken against different flags, so nothing may treat
		// it as the answer for these.
		if done && (target.Args.Raw != probed ||
			target.ConfiguredMaxConcurrency != probedConfiguredMaxConcurrency) {
			attempts = 0
			settled = false
			done = false
		}
		if done {
			continue
		}
		attempts++

		got := Probe(ctx, target)
		for _, w := range got.Warnings {
			slog.Warn("engine capacity: " + w)
		}
		if got.Source != SourceEngineArgs && got.Source != SourceEngineConfig {
			settled = true
		} else if attempts < retryLimit {
			// Nothing measured yet and attempts left. Keep the declared
			// numbers off the card for now: writing them and replacing
			// them a second later would have Router briefly believe a
			// declaration it will not be told was superseded.
			continue
		}
		probed = target.Args.Raw
		probedConfiguredMaxConcurrency = target.ConfiguredMaxConcurrency
		r.record(got)
	}
}

func (r *Reporter) record(c Capacity) {
	if r.Record == nil {
		return
	}
	changed, err := r.Record(c)
	if err != nil {
		slog.Warn("engine capacity: could not publish the reading",
			"err", err.Error(), "source", string(c.Source))
		return
	}
	if !changed {
		return
	}
	slog.Info("engine capacity published",
		"source", string(c.Source),
		FieldContextSize, c.ContextSize,
		FieldMaxConcurrency, c.MaxConcurrency,
		FieldPoolTokens, c.PoolTokens,
		"from_args", c.FromArgs)
}
