# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- A chat card whose `max_output_tokens` is unset, or as large as its
  `context_size`, now gets a quarter of the window, derived at boot and on
  `PUT /api/model-spec` alongside `context_size`. Router publishes the value
  and clients size `max_tokens` from it; a card reporting its whole window
  (131072 of 131072) had clients reserving the entire context for the reply,
  which the KV budget refused as `kv_budget_exhausted`. A ceiling the author
  set below the window is kept.

- A request the engine refuses to accept — nothing listening, which is what
  a card edit's relaunch looks like until the next health probe — is now
  503 `not_ready` with `Retry-After: 5`, the same answer the readiness gate
  gives, instead of 502 `upstream_unreachable`. Nothing reached the engine,
  so resending is safe; a connection the engine accepted and then dropped is
  still 502.

- `MODEL_MODE=music_generation` no longer trims `/v1` paths. It joins
  `chat`, `audio`, `tts` and `translate` as a pass-through mode, so a music
  engine can answer subresources under a generation that this repository
  never enumerated and refuse the ones it does not implement itself. The
  whitelist 404'd `/v1/music/generations/{id}/lyrics-alignment` on a staged
  ACE-Step engine that implements it; the application worked around that by
  pointing its shared entrance at the engine instead of Model Console, which
  cost Router the control plane it reads from the same host root and left
  the model permanently un-callable. `GET /api/endpoints` still lists only
  the standard music rows — the catalogue describes what this repository
  knows, not what the engine has.

- Audio documentation and contract coverage now distinguish the three
  WebSocket routes from async-capable ordinary HTTP operations, retain the
  engine-reported sample-rate metadata, and state the current engine-owned
  task lifetime: 1800 seconds or until the audio engine restarts.

### Added

- Non-LLM sibling engines can now publish their real execution width through
  `GET /api/engine-capacity`. Model Console accepts only a positive integer,
  records measurements as `extensions.capacity` with
  `source: engine_capacity`, and retries on readiness/relaunch/config changes
  without affecting readiness. Engines that cannot implement the endpoint may
  declare the same read-only value with `ENGINE_MAX_CONCURRENCY`, recorded as
  `source: engine_config` only after the measured endpoint remains unavailable.
  Queue depth, queue limits, current load, and quota concurrency are never used
  as engine capacity; when neither source is reliable the field stays absent.

- `POST /translate`, `POST /translate/batch` and `POST /detect` now declare
  what the translation cost on `X-Model-Usage-Prompt-Tokens`,
  `-Completion-Tokens` and `-Total-Tokens`, the same headers
  `/translate/transcript` has always sent. `Completer.Complete` read the chat
  upstream's usage and discarded it, so every call on these three routes
  reached Router as zero tokens — not measured as zero, never reported at
  all, which is what a spend row with an empty quota column was showing. A
  batch declares the sum of its items; `/detect` declares the classification
  it paid for, including when the answer cannot be parsed, because that turn
  was still spent. An all-zero usage sends no header, so a reader can still
  tell "not reported" from "measured zero". `total` is filled from prompt
  plus completion when the upstream omits it.

- "The engine has not reported its capabilities yet" is now a state of its
  own, distinct from "the engine has nothing to report". Until an engine
  answers `/api/engine-spec`, the static rows belonging to another modality
  are listed as unavailable with a reason naming the missing report, instead
  of being advertised and then withdrawn: an audio application used to claim
  `/v1/chat/completions` and `/v1/embeddings` for the length of its boot, and
  a dashboard rendered in that window — or a Router model sync landing in it
  — read that as settled. An engine that answers 404, 405 or 501 has
  reported, in the only way an engine written before this contract can, and
  its data plane is still described from llm-init's own catalog; a 5xx or an
  unreachable port has not. Once an engine has declared its routes, a fetch
  that fails afterwards no longer withdraws them.

- The engine-spec relay is now observable. Every read of the engine's
  `/api/engine-spec` counts under `llm_init_engine_spec_fetch_total{result}`,
  and every declared endpoint the relay refuses counts under
  `llm_init_engine_spec_rows_dropped_total{reason}`, where the reason names
  the field that failed. Both failures used to be silent and both produce
  the same empty report, so an operator could not tell an engine that
  declares nothing from one that could not be asked, and a capability lost
  to a single missing `sync_supported` left no trace anywhere between the
  engine and the gateway that then refused the route. The relay also logs
  these outcomes, on transition rather than per poll.

- The data-plane metrics now name the audio surface. `/v1/audio/*` and the
  ElevenLabs-shaped `/v1/voices`, `/v1/text-to-speech`, `/v1/text-to-voice`,
  `/v1/speech-to-speech` and `/v1/history` paths were all collapsed into
  `route="other"`, so a failing transcription and a failing synthesis were
  the same series. Each capability now has its own route label, several
  paths per capability collapse to one label, and the engines' deprecated
  `/v1/audio/tasks` alias is measured with `/v1/tasks` rather than apart
  from it.

- `/api/endpoints` schema v2 now relays the audio engine's stable operation
  IDs, protocol and transport, sync/async support, parameter definitions,
  formats, sample rates, limits and resource scope. Engine-spec v1 remains
  supported; unsafe methods, traversal and `/api/*` engine rows are rejected.

- Music model applications can expose asynchronous `POST /v1/music/drafts`
  and `GET /v1/music/drafts/{id}` song-drafting tasks, where the engine's
  own language model writes a caption, lyrics and a style plan from one
  brief. Mode guards, the `/api/endpoints` catalog and OpenAPI treat them
  as part of `music_generation`, so an application no longer needs a
  separate chat model to reach a first draft.

### Fixed

- A repository holding two files with the same name in different
  directories now reports both. `huggingface_hub` labels a download bar
  with the file name alone, and progress keyed on that label, so the
  second file landed on the first one's state: its bytes were filtered
  out as a rewind and its completion was swallowed by the de-dupe on the
  first. On `FireRedTeam/FireRedTTS3` that lost the whole of
  `redae/model.safetensors` — 3.5 GiB transferred with the aggregate bar
  frozen, the speed at zero and no ETA, until the end-of-pass on-disk
  reconcile jumped the console to 100%. The pass already fetches the Hub
  listing to size the download, and a bar prints the total it counts
  towards, so the size now tells two same-named files apart. Bars the
  listing cannot name still key on the label, where a rewind is read as
  the next file rather than as this one going backwards.
- A file row can no longer be left reading `downloading` on a model that
  is already serving. Leaving the download stage settles every row, so a
  bar whose final line never reached the aggregator cannot strand one,
  and the boot-time reset now clears the file list whatever phase was
  saved. The list described what the previous process was transferring;
  restoring it meant the two stuck FireRedTTS3 rows came back on every
  restart, and a snapshot saved as `ready` skipped the reset entirely. A
  boot with nothing to fetch therefore starts with no rows rather than
  with the previous run's, and the aggregate bar still reports the
  reconciled on-disk total.
- A `--kv-unified-per-slot` ceiling no longer disables the llama.cpp
  oversubscription warning or the KV gate when the ceiling still lets
  every slot claim the whole pool. The flag used to opt out on presence
  alone, so `-c 12288 -np 2 -kvu --kv-unified-per-slot 12288` looked
  safe and still oversubscribed. The ceiling now opts out only when the
  slots' shares add up to no more than the pool.
- An engine restart is now visible in `phase` while it happens. A model
  being relaunched moves `ready` → `loading` as soon as the engine stops
  answering, instead of holding `ready` for the whole 60-second grace
  window and landing back on `ready` — which left a console that had just
  asked for the restart, or edited `engine_args`, with nothing to show and
  no reason to poll faster. The grace still bounds it: an engine that does
  not come back reaches `degraded` on the same schedule as before, so a
  dead engine cannot park in a phase that reads as progress.
- A relaunch that finishes quickly is no longer missed entirely. The engine
  is probed every second while `Tracker.Relaunching` reports one, instead of
  at the 10-second steady-state interval: a healthy relaunch takes the engine
  away about two seconds after the signal and gives it back in under ten, so
  the whole dip could fall between two probes and the phase never moved at
  all. Measured on a real deployment as an 8.4-second outage with no phase
  movement logged, and as a 6.7-second lag on the outages that were caught.
  `RelaunchProbeInterval` configures it; a value that is not shorter than
  `HealthInterval` is ignored, and the fast cadence lasts only as long as the
  tracker's own observation window. `Tracker.Request` also prompts the loop to
  end its current wait (`Config.OnRequest` wired to
  `Options.RelaunchRequested`), because the gap is chosen at the end of the
  previous one: a ten-second wait armed just before the signal outlasts the
  relaunch it was meant to catch. Measured on a deployment, the same restart
  reached `loading` in 0.8s when the loop was already on the fast cadence and
  7.2s when it was not.
- `GET /api/engine/restart` answers `confirmed` for an engine that was
  already down when the restart was signaled and then came back. Only the
  dip used to count as evidence, so restarting a wedged engine — the usual
  reason to press the button — reported `unverified` and left
  `supervision` at `unknown` for the life of the run dir, which is what
  made a client warn about a restart that had in fact worked. Down when
  signaled and still down when the window closes remains `unverified`.
- A capability key in `MODEL_SUPPORTS` that this build does not recognize no
  longer stops this container from booting. It is kept and tagged in
  `extensions._unknown_flag`, which is what an unrecognized key read off a
  model card has always got. An engine base's chart passes a manifest's
  declaration through verbatim, so the old fail-fast turned a typo in a
  published OlaresManifest — or a key Router ratified before this mirror
  shipped — into an application that would not start, explaining itself only
  in its own log. An audio application also waits on its engine container,
  which still exits on a capability its base does not implement; that check
  asks whether the image can serve the capability rather than whether the
  vocabulary is known. `MODEL_REASONING_EFFORT` still rejects an unknown
  level: the engine's chat template raises on one, so it breaks requests
  rather than overstating a capability.
- Runtime edits to `extensions.translate` now survive a full application
  restart. An existing model card is authoritative for the language catalog;
  `TRANSLATE_LANGUAGES` and `TRANSLATE_PAIRS` seed only the first card instead
  of replacing a Router edit when the next Pod starts.
- Hugging Face multi-file and whole-repo downloads no longer look finished
  after the first file. The Hub tree now pins `bytes_total` and
  `files_total` for any `--include` / `--exclude` / `--subdir` (or the
  whole repository when there is no include), and the dashboard keeps
  showing "downloading" for the whole `phase=download` window. The CLI
  `Fetching N files` banner raises the file-count floor if the tree
  cannot be reached; a later source in the same pass can raise it
  further.
- Hugging Face tqdm byte labels are parsed as SI (1000-based), matching
  the CLI and the Hub tree. Treating `4.22G` as GiB used to inflate a
  4.7 GB repo to 5.0 GB and fill the bar while files were still left.
- Download speed and ETA stay at 0 until a bandwidth-bound file is
  transferring (8 MiB, or 1% of the pinned total). Handshake-sized
  config files no longer paint `4 KB/s` and a multi-hundred-hour ETA;
  the bar, byte totals and `N / M` file count still move from the
  first file. The dashboard names the file in that window
  (`Downloading config.json`) instead of a static "fetching files"
  stand-in; speed and ETA appear once a large file is in flight.
- The download size line raises decimals while a pass is still going,
  so a 4.68 GB / 4.70 GB transfer is not rounded to `4.7 GB / 4.7 GB`.
  The bar stays at most 99% until every file in the pass has finished.

### Changed

- The download card lists every file in the pass as soon as the Hub tree
  is known, instead of only the file currently in flight. Waiting rows
  keep a 0% bar (and `0 / size` when the tree knew the length); the
  active row shows its own byte progress. After `loading` / `ready`
  the card folds once so the services below stay on screen; expanding
  it still shows the finished totals and each file's size. `/api/progress`
  adds a `files[]` array (`path`, `status`, per-file bytes) alongside
  `files_total` / `current_file`.

### Added

- Music model applications can expose asynchronous `POST /v1/music/formats`
  and `GET /v1/music/formats/{id}` preflight tasks. Model Console mode guards,
  endpoint catalog, proxying, docs, and OpenAPI now include the contract while
  keeping engine-specific lyric/caption adaptation outside llm-init.

- **Engine-owned capability extensions are projected into the live model card.**
  Once a matching v1 `/api/engine-spec` is available, its `extensions`
  namespaces overlay `GET /api/model-spec`, allowing creative engines to
  declare operations such as music repaint without baking engine-specific
  fields into Model Console. `extensions.capacity` remains reserved for Model
  Console's measured runtime reporter.
- **Music is a first-class Model Console mode.** `MODEL_MODE=music_generation`
  has a strict route table for `GET /v1/models` and the asynchronous
  `/v1/music/generations*` contract. `ENGINE_KIND=music` uses the generic proxy
  adapter at `http://music-engine:8001`; readiness waits for that engine, the
  endpoint catalog and dashboard expose Music, and the model-card/OpenAPI
  contract accepts the canonical mode without adding a `music` alias.
- **TTS speed capabilities are model-card data.** A deployment may seed
  `extensions.tts.voice_settings.speed` from the all-or-none
  `TTS_SPEED_MIN`, `TTS_SPEED_DEFAULT`, and `TTS_SPEED_MAX` variables; an
  existing card remains authoritative. The range is validated as finite,
  positive, and ordered, and the audio proxy contract now pins resumable
  Range/status/header transparency together with the settings, history, and
  task API documentation.
- **`MODEL_MODE=tts` is a mode of its own**, not a flavour of `audio`.
  Speech synthesis answers ElevenLabs-shaped voice routes that recognition
  engines do not have, so the card, `/api/model-spec`, and the capability
  keys `supports_tts` / `supports_tts_clone` / `supports_tts_design` /
  `supports_tts_custom` now live under `tts`. Existing audio applications
  stay `audio` until their OAC sets `MODEL_MODE=tts` explicitly; the
  value is never inferred.
- **How many requests the engine serves at once is derived, never stored**:
  the width is read out of the card's `engine_args` when a caller asks for
  it — llamacpp `-np`, vllm `--max-num-seqs`, sglang
  `--max-running-requests`, ollama `OLLAMA_NUM_PARALLEL` — and is not a
  card field. It cannot be one: a card is parsed by whichever llm-init the
  application was installed with, an unknown top-level field is refused
  outright, and the card lives on the shared cache PVC and outlives the app
  that wrote it. An app installed from the test index and then reinstalled
  from the release index would meet a card its own build cannot read, and
  fail to boot on the ordinary path. Requests past the width are not
  refused, they queue: on a llama.cpp running `-np 1`, twenty concurrent
  calls all returned 200, the deferred gauge peaked at nineteen, and the one
  that waited 118 seconds still succeeded. Nothing above the engine could
  see any of that, so a queue read as a slow model — and the two have
  opposite remedies. Nothing is reported when the flags state no width: the
  engine then applies its own default, and none of the four is a number
  readable from here.
- **`GET /api/engine/load` reports queue depth**: how many requests the
  engine has in flight, how many are waiting behind them, and how many
  slots it was launched with. llama.cpp publishes the first two as
  gauges and nothing above it read them, so the fact that a call was
  queued rather than slow existed only inside the engine. Readings are
  cached for two seconds, failures included, so a caller polling it
  cannot become the load it is measuring. Other engine kinds get 501
  with the reason named rather than a zero that would read as idle.
- **The engine's own account of its capacity is published on the card**,
  at `extensions.capacity`: the per-request window, the number of requests
  the scheduler runs at once, and the whole KV pool, each with the endpoint
  that answered and when. The launch flags state a request, not a fact —
  llama.cpp caps the window at the model's training context, vLLM has to
  finish a profiling run before it knows how many KV blocks fit, SGLang
  computes the pool from a memory fraction and derives the width from it,
  and Ollama lowers `num_ctx` when memory is short without saying so. Each
  engine is asked on its own terms (`/props`, `/server_info`, `/api/ps`,
  `/metrics`) once per event that could have changed the answer rather than
  on a timer: these engines size the cache when they load the model, so a
  relaunch is the only thing that moves it. **Field by field,
  not block for block** — a reading is routinely partial, `/props` gives no
  pool total and vLLM's `/metrics` gives nothing else, so replacing the
  whole declaration would drop what only the flags know and filling gaps
  alone would keep a `-c` the engine had already cut down. A reading whose
  numbers contradict the flags is a warning naming both, because that means
  this repository's arithmetic about that engine is wrong and the log is the
  only place it would ever show. Written under `extensions` rather than as
  a top-level field for the reason above: an older build refuses an unknown
  one outright, and the card outlives the app that wrote it.
- **A shared KV pool is no longer oversold**: on a llama.cpp running a
  unified cache with more than one slot and no per-slot ceiling, a request
  reserves room for the prompt the engine counts for it plus a bounded
  allowance for the answer, and holds it until the response ends. Past the
  pool, callers wait; the ones that wait out their budget get
  `503 kv_budget_exhausted` with a `Retry-After`, which the Router already
  treats as retryable. Unified mode admits a request against its slot's
  window and never against the pool, so the pool running out is discovered
  mid-decode and takes down **every sequence generating at that moment** —
  including ones already streaming to somebody else. Turning one caller's
  overreach into one caller's wait is the whole trade. The prompt is
  measured with the engine's own tokenizer (`/apply-template` then
  `/tokenize`, two loopback hops) and only when the answer could change the
  decision; below an eighth of the pool a request is charged the loose bound
  its body length implies and skips both. Requests that occupy no cache pass
  through unmetered, as does one larger than the entire pool — waiting
  cannot make that fit, and only the engine's refusal carries the real
  limit. No other engine gets a gate: split mode already answers with a 400
  naming the limit, and vLLM and SGLang both refuse cleanly when full.

### Changed

- **A parameter rule that lists its values is now enforced, not just
  published**: a request naming a value outside a card's `options` is
  answered 400 `invalid_parameter_value`, with the accepted values in the
  message, before it reaches the engine. Only rules with `type: string`
  and a non-empty `options` are checked; numeric ranges and rules without
  options are still forwarded untouched, because a range is something the
  engine enforces on its own terms and a rule without options makes no
  claim about a domain. The enum half is different: llama.cpp's chat
  template raises inside Jinja for a reasoning level it does not know, so
  the caller was handed a 502 carrying a Python-looking stack that named
  neither the parameter at fault nor the levels that would have worked —
  while the card one hop away listed them. The rules are read at startup,
  so editing them through `PUT /api/model-spec` stores and serves the new
  list but does not change what the data plane accepts until the app
  restarts; that response now names `parameter_rules` in
  `X-Model-Spec-Pending-App-Restart`.

### Removed

- **`GET /api/model-spec/file` has been removed.** The public control-plane
  contract is `GET /api/model-spec`, which returns the validated,
  boot-normalized card Router consumes. The raw-file twin had no production
  caller and could disagree only after an operator changed the mounted file
  behind the running process. The on-disk `model-spec.json` remains the
  runtime source of truth and `PUT /api/model-spec` still persists it
  atomically.

### Fixed

- **A llama.cpp that was not told `-np` is no longer described as serving
  one request in the whole of `-c`.** Both halves of that were wrong, and
  wrong in the direction that hides the consequence. The server resolves an
  absent `-np` to **four** slots and turns the shared pool **on**, so the
  width was understated fourfold and every slot may claim the entire `-c` —
  while this build reported the window as `-c` divided by a slot count it
  believed was one. `-no-kvu` on its own does not opt out either: the auto
  override lands after it. The window now branches on the shape of the cache
  and pads to 256 the way the server does, the width reports four when the
  flags leave it to the server, and a card that oversubscribes its pool
  warns at startup naming the three ways out. The flags this needs are also
  recognised for the first time (`-kvu` / `-no-kvu` /
  `--kv-unified-per-slot`, with the negation on a key of its own, since a
  bare flag is recorded as present and one shared key would read a disabled
  pool as an enabled one). Separately, `LLAMA_ARG_N_PARALLEL` — the variable
  llama.cpp actually reads — was absent from the known-env table, which held
  only a spelling llama.cpp has never read, so a card declaring its width
  through the environment declared it to nobody.
- **`/translate/transcript` accepts a single turn's reply that carries no
  list number.** Hy-MT2-1.8B leaves the number off a lone line often
  enough that turns it had in fact translated reached the caller as
  `unaligned`, and the retry-then-halve tree makes this the *common* shape
  of the failure rather than a rare one: a group that fails twice is divided
  down to single turns, and each of those calls is then asked to number a
  reply with one line in it. Numbering is what makes a reply matchable, so a
  group without it is still rejected whole; one turn is the exception
  because there is no other line the answer could land on, which is also why
  numbering it wrong is now tolerated. Two cases stay unaligned: a reply cut
  off by the token budget still gets the same turn with double the room,
  because half a sentence reads as a finished translation forever if it is
  taken, and a one-turn reply carrying several lines is refused, because a
  model answering one turn with several lines is translating the reference
  material too. An unaligned reply is now logged, since the turn reaches the
  caller as the single word `unaligned` and what separates a model that will
  not number its answer from one answering something else entirely is the
  answer itself.
- **Editing `engine_args` at runtime now reaches the KV pool gate and the
  capacity reading.** Both had settled their answer while the process was
  starting, so the one configuration that most needs them — a llama.cpp
  arriving at a shared pool through a card edit rather than booting into one
  — got neither. On yaotest004, changing a deployment from `-np 8` to a pool
  shared across four slots left the card reading 32768 tokens with a
  concurrency of one, and sixteen overlapping prompts failed with the
  engine's 500 and a cancelled decode for every one of them instead of the
  gate's 503. Restarting the deployment was the only thing that corrected
  it, and the flags are on an editable card precisely so that nobody has to.
  - Whether the pool needs accounting for is now asked per request, off
    the card. The engine kind is all that is settled when the gate is
    mounted, because it comes from the environment and no edit moves it.
    The card's output ceiling had the same defect with a milder symptom.
  - The capacity reading is armed by two events readiness cannot see. One
    is the engine being observed back from a relaunch: it is relaunched
    inside the process, and the health loop holds its last verdict through
    a grace window longer than the relaunch takes, so **readiness never
    dips** and waiting for a dip meant never re-reading. The other is the
    launch flags no longer matching the ones the standing reading was taken
    under, which covers a card edit whose relaunch never happened — the
    measurement is still true of the running engine, but it cannot stand as
    the answer for flags it predates.

  A fresh install and a chart upgrade were never affected: both restart the
  process, and the boot-time answer was right for the flags it read.

## [1.5.1] - 2026-08-23

### Added

- **Audio endpoint task metadata**: `GET /api/endpoints` now relays the
  optional `async_supported` declaration from an audio engine's
  `/api/engine-spec`. An omitted value remains unknown rather than being
  reported as synchronous.

### Fixed

- **Audio capabilities can no longer drift at runtime**: `PUT
  /api/model-spec` rejects changes to audio `supports`; those capabilities
  are fixed at startup by `MODEL_SUPPORTS` and the engine report. Other
  model-card fields remain editable.
- **The audio catalogue matches the staged applications**: Pyannote model
  sources use the `beclab` mirrors and speaker embedding is documented as
  CPU-only.
- **Local release gates run as documented**: Compose validation checks
  `embed.gpu.yml` together with `embed.yml`, and vulnerability scanning can
  execute the pinned binary even when `GOPATH/bin` is not on `PATH`.

## [1.5.0] - 2026-08-22

### Added

- **The wrapper helpers ship with the image and are published at boot**: the
  supervise loop is what makes `POST /api/engine/restart` work, and it lived
  in a `common.sh` that forty Market charts each inlined a copy of — two of
  which had the loop at all, and five of which carried a trimmed helper with
  no `supervise_engine` in it. Copying the new script into a chart without
  the matching helper fails on the first line, so the fix could not be
  applied one chart at a time either. `common.sh` is now `COPY`d into
  `/usr/share/llm-init/wrappers` and `handoff.PublishWrapperHelpers` writes
  it to `${RUN_DIR}/wrappers/common.sh` before anything else on boot; a
  chart points `WRAPPER_COMMON` at that copy and its script waits for it,
  because nothing orders the two Deployments and the engine container may
  well start first. The default is still the copy next to the script, so
  compose and local development are unaffected. Per-application
  customisation stays with the chart, which is where it belongs: forty
  charts are thirteen runtime variants, and only the helper has to be one
  implementation. `WRAPPER_SRC_DIR` overrides where llm-init reads it from;
  a missing directory is a log line, not a failure. The four Ollama charts
  mount no run dir at all and keep their inlined copy.

- **A restart receipt instead of a claim**: `PUT /api/model-spec` and
  `POST /api/engine/restart` answer with `X-Engine-Restart` (was a signal
  sent), `X-Engine-Restart-Supervision` (has a signal ever been observed
  stopping this engine) and `X-Engine-Restart-Generation`. The old
  `X-Engine-Restarted` promised a completed relaunch that no synchronous
  answer can support, and on a deployment whose wrapper `exec`s the engine
  and leaves nobody watching the file it was wrong every time. What became
  of a particular request lands on
  `GET /api/engine/restart` once it concludes, including `no-supervisor`.

- **`X-Model-Spec-Pending-App-Restart` on a card write**: `mode` chose which
  routes were mounted, `supports.supports_reasoning` the thinking gate and
  `extensions.translate` the language catalogue, all when the process
  started. A save that changes them is stored and served but not obeyed, and
  the header names them so the operator is not told the edit took effect.
  Distinct from the engine restart above: nothing inside the container can
  restart the application.

### Fixed

- **The restart button did nothing on an audio, embedding or OCR
  application**: `POST /api/engine/restart` bumps `engine_restart`, and the
  only thing that reads it is a wrapper's supervise loop — but `audio.sh`,
  `embed.sh`, `clipembed.sh` and `ocrlayout.sh` `exec`ed the engine at the
  top level, so a wedged engine could not be relaunched from the console at
  all. All four now run under `supervise_engine`, with the one-off bootstrap
  (sentinel, model path, binary probe) left outside the loop so a missing
  path still fails immediately instead of looping. They deliberately do not
  wait for the `engine_args` handoff: `ParseEngineArgs` requires Model
  Console's own `ENGINE_ARGS` to be empty for these kinds, so the handoff is
  always empty and there is never a card argument to wait for. What the audio
  charts put in the engine container's `ENGINE_ARGS` is that engine's own
  setting and is left alone. The engine container runs the copy of the wrapper its own
  chart ships, so an existing deployment keeps the old behaviour until it
  syncs the new script; `GET /api/engine/restart` is what tells the two
  apart.

- **A restart left the old engine running and holding the port**: the
  supervised process was an intermediate subshell rather than the engine —
  `supervise_engine` backgrounds a `run_<engine>` shell function, and the
  binary it ran was that subshell's child. Killing the subshell reparented
  the engine to PID 1, where it kept its listener, so the relaunch could not
  bind. The process-group kill was supposed to cover this and is exactly the
  part that does not work in a Pod (`dash` needs a controlling terminal to
  put the job in its own group). Observed in the `embed-server` image: one
  restart, two live engines. All eight `run_<engine>` functions now `exec`
  their binary, so the pid the supervisor stops is the engine itself;
  `tests/wrappers/supervise_test.sh` asserts that the stopped pid is the one
  the engine reported and checks the shape of all eight, because reproducing
  the failure needs a real engine holding a real port.

- **A wrapper died on an empty `engine_args` handoff**: reading it logged the
  retained value's length, and the embed / clipembed / OCR charts do not
  declare `ENGINE_ARGS` at all, so `${#ENGINE_ARGS}` aborted the wrapper under
  `set -u` on the first supervised launch. The audio charts do declare it, for
  their own engine to read rather than for Model Console, which is what that
  branch is retaining in the first place. A non-zero engine
  exit also lost its `engine exited status=` log line, because `set -e` took
  the shell down on `wait` before the line was written.

- **Nothing ran the wrapper tests**: `tests/wrappers/` was referenced by no
  workflow, no make target and no doc, which is how its one test kept
  forbidding `supervise_engine` in the audio wrapper — the very change above
  — without anybody noticing. The CI `shellcheck` job is now `wrappers`: it
  shellchecks both directories and runs `tests/wrappers/*_test.sh`, which
  drive the four wrappers and `supervise_engine` itself against fake engine
  binaries. Locally: `make test-wrappers`.

- **An embedding or OCR application accepted requests it does not serve**:
  `/v1/` is registered as a catch-all and the adapter forwards whatever
  arrives, so `POST /v1/chat/completions` on an embedding application
  reached an engine with no such route and the client got that engine's
  404, in that engine's format, saying nothing about this application —
  while `GET /api/endpoints` had already reported chat as unavailable. The
  catalogue's per-mode whitelist and the port now read one table
  (`internal/moderoutes`), and a path the mode does not serve is `404`
  `endpoint_not_served` with no `Retry-After`. `chat`, `audio` and
  `translate` stay pass-through: audio's routes are declared by the engine
  at runtime rather than written down here, and a whitelist that omitted
  them would refuse the application's whole purpose.

- **`/healthz` could show a green light through an outage**: `status` and
  `ready` were computed separately, and `status` only asked whether the
  engine was alive. An engine that restarted without the model it had been
  given therefore read `ok` beside `"ready": false` for the length of the
  health loop's grace window, while every `/v1` request was refused with
  503. All three readiness answers — the `/v1` gate, `/readyz`, and both
  `/healthz` fields — now come from one `lifecycle.Readiness` computed at
  one instant, so they cannot describe the same moment differently.
  `/readyz`'s `reason` prefers the lifecycle's own recorded error over
  `engine_not_alive`, which only named the consequence.

- **A forced re-download could CrashLoop a restarting engine**: the run
  dir's sentinel says "ready, load this" about the path in `model_path`,
  and `force=true` deletes exactly that file before spending minutes
  fetching its replacement. A wrapper starting in that window — a kubelet
  restart, a CrashLoop, a supervise loop's first launch — read the sentinel
  and `exec`ed a path that was gone. The URL and ollama-URL paths now clear
  the sentinel first, so such a wrapper waits for the pass instead;
  `hf://` does not, because `--force-download` never invalidates the
  snapshot path in between.

### Changed

- **The card's `name` is the alias, and it cannot be changed online**:
  `Model.Name` is now derived from the card rather than left at
  `MODEL_NAME`, so a card that outlived a chart upgrade on the shared cache
  PVC no longer has the data plane advertising one name while Router
  registers another (`ollama-native` answered 404 for the name Router was
  routing). `MODEL_NAME` diverging from the card is a boot warning, and a
  `PUT` that renames the model is `400` — the alias is read once at boot by
  the request rewriter, so a half-applied rename is an outage rather than a
  stale field.

- **One owner for the configuration that moves**: `internal/runtimecfg.Store`
  serialises the card write — normalise, persist, publish `engine_args`,
  swap memory — under one lock, and every handler reads a snapshot. `PUT`
  used to be two unlocked assignments into a struct that `/api/config`,
  `/api/endpoints` and `/healthz` read from other goroutines, with maps
  inside it. `context_size` is re-derived on that path too, so a card edited
  to a new window stops advertising the old one until the next boot.

## [1.4.0] - 2026-08-22

### Added

- **The audio duration headers are part of the contract**: OpenAPI carries
  `X-Audio-Input-Duration-Seconds` and `X-Audio-Output-Duration-Seconds` on
  the audio routes, and the matching `input_duration_seconds` /
  `output_duration_seconds` on the Task schema. Audio is the one mode a
  gateway can only bill from a response header — the payload streams through
  undecoded, so the seconds exist nowhere else. `api.html` says which
  capability reports which of the two, and that an absent header means
  unmetered rather than zero: a per-second biller reading a missing header
  as 0 undercharges every stream. `audio.html` points at it.

- **A channel column on the audio capability table**: the table listed twelve
  delivered capabilities with no way to tell that `stt_stream` is a WebSocket
  and `align` is not, or which of them accept `async=1`. The answer is not in
  this repository — it is `audio-engines`' `wrapper/catalog.py` — and the
  column says so.

### Changed

- `TASK_TTL_S` and `TASK_QUEUE_MAX` are documented as the engine's values
  rather than this repository's. Nothing in the Go code reads them and no
  test here can check the numbers the page quotes.

## [1.3.11] - 2026-08-21

### Added

- **`POST /api/retry?level=remote`**: one pass that asks the upstream
  whether it still serves the recorded bytes — a HEAD comparing ETag and
  `Content-Length` for a URL, the Hub tree API comparing per-file LFS
  oids for `hf://`. The commit is not the test: a revision whose README
  moved resolves to a new commit with every weight file untouched, and
  calling that drift would re-download gigabytes to land the same bytes.
  Reachable only from `/api/retry` — as a standing default it would make
  every boot depend on the upstream being up, which is what the records
  exist to avoid. "The upstream agrees", "the upstream has moved" and
  "the upstream could not be reached" are three distinct outcomes, and
  the third is never reported as the first.

- **`VERIFY_ON_DRIFT`** (`report` | `follow`, default `report`): what a
  remote check does when the upstream has moved. Reporting is the
  default because a model application is usually serving requests by the
  time anybody runs one, and following swaps the weights under it.

- **Re-registration when the engine loses its model**: an Ollama daemon
  that restarts answers again with an empty model list, and nothing
  else puts back what `Register` pushed into it. The health loop now
  asks for one ensure pass on the `ready → degraded` edge when the
  engine is up and reports the model gone — one ask per outage, and
  only for the daemon-backed engine, since the proxy adapters'
  `Register` is a no-op that a pass cannot fix.

### Changed

- **`context_size` is derived from `engine_args`, not written by hand**:
  every card write reads the window off the launch flags per engine —
  llamacpp `-c` divided by `-np`, vllm `--max-model-len`, sglang
  `--context-length`, ollama `OLLAMA_NUM_CTX` — and overwrites whatever
  was submitted. The card lives on the shared cache PVC and outlives the
  application, so a stale value quietly overruled a chart that had since
  been raised, and every consumer believed the card: a caller sized its
  prompt against a window the engine did not serve. The llamacpp
  division is why the flags are parsed per engine rather than taken as
  one number — `-c` is the whole KV cache split across parallel slots,
  and a conversation only ever gets one slot. Where the flags pin down
  no window a stored value survives, since writing a zero says less than
  saying nothing. Disk still wins over the chart seed, which is what
  keeps a Dashboard edit across a restart; it now says so at WARN with
  both values, so an operator who raised `ENGINE_ARGS` and redeployed
  can see that the redeploy changed nothing.

- **Ensure no longer runs on a timer**: a ready model no longer re-runs
  ensure on a fixed one-hour cadence. Every pass reached the upstream
  registry, so an unreachable huggingface.co or ollama.com moved a model
  that was complete on disk into `degraded`. Ensure now runs on boot, on
  `POST /api/retry`, and on the self-heal above. Two consequences:
  `last_verify_at` only refreshes on those, and a moving HF revision is
  no longer followed — `POST /api/retry?level=remote` is how to ask.

- **A pass reuses bytes it can verify, without asking the upstream**:
  each completed download now leaves a record beside the bytes — under
  `<repo-cache>/.llm-init/` for `hf://`, `<file>.llm-init.json` for
  `http(s)://` and the first hop of `ollama://<URL>`. A later pass that
  finds the record describing the source it was configured with, and the
  files it lists still intact per `VERIFY_LEVEL`, skips that source
  whole: no `hf download` subprocess, not even a HEAD. A machine with no
  route to the upstream boots a model it already has, which `hf
  download` cannot do on its own — against a fully-cached snapshot it
  still asks the Hub what the revision resolves to now.

  The record holds a file list, sizes, whatever digests were free and
  the source's identity. It does not hold the URL: a presigned link
  carries a credential in its query, and this file sits on a volume
  other applications mount, so identity is the URL's sha256 and the
  human-readable field keeps scheme, host and path only. An `hf://`
  record lists the files that request's own `--include` matched, not
  everything in a snapshot several requests share.

- **`phase=download` is entered when a pass finds something to fetch**,
  not on the way in. It takes the model offline for `/v1/*` and pins the
  byte budget, which costs an upstream metadata request per source; a
  pass that reuses everything now does neither. Such a pass still moves
  through `loading`, which also gates `/v1/*` — what it saves is a
  window that scales with the model and depends on the network.

- **`VERIFY_LEVEL` is consumed**: it was parsed, exposed on
  `GET /api/config` and read by nothing. It now sets how hard a pass
  checks bytes before reusing them — `size` compares lengths, `sha256`
  re-hashes every file with a recorded digest.

- **`ollama://<URL>` registration reuses the recorded digest.** Ollama's
  `Register` hashed the file on every boot to arrive at a string already
  written down beside it; on a multi-gigabyte GGUF that is minutes per
  boot. No record, or none with a digest, and the file is hashed as
  before.

### Removed

- **`VERIFY_INTERVAL`**: retired and fail-fast, since there is no longer
  a timer for it to set. Ensure's triggers are boot, `POST /api/retry`
  and health-loop self-heal. `runtime.verify_interval` is gone from
  `GET /api/config`.

### Breaking

- **Perf diagnostic subsystem removed**: `POST /api/diag/perf` and
  `GET /api/diag/perf/last` now return 404, and the dashboard no longer
  has a Performance card or its throughput/TTFT charts. The probe's
  throughput numbers were never implemented (Prefill/Decode always
  rendered `—`) and it was the only diagnostic that sent real generation
  traffic straight at the engine. `GET /api/diag/gpu` is unchanged,
  including numeric hints and `?include_engine_native=true`; cold-start
  timing stays on `/api/progress` as `engine_cold_start_ms`. Prometheus
  loses `diag_perf_runs_total` and `diag_perf_duration_seconds`.

## [1.3.10] - 2026-08-20

### Added

- **`MODEL_REASONING_EFFORT` / `MODEL_REASONING_EFFORT_DEFAULT`**: declare
  which reasoning levels a deployment actually accepts without hand-writing
  the whole model card. The CSV becomes the card's `reasoning_effort`
  `parameter_rules` entry, which is what Router projects onto `/v1/models`,
  so a client can read the ladder instead of guessing at it. Levels are
  restricted to `none` / `minimal` / `low` / `medium` / `high` / `xhigh` /
  `max` and unknown ones fail at boot: a level the engine's chat template
  does not know is not a spare option, it is a request that errors out for
  whoever trusted the card. Order in the variable is irrelevant — the
  option list is written weakest-first so a menu renders in ladder order.

  The env is authoritative on every boot, replacing that one rule on an
  existing card and leaving the others alone. This is the second deliberate
  exception to disk-wins after `extensions.translate`, and for the same
  reason: every app installed before the variable existed has a card on a
  hostPath that a seed-only path would never reach. An admin edit to this
  rule through Router is therefore reverted at the next restart.

## [1.3.9] - 2026-08-20

### Added

- **`UPSTREAM_RESPONSE_HEADER_TIMEOUT`**: wait for the engine's first
  HTTP response header on proxied `/v1/chat/completions`,
  `/v1/messages`, and Ollama inference (Go duration, default `5m`,
  range `[5s, 30m]`). The previous 60s hard cap bounced large-GGUF
  prefills with `timeout awaiting response headers`.

## [1.3.8] - 2026-08-17

### Changed

- **CI Go toolchain**: pin `1.26.5` → `1.26.6` so `govulncheck` clears
  the Go 1.26.6 stdlib patch set (`GO-2026-6218` / `6090` / `6089` /
  `5972` / `5026`). Consumer floor in `go.mod` stays `1.23.0`.

- Documentation now uses canonical HTML and OpenAPI links, distinguishes
  implemented protocols from delivered model applications, and aligns the
  security policy with the complete unauthenticated read, write, and
  expensive surface.

### Fixed

- `APP_URL` now reaches the dashboard's Overview "API URL" panel. The
  front-end read `Runtime.PublicURL` while the response carried the value
  under a `runtime` parent, so the lookup always missed and the panel
  silently fell back to `window.location.origin` — showing the in-cluster
  address instead of the configured public entrance.

### Breaking

- `GET /api/config` renames the fields inside its `runtime` and `log`
  objects from Go identifiers to snake_case, matching every other field in
  the response (`runtime.RunDir` → `runtime.run_dir`,
  `runtime.PublicURL` → `runtime.public_url`, `log.UpstreamTrace` →
  `log.upstream_trace`, and so on). Both objects are now declared in the
  OpenAPI `ConfigRedacted` schema, which previously omitted them.
  `runtime.verify_interval` marshals as an integer nanosecond count.

## [1.3.7] - 2026-08-13

### Changed

- **Proxy rewrite admission**: `/v1/embeddings` body cap depends on
  `ENGINE_KIND`. `clipembed` admits up to 16 MiB (inline image data
  URLs); `embed` and other kinds stay at 1 MiB. Chat vision paths are
  unchanged (16 MiB).

### Breaking

- `MODEL_SOURCE` is now the only active model-source env. Replace
  `MODEL_SOURCE_NUM` and indexed source variables with comma-separated
  entries; align `MODEL_SOURCE_LOCAL` positions and leave non-URL
  placeholders empty:

  ```dotenv
  # Before
  MODEL_SOURCE_NUM=2
  MODEL_SOURCE_1=hf://org/main
  MODEL_SOURCE_1_ROLE=main
  MODEL_SOURCE_2=https://example.com/aux.gguf
  MODEL_SOURCE_2_LOCAL=/cache/aux.gguf

  # After
  MODEL_SOURCE=hf://org/main,https://example.com/aux.gguf
  MODEL_SOURCE_LOCAL=,/cache/aux.gguf
  ```

  The first source binds the engine; later sources are download/preload
  only.

  A llama.cpp vision projector keeps its own role, now requested with the
  inline `--role mmproj` flag on an `hf://` source instead of
  `MODEL_SOURCE_<i>_ROLE=mmproj`. It is still downloaded inline and still
  stays out of `extra_model_path`, so `model_path` keeps pointing at the
  main GGUF:

  ```dotenv
  # Before
  MODEL_SOURCE_NUM=2
  MODEL_SOURCE_1=hf://org/vlm --include model.gguf
  MODEL_SOURCE_1_ROLE=main
  MODEL_SOURCE_2=hf://org/vlm --include mmproj.gguf
  MODEL_SOURCE_2_ROLE=mmproj

  # After
  MODEL_SOURCE=hf://org/vlm --include model.gguf,hf://org/vlm --include mmproj.gguf --role mmproj
  ```

  `--role` is only accepted on `hf://` sources, only with the value
  `mmproj`, never on the first source, and on at most one source. A second
  `hf://` source without it is an extra download rather than a projector:
  the weights land in `extra_model_path` and the engine never receives
  `--mmproj`, so vision goes quiet instead of failing.

## [1.3.6] - 2026-08-11

### Added

- **Translate mode**: `MODEL_MODE=translate` mounts the MTran-compatible
  `/translate`, `/translate/batch`, `/languages`, and `/detect` surface
  at the host root. Its per-model language catalog resolves from
  model-spec extensions or deployment overrides, with a built-in
  MTran-compatible default based on Mozilla translations-models-v2
  (54 languages, 102 pairs). Detection uses a CJK
  heuristic before falling back to model inference.

- **Translate deployment examples**: a dedicated compose stack and local
  llama.cpp scenario document the model card, engine arguments, and
  host-root API contract.

- **API contracts**: OpenAPI now covers Translate and explicit engine
  restart.

### Changed

- Translate activation is mode-based only; it does not use a
  `MODEL_SUPPORTS` capability key.

### Breaking

- Model apps must set `mode=translate` and remove `supports_translate`.

## [1.3.5] - 2026-08-03

### Added

- **API contracts**: OpenAPI now covers OCR submission and the
  cross-engine async task relay.

## [1.2.1] - 2026-06-04

Patch: robust multi-file download progress and decimal dashboard units.

**Fixed:**

- Multi-file `bytes_total` now resolves Hugging Face `resolve` URLs
  (comma-form `ollama://` sources, e.g. E2) via the Hub tree API, so the
  download denominator covers every file even when a companion file's HEAD
  omits `Content-Length`.
- The download budget is pinned only when every file's size is known;
  partial HEAD success no longer freezes an undersized `bytes_total`.

**Changed:**

- Dashboard renders byte sizes in decimal (1000-based) GB to match the
  sizes shown on Hugging Face / vendor model cards (e.g. ~4.1 GB for E2
  instead of 3.8 GiB).

## [1.2.0] - 2026-06-03

Feature release: comma-separated `MODEL_SOURCE` for multi-file preload, stable
multi-file download progress on the dashboard, and a new local example (E2).

**Added:**

- **Comma-separated `MODEL_SOURCE`**: list multiple sources in one env (first
  segment binds the engine as `main`; later segments are `extra` and are
  downloaded only). `MODEL_SOURCE_LOCAL` aligns 1:1 when URL segments are
  present. Mutually exclusive with `MODEL_SOURCE_NUM`. See
  the documented `MODEL_SOURCE` rules.
- **Download `bytes_total` preflight**: lifecycle sums remote file sizes (URL
  HEAD for `ollama://` / `https://` overloads; HF Hub tree API for indexed
  `main` + `mmproj`) before streaming so `/api/progress` does not jump when a
  second file starts. Falls back to per-file `OnFileStart` when estimation
  fails.
- **Example E2** (`examples/local/E2-qwen35-vlm.env`): Ollama engine, comma
  dual URL for unsloth Qwen3.5 main GGUF + mmproj preload (`extra`).

**Changed:**

- OpenAPI `ModelSource.role` documents `extra` (comma form) alongside `main` /
  `mmproj`.

## [1.1.0] - 2026-06-02

The "unified configuration cutover" release. The runtime
contract collapses to one canonical surface each: model source (`MODEL_SOURCE` /
indexed `MODEL_SOURCE_<i>`), engine startup params (single `ENGINE_ARGS`), and
model spec (`MODEL_MODE` / `MODEL_SUPPORTS` seed → `MODEL_SPEC_PATH`, editable via
`PUT /api/model-spec`). Cache paths move to deployment-managed mounts. Contracts:
the shared-cache, model-spec, and engine-argument contracts.
`HF_ENDPOINT` / `HF_TOKEN` stay deployment-level
(`--endpoint` inside `MODEL_SOURCE` is blacklisted, fail-fast → `HF_ENDPOINT`).

**Breaking:**

- Retired all v1.0 llm-init-specific envs (`HF_REPO`, `HF_FILE`, `MODEL_URL`, `OLLAMA_MODEL`, `MODEL_DIR`, `ENGINE_URL`, `CONTEXT_LENGTH`, `GGUF_*`, `MODEL_TYPE`, `ENGINE_GPU_MEMORY_UTILIZATION`, `ENGINE_LLAMACPP_*`, …) → fail-fast with migration hint. Standard third-party envs stay ignored, not claimed.
- Removed deprecated metric `llm_init_engine_rtt_seconds` (was renamed in v1.0.5).
- `/api/diag/gpu` derives llama.cpp residency from `n_gpu_layers` in `ENGINE_ARGS`.
- `phase` enum is now `init | download | loading | ready | degraded | failed` (removed dead `verifying`, renamed `register` → `loading`). `loading` spans "downloaded + registered" → "engine alive"; proxy engines reach `ready` only after `WaitAlive` (no more racing ahead of the engine). Any `loading` failure is terminal `failed` (was retriable `degraded`), with the loop staying alive for `POST /api/retry`.

**Security:**

- `GET /api/diag/gpu` no longer returns the `engine_native` dump by default (opt in via `?include_engine_native=true`).
- Upstream HTTP error bodies in `HTTPStatusError` / `last_error` are scrubbed of echoed credentials (e.g. S3 `Authorization` / `X-Amz-Signature`).

**Fixes:**

- HF download panel stuck at `0 / 0`: runner now sets `TQDM_POSITION=-1` so byte progress is emitted/parsed for both LFS and Xet backends.
- vLLM / SGLang crash-loop (`exec: python: not found`): wrappers now `exec python3 -m ...`.
- "Run perf" button gated on live `engine_alive` (not just persisted `phase=ready`), with tooltip + inline "cannot run" message.
- Header phase badge folds to amber `degraded` when `phase=ready` but `engine_alive=false` (instead of stale green).
- "Model exists" is kind-aware (display only): file-backed engines report `yes` once download completes; Ollama still trusts the daemon.

**Removed:**

- `/api/progress` drops `files_total` / `files_completed` (under-counted with Xet/small files) and `current_file` (flaky cosmetic); dashboard drops those rows. `OnFilesTotal` / `OnFileActive` / `State.CurrentFile` removed. Byte progress unchanged.
- `/healthz` drops `model_loaded` (only Ollama populated it; VRAM residency is not part of readiness). `adapter.ReadyState.Loaded` and the Ollama `/api/ps` probe removed.

**Dependencies:**

- Bumped pinned Ollama image `0.23.4` → `0.30.0` across compose / k8s / integration / `SECURITY.md`.

## Released

- **1.0.9** (2026-05-20) — Anthropic `/v1/messages` + transparent `/v1/responses`; Ollama-native compat (`/api/version|tags|ps|show`); 4-tab dashboard rebuild wiring `/api/diag/*` + Prometheus.
- **1.0.8** (2026-05-18) — Docker Hub cutover (publish on tag from CI); `Integration (llamacpp)` unblocked; CI observability.
- **1.0.7** (2026-05-18) — Diagnostic surface (`/api/diag/*` + `/api/model-spec`); proxy engine startup ordering fix; image runs as UID 1000; branch protection.
- **1.0.6** (2026-05-16) — Lifecycle phase transition cleanup; manifest race-condition fix.
- **1.0.5** (2026-05-16) — `POST /api/retry?force=true` + `?level=sha256` wired into ensure; lint + race + CI hardening.
- **1.0.3** (2026-05-15) — `/metrics` Prometheus surface (11 collectors); OpenAI compat test suite; Go toolchain split (floor `1.23.0`, builder `1.26`).
- **1.0.2** (2026-04-30) — Doc-only: DESIGN.md major rewrite.
- **1.0.1** (2026-04-15) — `MODEL_NAME` client alias decoupled from upstream Ollama tag; `redacted_source` in `/api/config`.
- **1.0.0** (2026-05-14) — First public release: 4 engine adapters; HF / URL / Ollama sources; lifecycle state machine; control plane + OpenAI-compatible data plane.
