package config

// Known-key tables for ENGINE_ARGS parsing. Entries are derived from
// the supported known keys plus a few commonly-seen flags
// so the dashboard can render them in normalised form. Tables are
// intentionally non-exhaustive: any token llm-init does not match here
// is forwarded to engine_args.Unknown unchanged.
//
// All maps are read-only after init; do not mutate at runtime.

// ollamaKnownEnvKeys maps OLLAMA_* daemon env names to the normalised
// known-key name surfaced on /api/config. Both daemon-only and
// request-injection keys appear here; the request-injection subset is
// applied by the ollama adapter via Engine.Args.GetString /GetInt /
// GetFloat.
var ollamaKnownEnvKeys = map[string]string{
	"OLLAMA_NUM_CTX":           "num_ctx",
	"OLLAMA_KEEP_ALIVE":        "keep_alive",
	"OLLAMA_NUM_PARALLEL":      "num_parallel",
	"OLLAMA_NUM_THREAD":        "num_thread",
	"OLLAMA_MAX_LOADED_MODELS": "max_loaded_models",
	"OLLAMA_MAX_QUEUE":         "max_queue",
	"OLLAMA_FLASH_ATTENTION":   "flash_attention",
	"OLLAMA_KV_CACHE_TYPE":     "kv_cache_type",
	"OLLAMA_REPEAT_PENALTY":    "repeat_penalty",
	"OLLAMA_REPEAT_LAST_N":     "repeat_last_n",
	"OLLAMA_DEBUG":             "debug",
	"OLLAMA_HOST":              "host",
	"OLLAMA_ORIGINS":           "origins",
	"OLLAMA_NOPRUNE":           "noprune",
	"OLLAMA_LOAD_TIMEOUT":      "load_timeout",
	"OLLAMA_NUM_GPU":           "num_gpu",
	"OLLAMA_GPU_OVERHEAD":      "gpu_overhead",
	"OLLAMA_INTEL_GPU":         "intel_gpu",
	"OLLAMA_LLM_LIBRARY":       "llm_library",
	"OLLAMA_TMPDIR":            "tmpdir",
	"OLLAMA_MODELS":            "models",
	"OLLAMA_SCHED_SPREAD":      "sched_spread",
	"OLLAMA_NEW_ENGINE":        "new_engine",
	"OLLAMA_MAX_VRAM":          "max_vram",
	"OLLAMA_MULTIUSER_CACHE":   "multiuser_cache",
	"OLLAMA_NUM_PREDICT":       "num_predict",
	"OLLAMA_TEMPERATURE":       "temperature",
	"OLLAMA_TOP_P":             "top_p",
	"OLLAMA_TOP_K":             "top_k",
	"OLLAMA_SEED":              "seed",
	"OLLAMA_STOP":              "stop",
}

// vllmKnownFlags maps vLLM long-form CLI flags (with the leading "--")
// to normalised keys. vLLM does not have meaningful short forms, so
// only --long-flag entries appear.
var vllmKnownFlags = map[string]string{
	"--max-model-len":             "max_model_len",
	"--gpu-memory-utilization":    "gpu_memory_utilization",
	"--kv-cache-dtype":            "kv_cache_dtype",
	"--cpu-offload-gb":            "cpu_offload_gb",
	"--enforce-eager":             "enforce_eager",
	"--tensor-parallel-size":      "tensor_parallel_size",
	"--pipeline-parallel-size":    "pipeline_parallel_size",
	"--dtype":                     "dtype",
	"--quantization":              "quantization",
	"--max-num-seqs":              "max_num_seqs",
	"--max-num-batched-tokens":    "max_num_batched_tokens",
	"--swap-space":                "swap_space",
	"--trust-remote-code":         "trust_remote_code",
	"--disable-log-stats":         "disable_log_stats",
	"--port":                      "port",
	"--host":                      "host",
	"--max-logprobs":              "max_logprobs",
	"--seed":                      "seed",
	"--block-size":                "block_size",
	"--enable-chunked-prefill":    "enable_chunked_prefill",
	"--disable-custom-all-reduce": "disable_custom_all_reduce",
	"--rope-scaling":              "rope_scaling",
	"--rope-theta":                "rope_theta",
	"--num-scheduler-steps":       "num_scheduler_steps",
}

// llamacppKnownFlags maps llama.cpp short and long CLI flags to
// normalised keys. Both forms map to the same key so the dashboard
// shows a single row regardless of which form the operator wrote.
var llamacppKnownFlags = map[string]string{
	"-c":                "ctx_size",
	"--ctx-size":        "ctx_size",
	"-ngl":              "n_gpu_layers",
	"--n-gpu-layers":    "n_gpu_layers",
	"-fa":               "flash_attn",
	"--flash-attn":      "flash_attn",
	"-ctk":              "cache_type_k",
	"--cache-type-k":    "cache_type_k",
	"-ctv":              "cache_type_v",
	"--cache-type-v":    "cache_type_v",
	"--mmproj":          "mmproj",
	"-t":                "threads",
	"--threads":         "threads",
	"-tb":               "threads_batch",
	"--threads-batch":   "threads_batch",
	"-b":                "batch_size",
	"--batch-size":      "batch_size",
	"-ub":               "ubatch_size",
	"--ubatch-size":     "ubatch_size",
	"--rope-freq-base":  "rope_freq_base",
	"--rope-freq-scale": "rope_freq_scale",
	"--port":            "port",
	"--host":            "host",
	"-np":               "parallel",
	"--parallel":        "parallel",
	"-kvu":              "kv_unified",
	"--kv-unified":      "kv_unified",
	// The negated spellings get a key of their own rather than mapping to
	// kv_unified. parseLlamacppArgs records every bare flag as "true"
	// (knownFlagPresent) and has no notion of negation, so folding them in
	// would store the exact opposite of what the operator wrote.
	"-no-kvu":               "no_kv_unified",
	"--no-kv-unified":       "no_kv_unified",
	"--kv-unified-per-slot": "kv_unified_per_slot",
	"-cb":                   "cont_batching",
	"--cont-batching":       "cont_batching",
	"--mlock":               "mlock",
	"--no-mmap":             "no_mmap",
	"-tp":                   "tensor_split",
	"--tensor-split":        "tensor_split",
	"-mg":                   "main_gpu",
	"--main-gpu":            "main_gpu",
	"--temp":                "temperature",
	"--top-k":               "top_k",
	"--top-p":               "top_p",
	"--repeat-penalty":      "repeat_penalty",
	"--seed":                "seed",
}

// llamacppKnownEnvs maps the LLAMA_ARG_* env-form names to the same
// normalised keys as llamacppKnownFlags. The 1:1 mapping with the
// upstream's common/arg.cpp lets operators pick whichever form fits
// their automation.
var llamacppKnownEnvs = map[string]string{
	"LLAMA_ARG_CTX_SIZE":      "ctx_size",
	"LLAMA_ARG_N_GPU_LAYERS":  "n_gpu_layers",
	"LLAMA_ARG_FLASH_ATTN":    "flash_attn",
	"LLAMA_ARG_CACHE_TYPE_K":  "cache_type_k",
	"LLAMA_ARG_CACHE_TYPE_V":  "cache_type_v",
	"LLAMA_ARG_MMPROJ":        "mmproj",
	"LLAMA_ARG_THREADS":       "threads",
	"LLAMA_ARG_THREADS_BATCH": "threads_batch",
	"LLAMA_ARG_BATCH":         "batch_size",
	"LLAMA_ARG_UBATCH":        "ubatch_size",
	"LLAMA_ARG_N_PARALLEL":    "parallel",
	// LLAMA_ARG_PARALLEL is not a name upstream ever had; `-np` has always
	// read LLAMA_ARG_N_PARALLEL (common/arg.cpp, set_env on the -np opt).
	// Kept because Router's core/engineargs recognises the same misspelling
	// and dropping it on one side only would make the two disagree.
	"LLAMA_ARG_PARALLEL":   "parallel",
	"LLAMA_ARG_KV_UNIFIED": "kv_unified",
	// Upstream derives the negated env name by substitution and treats it as
	// falsey on presence alone, whatever the value: get_value_from_env
	// checks LLAMA_ARG_NO_* first and returns "0" if it is set at all.
	"LLAMA_ARG_NO_KV_UNIFIED":       "no_kv_unified",
	"LLAMA_ARG_KV_UNIFIED_PER_SLOT": "kv_unified_per_slot",
	"LLAMA_ARG_CONT_BATCHING":       "cont_batching",
	"LLAMA_ARG_PORT":                "port",
	"LLAMA_ARG_HOST":                "host",
	"LLAMA_ARG_MLOCK":               "mlock",
	"LLAMA_ARG_NO_MMAP":             "no_mmap",
	"LLAMA_ARG_MAIN_GPU":            "main_gpu",
	"LLAMA_ARG_TENSOR_SPLIT":        "tensor_split",
	"LLAMA_ARG_ROPE_FREQ_BASE":      "rope_freq_base",
	"LLAMA_ARG_ROPE_FREQ_SCALE":     "rope_freq_scale",
}

// sglangKnownFlags maps SGLang launch_server.py CLI flags to
// normalised keys.
var sglangKnownFlags = map[string]string{
	"--mem-fraction-static":  "mem_fraction_static",
	"--max-running-requests": "max_running_requests",
	"--tp":                   "tensor_parallel_size",
	"--tensor-parallel-size": "tensor_parallel_size",
	"--dp":                   "dp_size",
	"--dp-size":              "dp_size",
	"--enable-torch-compile": "enable_torch_compile",
	"--max-total-tokens":     "max_total_tokens",
	"--context-length":       "context_length",
	"--quantization":         "quantization",
	"--dtype":                "dtype",
	"--port":                 "port",
	"--host":                 "host",
	"--schedule-policy":      "schedule_policy",
	"--enable-p2p-check":     "enable_p2p_check",
	"--chunked-prefill-size": "chunked_prefill_size",
	"--max-prefill-tokens":   "max_prefill_tokens",
	"--trust-remote-code":    "trust_remote_code",
	"--disable-radix-cache":  "disable_radix_cache",
	"--enable-mixed-chunk":   "enable_mixed_chunk",
	"--random-seed":          "random_seed",
}

// contextSizeKeys names, per engine, the normalised key above that holds
// the per-request context window. It belongs beside the tables it quotes:
// renaming a key there without renaming it here would leave the window
// underivable, with nothing but a missing card field to say so. llamacpp
// is absent because its flag needs arithmetic; see
// deriveLlamacppContextSize.
var contextSizeKeys = map[EngineKind]string{
	EngineVLLM:   "max_model_len",
	EngineSGLang: "context_length",
	EngineOllama: "num_ctx",
}

// maxConcurrencyKeys names, per engine, the normalised key above that
// holds how many requests the engine will work on at once. Same
// belongs-beside-the-tables reasoning as contextSizeKeys; llamacpp is
// present here because `-np` is the number, with no arithmetic.
//
// Each engine calls it something else and means the same thing: the
// width of the batch the scheduler runs. Anything past it waits — it is
// not refused, which is exactly why the number has to travel: a caller
// who cannot see it reads its own queueing as the model being slow.
var maxConcurrencyKeys = map[EngineKind]string{
	EngineLlamaCpp: "parallel",
	EngineVLLM:     "max_num_seqs",
	EngineSGLang:   "max_running_requests",
	EngineOllama:   "num_parallel",
}
