/*
 * i18n.js — tiny zero-dependency localisation layer.
 *
 * Loaded BEFORE every other module so window.LLM.t() is available when
 * dashboard / config / gpu run. The active language is derived from the
 * `terminus-language` value Olares injects into the document head; only
 * `en-US` and `zh-CN` are emitted by the platform today, so detection
 * folds anything starting with "zh" to Chinese and everything else to
 * English.
 *
 * Two translation surfaces:
 *   * Static markup — elements tagged with data-i18n / data-i18n-html /
 *     data-i18n-<attr> are localised once, on load, by apply().
 *   * Dynamic strings — the JS modules call LLM.t(key, params); `{name}`
 *     placeholders are interpolated from params.
 */
(function () {
  'use strict';
  const LLM = (window.LLM = window.LLM || {});

  // ─── language detection ───────────────────────────────────────────
  // Priority: <meta name="terminus-language"> → <html terminus-language>
  // → cookie → navigator.language. Olares injects the meta at runtime;
  // the rest are fallbacks for standalone / local-preview use.
  function rawLang() {
    const meta = document.querySelector('meta[name="terminus-language"]');
    if (meta && meta.content) return meta.content;
    const attr = document.documentElement.getAttribute('terminus-language');
    if (attr) return attr;
    const m = document.cookie.match(/(?:^|;\s*)terminus-language=([^;]+)/);
    if (m) return decodeURIComponent(m[1]);
    return (navigator.language || navigator.userLanguage || 'en-US');
  }
  function normalize(v) {
    return /^zh/i.test(String(v || '').trim()) ? 'zh' : 'en';
  }

  const DICT = {
    en: {
      'loading': 'Loading…',
      'engine.label': 'Engine',
      'tab.status': 'Status',
      'tab.config': 'Configuration',

      // lifecycle phase badge (raw phase shown verbatim in en)
      'phase.init': 'init',
      'phase.download': 'download',
      'phase.loading': 'loading',
      'phase.ready': 'ready',
      'phase.degraded': 'degraded',
      'phase.failed': 'failed',

      // model-card download-status badge (byte-driven; see renderPhaseBadge)
      'badge.initializing': 'Initializing',
      'badge.downloading': 'Downloading',
      'badge.downloaded': 'Downloaded',
      'badge.failed': 'Failed',
      'badge.degraded': 'Degraded',

      // progress headline
      'pl.ready': 'Ready',
      'pl.loading': 'Loading model into the engine…',
      'pl.degraded': 'Degraded',
      'pl.failed': 'Failed',
      'pl.downloading_pct': 'Downloading… {pct}%',
      'pl.downloading': 'Downloading…',
      'pl.initializing': 'Initializing…',
      'prog.of': '{done} of {total}',
      'prog.files': '{done} / {total} files',
      'prog.left': '{eta} left',
      'prog.estimating': 'Estimating size',
      'prog.downloading_file': 'Downloading {file}',
      'prog.file_waiting': 'Waiting to download',
      'prog.file_downloading': 'Downloading',
      'prog.file_done': 'Done',
      'prog.retry': '{n} network retry',
      'prog.retries': '{n} network retries',

      // service-status pills
      'st.unknown': 'Unknown',
      'st.ready': 'Ready',
      'st.missing': 'Missing',
      'st.running': 'Running',
      'st.waiting': 'Waiting',

      // action button
      'act.redownload': 'Re-download',
      'act.retry': 'Retry',
      'act.cancel': 'Cancel',
      'act.pause': 'Pause',
      'act.resume': 'Resume',
      'tip.pending_backend': 'Pending backend support',
      'meta.config_unavailable': 'Configuration unavailable',
      'confirm.proceed': '{label}? This downloads the model again and reruns setup.',

      // copy
      'copy': 'Copy',
      'copied': 'Copied',

      // diagnostics
      'diag.title': 'Diagnostics',
      'diag.last_error': 'Last error',
      'diag.retry_count': 'Setup retries',
      'diag.transport_retries': 'Network retries',
      'diag.help': 'Retry re-downloads the model and reruns setup after you fix the source or network issue',

      // service status
      'svc.title': 'Service status',
      'dlonly.title': 'Download-only mode',
      'dlonly.desc': 'ENGINE_KIND is not configured. Only the model file is downloaded, and the inference engine is not started.',
      'svc.model': 'Model',
      'svc.engine': 'Engine',
      'svc.model_name': 'Model name',
      'svc.who': 'Connection source',
      'svc.aud_olares': 'Apps in Olares',
      'svc.aud_lan': 'Devices on your network',
      'svc.aud_remote': 'Remote',
      'svc.format': 'API format',
      'svc.base_url': 'Base URL',
      'svc.base_url_info': 'The Base URL used by OpenAI, Ollama, Anthropic, or Translate clients',
      'svc.remote_warning': 'Turn on LarePass VPN on the device you use to connect remotely',
      'svc.endpoints': 'Supported endpoints',
      'svc.copy_model_title': 'Copy model name',
      'svc.copy_url_title': 'Copy Base URL',

      // aria
      'aria.collapse_model': 'Collapse model details',
      'aria.toggle_spec': 'Toggle model spec editor',
      'aria.param_view': 'Parameter view',
      'aria.spec_json': 'Model spec JSON',
      'aria.engine_args': 'Model-card engine arguments',

      // config
      'cfg.model_spec': 'Model spec',
      'cfg.model_spec_help': 'The live <code>model-spec.json</code> from <code>GET /api/model-spec</code> (read-only). Engine behavior changes take full effect after the next restart.',
      'cfg.params': 'Parameters',
      'cfg.form': 'Form',
      'cfg.raw': 'Raw',
      'cfg.context_length': 'Context length',
      'cfg.advanced': 'Advanced parameters',
      'cfg.args_help': 'Parsed view of model-card <code>engine_args</code>. Edit the raw string in the Raw tab; save updates the card and restarts the engine.',
      'cfg.raw_help': 'Model-card <code>engine_args</code> (SSOT). Save writes <code>PUT /api/model-spec</code> and signals the engine wrapper to relaunch.',
      'cfg.save_engine_args': 'Save and restart engine',
      'cfg.saving': 'Saving…',
      'cfg.saved': 'Saved',
      'cfg.saved_restarted': 'Saved; engine restart signaled',
      'cfg.saved_no_restart': 'Saved, but engine restart was not signaled (RUN_DIR unset)',
      'cfg.save_failed': 'Unable to save: {msg}',
      'cfg.reset': 'Reset',
      'cfg.save': 'Save and reload',
      'cfg.no_flags': 'No recognized arguments',
      'cfg.unknown_label': 'Unrecognized arguments, passed through as-is',
      'cfg.forwarded': 'Forwarded as-is: {args}',
      'cfg.experimental': 'Experimental',
      'cfg.engine_args_section': '{kind} engine arguments',
      'cfg.runtime_params_section': 'Runtime parameters',
      'cfg.engine_launch_args': '{kind} launch arguments',
      'cfg.arg.max_num_seqs': 'Maximum concurrent predictions',
      'cfg.arg.seed': 'Seed',
      'cfg.arg.kv_cache_dtype': 'KV cache quantization',
      'cfg.arg.cpu_offload_gb': 'CPU offload (GB)',
      'meta.context': 'Context {ctx}',

      // tip icons (? info / ! warn)
      'tip.context_length': 'The maximum number of tokens the model can use in one request, saved to model-spec.json',
      'tip.experimental': 'Experimental engine flag whose behavior may change across releases',
      'tip.base_url_olares': 'The public URL for apps inside Olares to use with OpenAI, Ollama, or Anthropic clients',
      'tip.base_url_translate': 'The public URL for Translate clients — host root, no /v1. Paths are /translate, /translate/batch, /translate/transcript, /languages, /detect',
      'tip.base_url_remote': 'The public URL for remote access, which may require LarePass VPN',
      'tip.base_url_mac': 'The local network address to use from Safari or other macOS apps',
      'tip.base_url_win': 'The local network address to use from Edge or other Windows apps',

      // arg tooltips
      'tip.max_model_len': 'Maximum context window the engine serves',
      'tip.gpu_memory_utilization': 'Fraction of GPU memory vLLM may reserve (0–1)',
      'tip.tensor_parallel_size': 'Number of GPUs to shard the model across',
      'tip.max_num_seqs': 'Maximum sequences decoded in parallel',
      'tip.cpu_offload_gb': 'GiB of model weights to offload to CPU RAM',
      'tip.kv_cache_dtype': 'KV cache data type, such as auto or fp8',
      'tip.seed': 'Deterministic sampling seed',

      // spec save status
      'spec.saving': 'Saving…',
      'spec.saved': 'Saved',
      'spec.unsaved': 'There are unsaved changes',
      'spec.invalid_json': 'Invalid JSON: {msg}',
      'spec.save_failed': 'Unable to save',
      'spec.save_failed_msg': 'Unable to save: {msg}',
      'spec.load_failed': 'Unable to load: {msg}',

      // gpu
      'gpu.engine_not_ready': 'Engine not ready',
      'gpu.waiting_help': 'Waiting for the model download to finish and the engine to come online (<code>phase=ready</code>). GPU residency appears once ready. Current phase: <code id="gpuWaitingPhase">unknown</code>.',
      'gpu.see_status': 'See the Status tab for download progress',
      'gpu.title': 'GPU residency',
      'gpu.mode': 'Mode',
      'gpu.vram': 'VRAM',
      'gpu.kv': 'KV cache used',
      'gpu.memutil': 'GPU memory utilization',
      'gpu.reported_by': 'Reported by',
      'gpu.sampled': 'Sampled at',
      'gpu.detect': 'Detect',
      'gpu.help': 'Source: <code>GET /api/diag/gpu</code>. Mode <code>partial</code>/<code>cpu_only</code> on a GPU host usually means the GPU is not mounted into the container.',
      'gpu.unknown': 'Unknown',
      'gpu.full': 'Full GPU',
      'gpu.split': 'Split',
      'gpu.cpu': 'CPU',
      'gpu.unavailable': 'Unavailable ({msg})',

      // mode labels (model-spec mode → display)
      'mode.chat': 'Chat',
      'mode.embedding': 'Embedding',
      'mode.completion': 'Completion',
      'mode.rerank': 'Rerank',
      'mode.moderation': 'Moderation',
      'mode.music_generation': 'Music',
    },

    zh: {
      'loading': '加载中…',
      'engine.label': '引擎',
      'tab.status': '状态',
      'tab.config': '配置',

      'phase.init': '初始化',
      'phase.download': '下载中',
      'phase.loading': '加载中',
      'phase.ready': '就绪',
      'phase.degraded': '降级',
      'phase.failed': '失败',

      'badge.initializing': '初始化中',
      'badge.downloading': '下载中',
      'badge.downloaded': '已下载',
      'badge.failed': '失败',
      'badge.degraded': '降级',

      'pl.ready': '就绪',
      'pl.loading': '正在将模型载入引擎…',
      'pl.degraded': '已降级',
      'pl.failed': '已失败',
      'pl.downloading_pct': '下载中… {pct}%',
      'pl.downloading': '下载中…',
      'pl.initializing': '初始化中…',
      'prog.of': '{done} / {total}',
      'prog.files': '{done} / {total} 个文件',
      'prog.left': '剩余 {eta}',
      'prog.estimating': '正在统计大小',
      'prog.downloading_file': '正在下载 {file}',
      'prog.file_waiting': '等待下载',
      'prog.file_downloading': '下载中',
      'prog.file_done': '已完成',
      'prog.retry': '{n} 次网络重试',
      'prog.retries': '{n} 次网络重试',

      'st.unknown': '未知',
      'st.ready': '就绪',
      'st.missing': '缺失',
      'st.running': '运行中',
      'st.waiting': '等待中',

      'act.redownload': '重新下载',
      'act.retry': '重试',
      'act.cancel': '取消',
      'act.pause': '暂停',
      'act.resume': '继续',
      'tip.pending_backend': '待后端支持',
      'meta.config_unavailable': '配置不可用',
      'confirm.proceed': '{label}？这将重新下载模型并重新运行设置流程。',

      'copy': '复制',
      'copied': '已复制',

      'diag.title': '诊断',
      'diag.last_error': '最近错误',
      'diag.retry_count': '初始化重试次数',
      'diag.transport_retries': '网络重试次数',
      'diag.help': '修复模型源或网络问题后重试，将重新下载模型并重新运行设置',

      'svc.title': '服务状态',
      'dlonly.title': '仅下载模式',
      'dlonly.desc': '未配置 ENGINE_KIND，仅下载模型文件，不启动推理引擎。',
      'svc.model': '模型',
      'svc.engine': '引擎',
      'svc.model_name': '模型名称',
      'svc.who': '连接来源',
      'svc.aud_olares': 'Olares 内应用',
      'svc.aud_lan': '本地网络设备',
      'svc.aud_remote': '远程',
      'svc.format': 'API 格式',
      'svc.base_url': 'Base URL',
      'svc.base_url_info': 'OpenAI、Ollama、Anthropic 或 Translate 客户端使用的 Base URL',
      'svc.remote_warning': '请在用于远程访问的设备上开启 LarePass 专用网络',
      'svc.endpoints': '支持的端点',
      'svc.copy_model_title': '复制模型名称',
      'svc.copy_url_title': '复制 Base URL',

      'aria.collapse_model': '折叠模型详情',
      'aria.toggle_spec': '切换模型规格编辑器',
      'aria.param_view': '参数视图',
      'aria.spec_json': '模型规格 JSON',
      'aria.engine_args': '模型卡引擎启动参数',

      'cfg.model_spec': '模型规格',
      'cfg.model_spec_help': '当前生效的 <code>model-spec.json</code>（只读），来自 <code>GET /api/model-spec</code>。引擎行为变更会在下次重启后完全生效。',
      'cfg.params': '参数',
      'cfg.form': '表单',
      'cfg.raw': '原始',
      'cfg.context_length': '上下文长度',
      'cfg.advanced': '高级参数',
      'cfg.args_help': '模型卡 <code>engine_args</code> 的解析视图。在 Raw 页编辑原始字符串；保存会更新模型卡并重启引擎。',
      'cfg.raw_help': '模型卡 <code>engine_args</code>（唯一事实源）。保存会 <code>PUT /api/model-spec</code> 并通知引擎 wrapper 重新拉起。',
      'cfg.save_engine_args': '保存并重启引擎',
      'cfg.saving': '保存中…',
      'cfg.saved': '已保存',
      'cfg.saved_restarted': '已保存；已发出引擎重启信号',
      'cfg.saved_no_restart': '已保存，但未发出引擎重启信号（RUN_DIR 为空）',
      'cfg.save_failed': '保存失败：{msg}',
      'cfg.reset': '重置',
      'cfg.save': '保存并重载',
      'cfg.no_flags': '无可识别参数',
      'cfg.unknown_label': '未识别参数，会按原样传入',
      'cfg.forwarded': '原样透传：{args}',
      'cfg.experimental': '实验性',
      'cfg.engine_args_section': '{kind} 引擎参数',
      'cfg.runtime_params_section': '运行时参数',
      'cfg.engine_launch_args': '{kind} 启动参数',
      'cfg.arg.max_num_seqs': '最大并发预测数',
      'cfg.arg.seed': '随机种子',
      'cfg.arg.kv_cache_dtype': 'KV 缓存量化',
      'cfg.arg.cpu_offload_gb': 'CPU 卸载 (GB)',
      'meta.context': '上下文 {ctx}',

      'tip.context_length': '单次请求中模型可使用的最大 token 数，保存至 model-spec.json',
      'tip.experimental': '实验性引擎参数，行为可能随版本变化',
      'tip.base_url_olares': 'Olares 内应用使用的公网地址，可用于 OpenAI、Ollama 或 Anthropic 客户端',
      'tip.base_url_translate': 'Translate 客户端使用的公网地址：主机根路径，不含 /v1。路径为 /translate、/translate/batch、/translate/transcript、/languages、/detect',
      'tip.base_url_remote': '远程访问使用的公网地址，可能需要开启 LarePass 专用网络',
      'tip.base_url_mac': 'Safari 或其他 macOS 应用可使用的本地网络地址',
      'tip.base_url_win': 'Edge 或其他 Windows 应用可使用的本地网络地址',

      'tip.max_model_len': '引擎对外提供的最大上下文窗口',
      'tip.gpu_memory_utilization': 'vLLM 可占用的 GPU 显存比例 (0–1)',
      'tip.tensor_parallel_size': '将模型切分到的 GPU 数量',
      'tip.max_num_seqs': '并行解码的最大序列数',
      'tip.cpu_offload_gb': '卸载到 CPU 内存的模型权重大小 (GiB)',
      'tip.kv_cache_dtype': 'KV 缓存的数据类型，如 auto 或 fp8',
      'tip.seed': '确定性采样的随机种子',

      'spec.saving': '保存中…',
      'spec.saved': '已保存',
      'spec.unsaved': '有未保存的更改',
      'spec.invalid_json': 'JSON 无效：{msg}',
      'spec.save_failed': '保存失败',
      'spec.save_failed_msg': '保存失败：{msg}',
      'spec.load_failed': '加载失败：{msg}',

      'gpu.engine_not_ready': '引擎未就绪',
      'gpu.waiting_help': '正在等待模型下载完成、引擎上线 (<code>phase=ready</code>)。就绪后会显示 GPU 驻留信息。当前 phase：<code id="gpuWaitingPhase">unknown</code>。',
      'gpu.see_status': '下载进度可在状态标签页查看',
      'gpu.title': 'GPU 驻留',
      'gpu.mode': '模式',
      'gpu.vram': '显存',
      'gpu.kv': 'KV 缓存占用',
      'gpu.memutil': 'GPU 显存利用率',
      'gpu.reported_by': '数据来源',
      'gpu.sampled': '采样时间',
      'gpu.detect': '检测',
      'gpu.help': '来源：<code>GET /api/diag/gpu</code>。在 GPU 主机上出现 <code>partial</code>/<code>cpu_only</code> 模式，通常说明容器未正确挂载 GPU。',
      'gpu.unknown': '未知',
      'gpu.full': '全量 GPU',
      'gpu.split': '部分卸载',
      'gpu.cpu': 'CPU',
      'gpu.unavailable': '不可用 ({msg})',

      'mode.chat': '对话',
      'mode.embedding': '嵌入',
      'mode.completion': '补全',
      'mode.rerank': '重排序',
      'mode.moderation': '审核',
      'mode.music_generation': '音乐生成',
    },
  };

  const lang = normalize(rawLang());

  function interpolate(str, params) {
    if (!params) return str;
    return str.replace(/\{(\w+)\}/g, (m, k) => (params[k] != null ? params[k] : m));
  }

  // t(key, params) → localised string. Falls back to English, then the
  // raw key, so a missing translation degrades visibly but harmlessly.
  function t(key, params) {
    const table = DICT[lang] || DICT.en;
    const s = (table[key] != null) ? table[key] : (DICT.en[key] != null ? DICT.en[key] : key);
    return interpolate(s, params);
  }

  // wireTip attaches a visible hover/focus bubble to a ? / ! glyph.
  function wireTip(el, text) {
    if (!el || !text) return;
    el.setAttribute('aria-label', text);
    if (!el.hasAttribute('tabindex')) el.tabIndex = 0;
    let bubble = el.querySelector('.tip-bubble');
    if (!bubble) {
      bubble = document.createElement('span');
      bubble.className = 'tip-bubble';
      bubble.setAttribute('aria-hidden', 'true');
      el.appendChild(bubble);
    }
    bubble.textContent = text;
  }

  // tipIcon builds a hoverable ? (info) or ! (warn) glyph with a localised
  // tooltip bubble (native title is unreliable in embedded webviews).
  function tipIcon(kind, tipKey) {
    const warn = kind === 'warn' || kind === '!';
    const span = document.createElement('span');
    span.className = warn ? 'warn-dot' : 'info-dot';
    span.textContent = warn ? '!' : '?';
    span.setAttribute('role', 'img');
    wireTip(span, t(tipKey));
    return span;
  }

  // apply localises every tagged element under root (default document).
  //   data-i18n          → textContent
  //   data-i18n-html     → innerHTML (strings that embed <code> etc.)
  //   data-i18n-title    → title attribute
  //   data-i18n-aria-label / data-i18n-placeholder → matching attribute
  function apply(root) {
    const scope = root || document;
    scope.querySelectorAll('[data-i18n]').forEach((el) => {
      el.textContent = t(el.getAttribute('data-i18n'));
    });
    scope.querySelectorAll('[data-i18n-html]').forEach((el) => {
      el.innerHTML = t(el.getAttribute('data-i18n-html'));
    });
    const attrs = ['title', 'aria-label', 'placeholder'];
    attrs.forEach((a) => {
      scope.querySelectorAll('[data-i18n-' + a + ']').forEach((el) => {
        const val = t(el.getAttribute('data-i18n-' + a));
        el.setAttribute(a, val);
        // Static ? / ! glyphs: show a hover bubble from data-i18n-title.
        if (a === 'title' && (el.classList.contains('info-dot') || el.classList.contains('warn-dot'))) {
          wireTip(el, val);
        }
      });
    });
  }

  // modeLabel localises a model-spec `mode` (chat / embedding / …) for
  // the header + capability pills, falling back to a capitalised raw
  // value when the mode has no dedicated translation.
  function modeLabel(mode) {
    if (!mode) return '';
    const key = 'mode.' + mode;
    const v = t(key);
    return v === key ? (mode.charAt(0).toUpperCase() + mode.slice(1)) : v;
  }

  // engineName maps an engine kind to its brand-cased display name.
  // Unknown kinds fall back to first-letter capitalisation.
  const ENGINE_NAMES = {
    llamacpp: 'llama.cpp',
    'llama.cpp': 'llama.cpp',
    llama_cpp: 'llama.cpp',
    ollama: 'Ollama',
    vllm: 'vLLM',
    sglang: 'SGLang',
  };
  function engineName(kind) {
    if (!kind) return '';
    const key = String(kind).toLowerCase();
    return ENGINE_NAMES[key] || (key.charAt(0).toUpperCase() + key.slice(1));
  }

  LLM.lang = lang;
  LLM.t = t;
  LLM.tipIcon = tipIcon;
  LLM.modeLabel = modeLabel;
  LLM.engineName = engineName;
  LLM.i18n = { lang: lang, t: t, apply: apply, modeLabel: modeLabel };

  // Reflect the language on <html> for CSS / a11y, and localise the
  // static markup. Scripts live at the end of <body>, so the DOM is
  // already parsed by the time this runs.
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en-US';
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => apply());
  } else {
    apply();
  }
})();
