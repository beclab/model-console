/*
 * config.js — Config tab.
 *
 * Drives the second dashboard tab (LLM.initTab.config):
 *   * Model capability card — mode / context / supports_* pills
 *   * Parameters — Form (Context Length + Advanced) / Raw (model-spec JSON
 *     editor); Save & Reload PUTs /api/model-spec, Reset reloads it
 *   * Engine args  — model-card engine_args (editable Raw + parsed Advanced)
 *   * GPU residency — /api/diag/gpu snapshot + Detect (gpu.js)
 *
 * The framework fires one init per tab, so this module kicks the GPU
 * card (LLM.initGpu, gpu.js) on first activation. In download-only
 * mode (empty ENGINE_KIND) the GPU block is hidden by dashboard.js, so
 * we skip it here — only the spec editor / form is wired.
 */
(function () {
  'use strict';
  const LLM = (window.LLM = window.LLM || {});
  LLM.initTab = LLM.initTab || {};

  // ─── model capability card ────────────────────────────────────────
  // Context size renders in "k" units to match the Figma ("Context 32k").
  function fmtCtx(n) {
    if (!n || n <= 0) return null;
    if (n >= 1024) return Math.round(n / 1024) + 'k';
    return String(n);
  }

  function renderModelCard(spec) {
    const name = LLM.el('cfgModelName');
    const meta = LLM.el('cfgModelMeta');
    const caps = LLM.el('capsList');
    if (name) name.textContent = (spec && spec.name) || (LLM.config && LLM.config.model_name) || '—';

    if (meta) {
      const bits = [];
      if (spec && spec.mode) bits.push(LLM.modeLabel(spec.mode));
      const ctx = fmtCtx(spec && spec.context_size);
      if (ctx) bits.push(LLM.t('meta.context', { ctx: ctx }));
      meta.textContent = bits.join(' · ');
    }

    if (caps) {
      caps.innerHTML = '';
      const supports = (spec && spec.supports) || {};
      const enabled = Object.keys(supports)
        .filter((k) => supports[k] === true)
        .map((k) => k.replace(/^supports_/, ''))
        .sort();
      enabled.forEach((k) => {
        const span = document.createElement('span');
        span.className = 'pill cap';
        span.textContent = k;
        caps.appendChild(span);
      });
    }
  }

  // ─── Context Length form field ────────────────────────────────────
  // Hidden entirely when the spec carries no positive context_size.
  function renderForm(spec) {
    const ctx = LLM.el('ctxLen');
    const field = LLM.el('ctxField');
    const size = spec && spec.context_size;
    const has = size != null && size > 0;
    if (ctx) ctx.value = has ? size : '';
    if (field) field.hidden = !has;
  }

  // ─── Form / Raw toggle ────────────────────────────────────────────
  function bindViewToggle() {
    const seg = document.querySelector('.seg');
    if (!seg) return;
    seg.querySelectorAll('.seg__btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const view = btn.dataset.view;
        seg.querySelectorAll('.seg__btn').forEach((b) => b.classList.toggle('active', b === btn));
        const form = LLM.el('formView');
        const raw = LLM.el('rawView');
        if (form) form.classList.toggle('active', view === 'form');
        if (raw) raw.classList.toggle('active', view === 'raw');
      });
    });
  }

  // ─── engine args (Advanced parsed table + Raw editor) ─────────────
  // Presentation metadata for recognised normalised keys (config's
  // engine_args_known.go). Raw string is model-card engine_args (SSOT).
  const ARG_META = {
    max_model_len:          { flag: '--max-model-len', tip: 'tip.max_model_len' },
    gpu_memory_utilization: { flag: '--gpu-memory-utilization', tip: 'tip.gpu_memory_utilization' },
    tensor_parallel_size:   { flag: '--tensor-parallel-size', tip: 'tip.tensor_parallel_size' },
    pipeline_parallel_size: { flag: '--pipeline-parallel-size' },
    max_num_seqs:           { flag: '--max-num-seqs', group: 'runtime', label: 'cfg.arg.max_num_seqs', exp: true, tip: 'tip.max_num_seqs' },
    max_num_batched_tokens: { flag: '--max-num-batched-tokens' },
    cpu_offload_gb:         { flag: '--cpu-offload-gb', group: 'runtime', label: 'cfg.arg.cpu_offload_gb', tip: 'tip.cpu_offload_gb' },
    swap_space:             { flag: '--swap-space' },
    block_size:             { flag: '--block-size' },
    kv_cache_dtype:         { flag: '--kv-cache-dtype', group: 'runtime', label: 'cfg.arg.kv_cache_dtype', exp: true, tip: 'tip.kv_cache_dtype' },
    seed:                   { flag: '--seed', group: 'runtime', label: 'cfg.arg.seed', tip: 'tip.seed' },
    num_ctx:                { flag: 'num_ctx' },
    num_parallel:           { flag: 'num_parallel' },
    ctx_size:               { flag: '--ctx-size' },
    n_gpu_layers:           { flag: '--n-gpu-layers' },
  };

  function argLabelText(meta, key) {
    if (meta.label) return LLM.t(meta.label);
    return meta.flag || key;
  }

  function argRow(key, value, meta) {
    const row = document.createElement('div');
    row.className = 'param-row';

    const label = document.createElement('span');
    label.className = 'param-row__label' + (meta.label ? '' : ' param-row__label--mono');
    const text = document.createElement('span');
    text.textContent = argLabelText(meta, key);
    label.appendChild(text);
    if (meta.tip) label.appendChild(LLM.tipIcon('info', meta.tip));
    if (meta.exp) {
      const exp = document.createElement('span');
      exp.className = 'pill pill--exp';
      exp.textContent = LLM.t('cfg.experimental');
      label.appendChild(exp);
    }

    const val = document.createElement('span');
    val.className = 'param-row__value';
    val.textContent = value === '' ? '—' : value;

    row.append(label, val);
    return row;
  }

  function appendArgGroup(host, title, keys, known) {
    if (!keys.length) return;
    const sec = document.createElement('div');
    sec.className = 'param-form__section';
    sec.textContent = title;
    host.appendChild(sec);
    keys.forEach((k) => host.appendChild(argRow(k, String(known[k]), ARG_META[k] || {})));
  }

  function renderArgsForm(cfg) {
    const host = LLM.el('argsForm');
    const unknownEl = LLM.el('argsUnknown');
    const raw = LLM.el('argsRaw');
    const title = LLM.el('engineArgsTitle');

    const kind = (cfg && cfg.engine && cfg.engine.kind) || '';
    const args = (cfg && cfg.engine && cfg.engine.args) || {};

    if (host) {
      host.innerHTML = '';
      const known = args.known || {};
      const keys = Object.keys(known).sort();
      if (keys.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'muted';
        empty.textContent = LLM.t('cfg.no_flags');
        host.appendChild(empty);
      } else {
        const engineKeys = keys.filter((k) => (ARG_META[k] || {}).group !== 'runtime');
        const runtimeKeys = keys.filter((k) => (ARG_META[k] || {}).group === 'runtime');
        appendArgGroup(
          host,
          LLM.t('cfg.engine_args_section', { kind: kind ? LLM.engineName(kind) : LLM.t('svc.engine') }),
          engineKeys,
          known,
        );
        appendArgGroup(host, LLM.t('cfg.runtime_params_section'), runtimeKeys, known);
      }
    }

    if (unknownEl) {
      const extras = args.unknown || [];
      unknownEl.innerHTML = '';
      if (extras.length) {
        const label = document.createElement('div');
        label.className = 'args-unknown__label muted';
        label.textContent = LLM.t('cfg.unknown_label');
        const body = document.createElement('div');
        body.className = 'args-unknown__body mono';
        body.textContent = extras.join(' ');
        unknownEl.append(label, body);
      }
    }

    if (title) title.textContent = LLM.t('cfg.engine_launch_args', { kind: kind ? LLM.engineName(kind) : LLM.t('svc.engine') });
    if (raw && document.activeElement !== raw) {
      const rawStr = (args.raw && args.raw.length) ? args.raw : '';
      raw.value = rawStr;
    }
  }

  function setArgsStatus(msg, kind) {
    const el = LLM.el('argsStatus');
    if (!el) return;
    el.textContent = msg;
    el.className = 'spec-status ' + (kind || 'muted');
  }

  function saveEngineArgs() {
    const raw = LLM.el('argsRaw');
    const next = raw ? String(raw.value || '').trim() : '';
    setArgsStatus(LLM.t('cfg.saving'), 'muted');
    return fetch('/api/model-spec')
      .then((r) => {
        if (!r.ok) throw new Error('GET /api/model-spec ' + r.status);
        return r.json();
      })
      .then((spec) => {
        spec.engine_args = next;
        return fetch('/api/model-spec', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(spec),
        });
      })
      .then((r) => r.text().then((body) => ({ r, body })))
      .then(({ r, body }) => {
        // 503 run_dir_unset: card + handoff path may already be persisted;
        // only the restart signal failed. Treat as partial success.
        const runDirUnset = r.status === 503 && /run_dir_unset/.test(body || '');
        if (!r.ok && !runDirUnset) throw new Error(body || ('PUT ' + r.status));
        if (runDirUnset) {
          setArgsStatus(LLM.t('cfg.saved_no_restart'), 'warn');
        } else {
          const restarted = r.headers.get('X-Engine-Restarted') === 'true';
          setArgsStatus(
            restarted ? LLM.t('cfg.saved_restarted') : LLM.t('cfg.saved'),
            'ok',
          );
        }
        return fetch('/api/config')
          .then((cr) => cr.json())
          .then((cfg) => {
            LLM.config = cfg;
            renderArgsForm(cfg);
          })
          .then(() => loadSpec());
      })
      .catch((e) => setArgsStatus(LLM.t('cfg.save_failed', { msg: e.message }), 'err'));
  }

  // ─── spec editor (Raw) + form save ────────────────────────────────
  function setSpecStatus(msg, kind) {
    const el = LLM.el('specStatus');
    if (!el) return;
    el.textContent = msg;
    el.className = 'spec-status ' + (kind || 'muted');
  }

  function applySpec(spec) {
    const ed = LLM.el('specEditor');
    if (ed) ed.value = JSON.stringify(spec, null, 2);
    renderModelCard(spec);
    renderForm(spec);
  }

  function loadSpec() {
    return fetch('/api/model-spec')
      .then((r) => r.json())
      .then((spec) => { applySpec(spec); return spec; })
      .catch((e) => setSpecStatus(LLM.t('spec.load_failed', { msg: e.message }), 'err'));
  }

  let gpuStarted = false;
  function initEngineBits() {
    if (LLM.downloadOnly) return;
    renderArgsForm(LLM.config);
    if (!gpuStarted && LLM.initGpu) {
      gpuStarted = true;
      LLM.initGpu();
    }
  }

  // Model-card collapse reveals the full Model Spec editor (default folded).
  function bindModelCollapse() {
    const collapse = LLM.el('cfgModelCollapse');
    const body = LLM.el('cfgModelBody');
    if (!collapse || !body) return;
    collapse.addEventListener('click', () => {
      const open = collapse.getAttribute('aria-expanded') === 'true';
      collapse.setAttribute('aria-expanded', open ? 'false' : 'true');
      body.hidden = open;
    });
  }

  LLM.initTab.config = function () {
    bindViewToggle();
    bindModelCollapse();

    const saveBtn = LLM.el('argsSaveBtn');
    if (saveBtn) saveBtn.addEventListener('click', () => { saveEngineArgs(); });

    // Seed the capability card immediately from the redacted config's
    // embedded model_spec so the tab isn't blank before /api/model-spec
    // resolves; loadSpec() then refines it with the live, editable spec.
    if (LLM.specFromConfig) {
      renderModelCard(LLM.specFromConfig);
      renderForm(LLM.specFromConfig);
    }
    loadSpec();

    if (LLM.config) {
      initEngineBits();
    } else {
      LLM.onConfig = function () { initEngineBits(); };
    }
  };
})();
