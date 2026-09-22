/*
 * gpu.js — GPU residency card (a block inside the Config tab).
 *
 * Lazy-init flow:
 *   1. config.js calls LLM.initGpu() the first time the Config tab is
 *      activated in a non-download-only run
 *   2. GET /api/diag/gpu fires once phase reaches ready, and again on
 *      every "Detect" click
 *
 * Read-only: /api/diag/gpu has no side effects on the engine.
 */
(function () {
  'use strict';
  const LLM = (window.LLM = window.LLM || {});

  // foldResidency derives a single human label from the report. The
  // wire no longer carries an inferred `mode` (dropped in v1.1.0), so
  // the UI folds it locally per the /api/diag/gpu contract:
  // vram_bytes >= model_bytes -> full, vram_bytes > 0 -> split,
  // else CPU; when VRAM bytes are absent (vLLM/SGLang/llama.cpp) fall
  // back to cpu_offload_gb (0/absent -> full, >0 -> split).
  function foldResidency(g) {
    if (!g) return { label: LLM.t('gpu.unknown'), cls: 'muted' };
    const v = g.vram_bytes, m = g.model_bytes;
    if (v != null && m != null) {
      if (v >= m) return { label: LLM.t('gpu.full'), cls: 'ok' };
      if (v > 0) return { label: LLM.t('gpu.split'), cls: 'warn' };
      return { label: LLM.t('gpu.cpu'), cls: 'err' };
    }
    if (g.source === 'unavailable') return { label: LLM.t('gpu.unknown'), cls: 'muted' };
    if (g.cpu_offload_gb != null && g.cpu_offload_gb > 0) {
      return { label: LLM.t('gpu.split'), cls: 'warn' };
    }
    return { label: LLM.t('gpu.full'), cls: 'ok' };
  }

  // setKv fills a kv-grid value cell and hides the whole cell (label +
  // value) when the underlying field is absent, so only present fields
  // show. Pass null/'' to hide.
  function setKv(valId, text, cls) {
    const el = LLM.el(valId);
    if (!el) return;
    const cell = el.closest('.kv-grid__cell');
    const has = text != null && text !== '';
    if (cell) cell.hidden = !has;
    el.textContent = has ? text : '—';
    if (valId === 'gpuMode') el.className = 'kv-grid__v ' + (cls || '');
  }

  function renderGPU(r) {
    const g = r.gpu || {};

    // Mode (residency) is only meaningful when we have GPU data to fold;
    // hide it when the report is empty or explicitly unavailable.
    const haveGpu = r.gpu && g.source !== 'unavailable';
    const fold = haveGpu ? foldResidency(r.gpu) : null;
    setKv('gpuMode', fold ? fold.label : null, fold ? fold.cls : '');

    // VRAM line: "<vram>/<model>" when both known, just vram when not,
    // cpu_offload fallback, else hidden.
    const v = g.vram_bytes;
    const m = g.model_bytes;
    let vram = null;
    if (v != null && m != null) vram = LLM.fmt.bytes(v) + ' / ' + LLM.fmt.bytes(m);
    else if (v != null) vram = LLM.fmt.bytes(v);
    else if (g.cpu_offload_gb != null) vram = 'cpu_offload_gb=' + g.cpu_offload_gb;
    setKv('gpuVram', vram);

    setKv('gpuKvCache', g.kv_cache_usage_perc != null
      ? (g.kv_cache_usage_perc * 100).toFixed(1) + '%' : null);
    setKv('gpuMemUtil', g.gpu_memory_utilization != null
      ? (g.gpu_memory_utilization * 100).toFixed(1) + '%' : null);
    setKv('gpuSource', g.source || null);
    setKv('gpuSampled', r.generated_at ? LLM.fmt.time(r.generated_at) : null);

    const warn = LLM.el('gpuWarnings');
    warn.innerHTML = '';
    (r.warnings || []).forEach((w) => {
      const li = document.createElement('li');
      li.textContent = w;
      warn.appendChild(li);
    });
  }

  // ─── readiness gating ─────────────────────────────────────────────
  // During download the engine is alive but no model is loaded, so
  // /api/diag/gpu would fold to a "CPU" label, which is misleading.
  // Instead we hide #gpuBody, show #gpuWaitingBanner, and skip the boot
  // fetch until phase first reaches ready.
  //
  // dataLoaded is sticky-true after the first /api/diag/gpu completes
  // per ready-transition. A phase regression (ready -> download/loading
  // after a Force /api/retry) flips it back to false so the next
  // ready-transition re-fetches.
  let latestPhase = '';
  let dataLoaded = false;

  function resetGpuData() {
    ['gpuMode', 'gpuVram', 'gpuKvCache', 'gpuMemUtil', 'gpuSource', 'gpuSampled'].forEach((id) => {
      const e = LLM.el(id);
      if (!e) return;
      e.textContent = '—';
      // renderGPU may have hidden a cell for a missing field; restore the
      // placeholder so the reset state shows the full grid.
      const cell = e.closest && e.closest('.kv-grid__cell');
      if (cell) cell.hidden = false;
    });
    const mode = LLM.el('gpuMode');
    if (mode) mode.className = 'kv-grid__v muted';
    const warn = LLM.el('gpuWarnings');
    if (warn) warn.innerHTML = '';
  }

  // refreshGPU pulls a fresh /api/diag/gpu snapshot. Used both by the
  // initial ready-transition load and by the manual "Detect" button.
  function refreshGPU() {
    return fetch('/api/diag/gpu').then((r) => {
      if (!r.ok) throw new Error('status ' + r.status);
      return r.json();
    }).then(renderGPU).catch((e) => {
      const mode = LLM.el('gpuMode');
      if (mode) { mode.textContent = LLM.t('gpu.unavailable', { msg: e.message }); mode.className = 'kv-grid__v err'; }
    });
  }

  function applyReadyState(phase) {
    latestPhase = phase || '';
    const ready = latestPhase === 'ready';

    const banner = LLM.el('gpuWaitingBanner');
    const body = LLM.el('gpuBody');
    const phaseEl = LLM.el('gpuWaitingPhase');
    if (banner && body) {
      banner.hidden = ready;
      body.hidden = !ready;
    }
    if (phaseEl) phaseEl.textContent = latestPhase || LLM.t('gpu.unknown');

    if (!ready) {
      // Phase regression also lands here (ready -> download/loading after
      // a Force /api/retry). Drop dataLoaded so the next ready
      // transition triggers a fresh fetch instead of leaving a
      // stale GPU snapshot up.
      dataLoaded = false;
      resetGpuData();
      return;
    }
    if (!dataLoaded) {
      dataLoaded = true;
      refreshGPU();
    }
  }

  // The GPU card is a block inside the Config tab. config.js calls
  // LLM.initGpu() once, the first time that tab is activated in a
  // non-download-only run. Guard against a double-call (config.js
  // already guards, but keep this idempotent).
  let inited = false;
  LLM.initGpu = function () {
    if (inited) return;
    inited = true;

    // "Detect" re-samples GPU residency on demand (read-only, safe to
    // click any time the engine is up).
    const detect = LLM.el('gpuDetectBtn');
    if (detect) detect.addEventListener('click', () => {
      detect.disabled = true;
      refreshGPU().finally(() => { detect.disabled = false; });
    });

    // Subscribe to dashboard.js's phase broadcast. dashboard fires
    // 'llm:phase' both on first /api/progress fetch and on every
    // SSE progress/phase event, so this single listener covers
    // first-paint and live updates. applyReadyState owns the data
    // fetch: on the first ready transition it calls refreshGPU,
    // and on regression it blanks the card and re-shows the banner.
    window.addEventListener('llm:phase', (e) => {
      applyReadyState(e && e.detail && e.detail.phase);
    });

    // Belt-and-braces fallback: if the user opens the Config tab
    // *before* dashboard.js's first 'llm:phase' broadcast lands (race on
    // tab-switch vs SSE first frame), pull the phase ourselves once at
    // init. refreshGPU fires from inside applyReadyState the moment we
    // observe ready, so this single call also kicks the initial fetch.
    fetch('/api/progress').then((r) => r.ok ? r.json() : null).then((p) => {
      if (p && typeof p.phase === 'string') applyReadyState(p.phase);
      else applyReadyState('');
    }).catch(() => { applyReadyState(''); });
  };
})();
