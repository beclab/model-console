/*
 * dashboard.js — main entrypoint for the llm-init UI.
 *
 * Responsible for:
 *   * shared formatters / DOM helpers exposed on `window.LLM`
 *   * tab switching (declarative via [data-tab]) — two tabs: status / config
 *   * Status tab — download progress, error + Retry, engine LEDs + API card
 *   * boot-time fetches: /api/build-info, /api/config, /healthz, /api/progress
 *   * 1 s /api/progress polling (SSE was retired in v1.1.0; the poll
 *     interval matches the prior SSE tick budget without holding a
 *     long-lived connection per tab).
 *
 * The Config tab is driven by config.js (LLM.initTab.config), which also
 * kicks the GPU residency card (gpu.js). dashboard.js drives the Status
 * tab directly — it has no lazy init.
 *
 * download-only mode (empty ENGINE_KIND): refreshConfig sets
 * document.body.dataset.downloadOnly and hides the engine status card,
 * the engine-args card, and the GPU block. Only the download progress,
 * error/Retry, and model-spec editor remain.
 */
(function () {
  'use strict';

  const el = (id) => document.getElementById(id);

  // ─── formatters ───────────────────────────────────────────────────
  // Decimal (1000-based) units so download totals match the sizes shown on
  // Hugging Face / vendor model cards (e.g. 3.19 GB + 0.92 GB ≈ 4.1 GB),
  // rather than 1024-based GiB which would render the same bytes as "3.8 GB".
  const BYTE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB'];
  function byteScale(n) {
    let i = 0;
    let v = Math.max(0, +n || 0);
    while (v >= 1000 && i < BYTE_UNITS.length - 1) {
      v /= 1000;
      i++;
    }
    return { v: v, i: i };
  }
  function fmtBytes(n) {
    if (!n || n < 0) return '0';
    const s = byteScale(n);
    if (s.i === 0) return Math.round(n) + ' B';
    return s.v.toFixed(s.v >= 100 ? 0 : 1) + ' ' + BYTE_UNITS[s.i];
  }
  function fmtBytesInUnit(n, unitIndex, digits) {
    const raw = Math.max(0, +n || 0);
    if (unitIndex === 0) return Math.round(raw) + ' B';
    return (raw / Math.pow(1000, unitIndex)).toFixed(digits) + ' ' + BYTE_UNITS[unitIndex];
  }
  // Progress headline: while the pass is still going, raise decimals
  // (same unit as the total) until the two sides look different.
  // 1 decimal at GB hides ~50 MB, so 4.68 / 4.70 collapses to 4.7 / 4.7.
  function fmtProgressPair(done, total, incomplete) {
    if (!(total > 0) || !incomplete) {
      return { done: fmtBytes(done), total: fmtBytes(total) };
    }
    const unit = byteScale(total).i;
    const start = unit >= 2 ? 2 : (unit === 0 ? 0 : 1);
    const end = unit === 0 ? 0 : 3;
    let ds = '';
    let ts = '';
    for (let d = start; d <= end; d++) {
      ds = fmtBytesInUnit(done, unit, d);
      ts = fmtBytesInUnit(total, unit, d);
      if (ds !== ts) return { done: ds, total: ts };
    }
    return { done: ds, total: ts };
  }
  function fmtSpeed(bps) { return bps > 0 ? fmtBytes(bps) + '/s' : '—'; }
  // HF tqdm labels are usually a basename; fall back to the last path
  // segment if a snapshot path slips through.
  function displayFileName(path) {
    const s = String(path || '');
    if (!s) return '';
    const slash = Math.max(s.lastIndexOf('/'), s.lastIndexOf('\\'));
    return slash >= 0 ? s.slice(slash + 1) : s;
  }
  function fmtETA(s) {
    if (!s || s <= 0) return '—';
    if (s < 60) return s + 's';
    if (s < 3600) return Math.floor(s / 60) + 'm ' + (s % 60) + 's';
    return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
  }
  // Reject Go's JSON zero time (0001-01-01…), Unix epoch mistakes, and
  // other bogus strings that are truthy in JS but render as "1/1/1".
  function isValidIso(iso) {
    if (!iso) return false;
    try {
      const d = new Date(iso);
      const t = d.getTime();
      if (isNaN(t)) return false;
      if (d.getFullYear() < 2000) return false;
      return true;
    } catch (_) {
      return false;
    }
  }
  function fmtTime(iso) {
    if (!isValidIso(iso)) return '—';
    try {
      return new Date(iso).toLocaleString();
    } catch (_) { return '—'; }
  }
  function fmtAgo(iso) {
    if (!isValidIso(iso)) return '';
    const d = new Date(iso).getTime();
    const dt = (Date.now() - d) / 1000;
    if (dt < 60) return Math.round(dt) + 's ago';
    if (dt < 3600) return Math.round(dt / 60) + 'm ago';
    if (dt < 86400) return Math.round(dt / 3600) + 'h ago';
    return Math.round(dt / 86400) + 'd ago';
  }
  function fmtMs(ms) {
    if (ms == null) return '—';
    if (ms < 1000) return ms.toFixed(0) + ' ms';
    return (ms / 1000).toFixed(2) + ' s';
  }
  function fmtNum(n) {
    if (n == null) return '—';
    if (n >= 1e6) return (n / 1e6).toFixed(2) + ' M';
    if (n >= 1e3) return (n / 1e3).toFixed(2) + ' k';
    if (typeof n === 'number' && !Number.isInteger(n)) return n.toFixed(2);
    return String(n);
  }

  // Expose helpers so the per-tab modules can reuse them without a
  // bundler. Single global LLM object keeps the surface small.
  const LLM = window.LLM = (window.LLM || {});
  LLM.fmt = { bytes: fmtBytes, speed: fmtSpeed, eta: fmtETA, time: fmtTime, ago: fmtAgo, ms: fmtMs, num: fmtNum };
  LLM.el = el;

  // The Event-log panel was dropped in the 2-tab redesign; keep a thin
  // console shim so gpu.js / config.js calls to LLM.log() stay valid
  // without a #log DOM node.
  LLM.log = function (text, kind) {
    if (kind === 'error') console.error('[llm]', text);
    else console.log('[llm]', text);
  };

  // ─── tabs ─────────────────────────────────────────────────────────
  const tabButtons = document.querySelectorAll('nav.tabs button[data-tab]');
  const tabPanels = document.querySelectorAll('.tab-panel');
  const tabActivated = {};

  function activateTab(name) {
    tabButtons.forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
    tabPanels.forEach((p) => p.classList.toggle('active', p.id === 'tab-' + name));
    // Lazy init: each tab's init() runs at most once.
    if (!tabActivated[name]) {
      tabActivated[name] = true;
      const init = LLM.initTab && LLM.initTab[name];
      if (init) {
        try { init(); } catch (e) { console.error('initTab', name, e); }
      }
    }
    if (history.replaceState) {
      history.replaceState(null, '', '#' + name);
    }
  }
  tabButtons.forEach((b) => b.addEventListener('click', () => activateTab(b.dataset.tab)));

  // Wiring assertion: every data-tab button must have a matching
  // LLM.initTab.<name> registered by a per-tab module. Catches the bug
  // class where a module registers under a different name than the
  // button's data-tab value. "status" is exempt — dashboard.js drives
  // it directly with no lazy init.
  tabButtons.forEach((b) => {
    const name = b.dataset.tab;
    if (name === 'status') return;
    if (!(LLM.initTab && LLM.initTab[name])) {
      console.error('initTab.' + name + ' missing — tab will be inert; check the matching per-tab module registers under this exact key');
      b.classList.add('broken');
      b.title = 'wiring bug: LLM.initTab.' + name + ' is undefined';
    }
  });

  // Activate by URL hash on load (e.g. /#config) so deep-linking to a
  // tab survives reloads. Falls back silently if the hash is unknown.
  const hash = (location.hash || '').replace(/^#/, '');
  if (hash && document.getElementById('tab-' + hash)) {
    activateTab(hash);
  }

  // ─── status rendering ─────────────────────────────────────────────
  //
  // The badge shows the documented state-machine phase
  // (init/download/loading/ready/degraded/failed). The loading phase
  // already covers "downloaded, waiting for the engine to come alive",
  // so the normal cold-start window renders as `loading`.
  //
  // It does not fold engine health in, and must not: `loading` and
  // `ready` both render as "downloaded", so the badge never claims the
  // engine is up. Whether requests can be served is the Service-status
  // pills and the connection panel, both of which read /healthz.
  // The model-card badge expresses *download* status, not engine health
  // (per the Figma: it reads "Downloaded" green even once the engine is
  // running — engine liveness lives in the Service-status pills).
  // Completion is phase-driven: while phase is download, bytes hitting
  // the current total is not "done" (the next HF file or source can
  // still grow the denominator). loading/ready are the only green states.
  function renderPhaseBadge() {
    const b = el('phase');
    if (!b) return;
    const phase = svc.phase || 'init';
    let cls, key;
    if (phase === 'failed') { cls = 'failed'; key = 'badge.failed'; }
    else if (phase === 'degraded') { cls = 'degraded'; key = 'badge.degraded'; }
    else if (phase === 'loading' || phase === 'ready') { cls = 'ready'; key = 'badge.downloaded'; }
    else if (phase === 'download') { cls = 'download'; key = 'badge.downloading'; }
    else { cls = 'init'; key = 'badge.initializing'; }
    b.className = 'badge ' + cls;
    b.textContent = LLM.t(key);
    b.title = '';
  }

  // Human-friendly progress headline for the Status model card.
  function phaseLabel(phase, total, done) {
    switch (phase) {
      case 'ready': return LLM.t('pl.ready');
      case 'loading': return LLM.t('pl.loading');
      case 'degraded': return LLM.t('pl.degraded');
      case 'failed': return LLM.t('pl.failed');
      case 'download':
        return total > 0 ? LLM.t('pl.downloading_pct', { pct: Math.floor((done / total) * 100) }) : LLM.t('pl.downloading');
      default: return LLM.t('pl.initializing');
    }
  }

  function setLed(id, val) {
    const e = el(id);
    if (!e) return;
    e.className = 'led ' + (val === true ? 'ok' : val === false ? 'err' : 'unknown');
  }

  // setStatePill renders a Figma "Service status" pill (MISSING / WAITING /
  // …) with a colour class. `cls` is one of 'ok' | 'err' | 'warn' | ''.
  function setStatePill(id, text, cls) {
    const e = el(id);
    if (!e) return;
    e.textContent = text;
    e.className = 'pill pill--state' + (cls ? ' ' + cls : '');
  }

  // "Model exists" is kind-aware. Ollama is daemon-backed: the daemon can
  // be alive while the requested tag is not pulled, so we trust the
  // /healthz model_exists probe. File-backed engines (vLLM / llama.cpp /
  // SGLang) own no model registry — the model "exists" the moment its
  // files are on disk, independent of whether the engine process is up.
  // For those we treat a completed download as exists=true (a live engine
  // that already lists it also counts).
  function renderModelExists() {
    const kind = (LLM.engineKind || '').toLowerCase();
    let val;
    if (kind === 'ollama') {
      val = svc.modelExistsHealth; // true | false | null
    } else {
      val = svc.downloadComplete || svc.modelExistsHealth === true;
    }
    setLed('ledModel', val);
    setStatePill('modelExists',
      val == null ? LLM.t('st.unknown') : (val ? LLM.t('st.ready') : LLM.t('st.missing')),
      val === true ? 'ok' : val === false ? 'err' : '');
    svc.modelReady = val === true;
    // The model-name inset only appears once the model is present, as in
    // the Figma (the MISSING state shows just the LED + badge).
    const detail = el('modelDetail');
    if (detail) detail.hidden = !svc.modelReady;
  }

  // ─── status > Service status (connection panel) ───────────────────
  //
  // Lives inside #engineCard (hidden in download-only mode). Driven by
  // four feeds: Config (PublicURL + ModelName), Progress (phase), Health
  // (engine_alive), and Endpoints (/api/endpoints — which API formats are
  // wired on this build + their endpoint catalog). The audience + format
  // radios pick which base URL + endpoint set to surface; the base URL
  // link/copy unlock on `/healthz`'s `ready`, so an operator never copies
  // a URL that the data plane is still refusing. That field is the whole
  // verdict — phase, engine liveness and the model being present — and is
  // read rather than rebuilt here: this panel used to check the phase and
  // liveness itself, which unlocked a URL for an Ollama daemon that was up
  // without the tag pulled, and every request to it came back 503.
  const svc = {
    configReady: false,
    publicURL: '',
    modelName: '',
    phase: '',
    engineAlive: false,
    ready: false,
    modelExistsHealth: null,
    modelReady: false,
    downloadComplete: false,
    bytesTotal: 0,
    bytesCompleted: 0,
    filesTotal: 0,
    filesCompleted: 0,
    lastCurrentFile: '',
    audience: 'olares',
    format: '',
    // Per-format availability + endpoint catalog, derived from /api/endpoints.
    formats: { ollama: false, openai: false, anthropic: false, translate: false },
    endpoints: { ollama: [], openai: [], anthropic: [], translate: [] },
  };

  // Catalog category → UI format bucket. Categories come from
  // controlplane/endpoints.go (categoryOpenAI / categoryAnthropic /
  // categoryOllama / categoryTranslate).
  const FORMAT_META = {
    translate: { label: 'Translate', category: 'translate' },
    ollama: { label: 'Ollama', category: 'ollama-native' },
    openai: { label: 'OpenAI-Compatible', category: 'data-plane-openai' },
    anthropic: { label: 'Anthropic-Compatible', category: 'anthropic' },
  };

  // Base URL for a given format: Translate + native Ollama are rooted at
  // the host (paths are /translate* or /api/*). OpenAI + Anthropic live
  // under /v1 — do NOT append /v1 for Translate or clients will 404 on
  // /v1/languages etc.
  function pathFor(format) {
    return (format === 'ollama' || format === 'translate') ? '' : '/v1';
  }
  function baseURLFor(format) {
    const root = (svc.publicURL || (window.location && window.location.origin) || '').replace(/\/+$/, '');
    if (!root) return '';
    return root + pathFor(format);
  }

  // LAN (.local) hostnames derived from the public host. Example:
  //   public  b76c1d66.cidiskid.olares.com
  //   mac     b76c1d66.cidiskid.olares.local   (TLD .com → .local, keep dots)
  //   windows b76c1d66-cidiskid-olares.local   (stem dots → dashes, + .local)
  // Returns null when there is no usable multi-label host.
  function lanHosts() {
    const root = svc.publicURL || (window.location && window.location.origin) || '';
    let host = '';
    try { host = new URL(root).hostname; } catch (e) { return null; }
    const parts = host.split('.').filter(Boolean);
    if (parts.length < 2) return null;
    const stem = parts.slice(0, -1); // drop the public TLD (.com)
    return { mac: stem.join('.') + '.local', win: stem.join('-') + '.local' };
  }
  function lanURLFor(host, format) { return 'http://' + host + pathFor(format); }

  // Prefer Translate when /api/endpoints reports an available translate
  // category so MODEL_MODE=translate presets do not lead with a misleading
  // …/v1 OpenAI Base URL.
  // Ollama engines otherwise lead with native Ollama; everything else
  // leads with OpenAI-compatible.
  function defaultFormat() {
    if (svc.formats.translate) return 'translate';
    const order = (LLM.engineKind || '').toLowerCase() === 'ollama'
      ? ['ollama', 'openai', 'anthropic']
      : ['openai', 'anthropic', 'ollama'];
    return order.find((f) => svc.formats[f]) || '';
  }

  function renderFormatRadios() {
    const group = el('formatGroup');
    if (!group) return;
    const avail = Object.keys(FORMAT_META).filter((f) => svc.formats[f]);
    if (!svc.format || !svc.formats[svc.format]) svc.format = defaultFormat();
    group.innerHTML = '';
    avail.forEach((f) => {
      const label = document.createElement('label');
      label.className = 'radio';
      const input = document.createElement('input');
      input.type = 'radio';
      input.name = 'apiFormat';
      input.value = f;
      input.checked = f === svc.format;
      input.addEventListener('change', () => {
        if (input.checked) { svc.format = f; renderConnection(); }
      });
      const span = document.createElement('span');
      span.textContent = FORMAT_META[f].label;
      label.append(input, span);
      group.append(label);
    });
  }

  function renderEndpoints() {
    const list = el('endpointList');
    const count = el('endpointCount');
    if (!list) return;
    const rows = svc.endpoints[svc.format] || [];
    if (count) count.textContent = String(rows.length);
    list.innerHTML = '';
    rows.forEach((ep) => {
      const li = document.createElement('li');
      const m = document.createElement('span');
      m.className = 'ep-method ' + (ep.method || '').toLowerCase();
      m.textContent = ep.method || '';
      const p = document.createElement('span');
      p.className = 'ep-path';
      p.textContent = ep.path || '';
      const d = document.createElement('span');
      d.className = 'ep-desc';
      d.textContent = ep.description || '';
      li.append(m, p, d);
      list.append(li);
    });
  }

  // Build one boxed "Base URL" row. tipKey selects the ? hover text per row
  // (LAN Mac/Windows vs Olares vs Remote); the label stays "Base URL".
  function buildUrlRow(url, ready, tipKey) {
    const row = document.createElement('div');
    row.className = 'inset';

    const label = document.createElement('span');
    label.className = 'inset__label';
    const text = document.createElement('span');
    text.textContent = LLM.t('svc.base_url');
    label.append(text, LLM.tipIcon('info', tipKey || 'svc.base_url_info'));

    const a = document.createElement('a');
    a.className = 'inset__value mono';
    a.textContent = url || '—';
    if (url && ready) { a.href = url; a.target = '_blank'; a.rel = 'noopener'; }

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'copy-icon';
    btn.title = LLM.t('svc.copy_url_title');
    btn.disabled = !ready || !url;
    const lbl = document.createElement('span');
    lbl.className = 'copy-icon-label';
    lbl.textContent = LLM.t('copy');
    btn.append(lbl);
    btn.addEventListener('click', (ev) => {
      ev.preventDefault();
      if (!btn.disabled) doCopy(btn, url);
    });

    row.append(label, a, btn);
    return row;
  }

  // renderConnection paints the audience/format-dependent bits: the base
  // URL row(s), the Remote VPN callout, and the endpoint catalog. Safe to
  // call repeatedly (config / health / endpoints / radio changes).
  let lastBaseUrlSig = null;
  function renderConnection() {
    if (!svc.configReady) return;
    const m = el('apiModel');
    if (m) m.textContent = svc.modelName || '—';

    // The connection panel (audience/format + base URL + endpoints) is
    // gated on a live engine, mirroring the Figma where it appears only in
    // the Engine=READY state.
    const detail = el('engineDetail');
    if (detail) detail.hidden = svc.engineAlive !== true;

    const ready = svc.ready;
    const list = el('baseUrlList');
    if (list) {
      // Compute the desired rows first. Apps in Olares / Remote → one
      // public-URL row. Devices in LAN → two .local rows (Windows first,
      // then Mac); each ? shows its platform.
      const rows = [];
      if (svc.audience === 'lan') {
        const h = lanHosts();
        if (h) {
          rows.push({ url: lanURLFor(h.win, svc.format), tip: 'tip.base_url_win' });
          rows.push({ url: lanURLFor(h.mac, svc.format), tip: 'tip.base_url_mac' });
        } else {
          rows.push({ url: '', tip: 'svc.base_url_info' });
        }
      } else if (svc.audience === 'remote') {
        rows.push({
          url: svc.format ? baseURLFor(svc.format) : '',
          tip: svc.format === 'translate' ? 'tip.base_url_translate' : 'tip.base_url_remote',
        });
      } else {
        rows.push({
          url: svc.format ? baseURLFor(svc.format) : '',
          tip: svc.format === 'translate' ? 'tip.base_url_translate' : 'tip.base_url_olares',
        });
      }
      // Only rebuild when the rows actually change. The 5 s health poll
      // calls renderConnection repeatedly; rebuilding identical rows would
      // destroy and recreate the ? tooltip dots mid-hover, making the
      // bubble flicker.
      const sig = JSON.stringify({ ready: ready, rows: rows });
      if (sig !== lastBaseUrlSig) {
        lastBaseUrlSig = sig;
        list.innerHTML = '';
        rows.forEach((r) => list.append(buildUrlRow(r.url, ready, r.tip)));
      }
    }

    const warn = el('remoteWarning');
    if (warn) warn.hidden = svc.audience !== 'remote';

    renderEndpoints();
  }

  function fetchEndpoints() {
    fetch('/api/endpoints').then((r) => r.json()).then((data) => {
      const rows = (data && data.endpoints) || [];
      svc.formats = { ollama: false, openai: false, anthropic: false, translate: false };
      svc.endpoints = { ollama: [], openai: [], anthropic: [], translate: [] };
      rows.forEach((ep) => {
        if (!ep || ep.available === false) return;
        for (const key of Object.keys(FORMAT_META)) {
          if (ep.category === FORMAT_META[key].category) {
            svc.formats[key] = true;
            svc.endpoints[key].push(ep);
          }
        }
      });
      renderFormatRadios();
      renderConnection();
    }).catch(() => {});
  }

  // When the console itself is opened over a LAN .local hostname (Mac
  // dot-host / Windows dash-host), the caller is by definition a device on
  // the LAN — the "Apps in Olares" and "Remote" audiences don't apply, so
  // we drop them and lock the selector to "Devices in LAN".
  function isLanLocalHost() {
    const h = (window.location && window.location.hostname) || '';
    return /\.local$/i.test(h);
  }

  // Audience radios + model-card collapse are static controls — wire once.
  (function bindServiceControls() {
    const lanOnly = isLanLocalHost();
    if (lanOnly) svc.audience = 'lan';
    document.querySelectorAll('#audienceGroup input[name="audience"]').forEach((input) => {
      if (lanOnly) {
        const lan = input.value === 'lan';
        input.checked = lan;
        const label = input.closest('label');
        if (label) label.hidden = !lan;
      }
      input.addEventListener('change', () => {
        if (input.checked) { svc.audience = input.value; renderConnection(); }
      });
    });
    const collapse = el('modelCollapse');
    const body = el('modelCardBody');
    if (collapse && body) {
      collapse.addEventListener('click', () => {
        const open = collapse.getAttribute('aria-expanded') !== 'false';
        collapse.setAttribute('aria-expanded', open ? 'false' : 'true');
        body.hidden = open;
      });
    }
  })();

  // Shared clipboard write + transient "Copied" feedback on a copy button.
  async function doCopy(btn, text) {
    try {
      await navigator.clipboard.writeText((text || '').trim());
      const label = btn.querySelector('.copy-icon-label');
      const orig = label ? label.textContent : '';
      btn.classList.add('copied');
      if (label) label.textContent = LLM.t('copied');
      setTimeout(() => {
        btn.classList.remove('copied');
        if (label) label.textContent = orig;
      }, 1200);
    } catch (err) {
      LLM.log('clipboard copy failed: ' + err.message, 'error');
    }
  }

  // Static [data-target] copy buttons (e.g. the Model name box). The Base
  // URL rows are dynamic and wire their own copy handler in buildUrlRow.
  function bindCopyButtons() {
    document.querySelectorAll('button.copy-icon[data-target]').forEach((btn) => {
      btn.addEventListener('click', (ev) => {
        ev.preventDefault();
        if (btn.disabled) return;
        const tgt = el(btn.dataset.target);
        if (!tgt) return;
        const text = (tgt.tagName === 'A' ? (tgt.getAttribute('href') || tgt.textContent) : tgt.textContent) || '';
        doCopy(btn, text);
      });
    });
  }
  bindCopyButtons();

  // ─── status > error card ──────────────────────────────────────────
  // The Diagnostics card is hidden on the happy path and only shown
  // when the lifecycle is unhealthy: phase in {degraded, failed} or a
  // non-empty last_error. last_error is a current-failure signal
  // (cleared on download re-entry / ready), so a recovered retry hides
  // the card even if retry_count > 0.
  function applyErrorState(p) {
    const card = el('errorCard');
    if (!card) return;
    const phase = p.phase || '';
    const bad = phase === 'degraded' || phase === 'failed' || !!(p.last_error && p.last_error.length);
    card.hidden = !bad;
  }

  // The single action button mirrors the Figma's state-dependent affordance,
  // but only exposes the action the control plane actually backs: a forced
  // ensure pass (POST /api/retry?force=true). It shows as "Re-download" on
  // the happy path and "Retry" on the failure path; during init/download/
  // loading the row stays empty (no cancel/pause API).
  function applyActionButton(phase) {
    const btn = el('retryBtn');
    if (!btn) return;
    if (phase === 'ready') {
      btn.hidden = false;
      btn.textContent = LLM.t('act.redownload');
    } else if (phase === 'degraded' || phase === 'failed') {
      btn.hidden = false;
      btn.textContent = LLM.t('act.retry');
    } else {
      btn.hidden = true;
    }
  }

  let autoCollapsedOnce = false;

  function fileRowStatus(file, complete) {
    const status = complete ? 'done' : (file && file.status);
    const total = (file && file.bytes_total) || 0;
    const done = status === 'waiting' ? 0 : ((file && file.bytes_completed) || 0);
    // Finished rows keep the size; "Done" is only a fallback when
    // the tree never learned a length. In-flight rows stay a pair
    // so waiting is 0 / total, not an empty track.
    if (status === 'done') {
      const size = total > 0 ? total : done;
      return size > 0 ? fmtBytes(size) : LLM.t('prog.file_done');
    }
    if (total > 0) {
      const pair = fmtProgressPair(Math.min(done, total), total, true);
      return pair.done + ' / ' + pair.total;
    }
    if (status === 'waiting') return LLM.t('prog.file_waiting');
    if (done > 0) return fmtBytes(done);
    return LLM.t('prog.file_downloading');
  }

  function renderFileList(files, complete) {
    const list = el('fileList');
    if (!list) return false;
    const show = files.length > 1;
    list.hidden = !show;
    if (!show) {
      list.replaceChildren();
      return false;
    }
    const frag = document.createDocumentFragment();
    files.forEach((file) => {
      const status = complete ? 'done' : (file.status || 'waiting');
      const li = document.createElement('li');
      li.className = 'file-row file-row--' + status;
      const name = document.createElement('span');
      name.className = 'file-row__name';
      name.textContent = displayFileName(file.path) || file.path || '—';
      name.title = file.path || '';
      const meta = document.createElement('span');
      meta.className = 'file-row__status';
      meta.textContent = fileRowStatus(file, complete);
      li.appendChild(name);
      li.appendChild(meta);
      const bar = document.createElement('div');
      bar.className = 'file-row__bar';
      const fill = document.createElement('div');
      fill.className = 'file-row__bar-fill';
      const total = file.bytes_total || 0;
      const done = status === 'waiting' ? 0 : (file.bytes_completed || 0);
      let pct = 0;
      if (status === 'done') pct = 100;
      else if (status === 'waiting') pct = 0;
      else if (total > 0) pct = Math.min(100, Math.floor((done / total) * 100));
      fill.style.width = pct + '%';
      bar.appendChild(fill);
      li.appendChild(bar);
      frag.appendChild(li);
    });
    list.replaceChildren(frag);
    return true;
  }

  function applyProgress(p) {
    svc.phase = p.phase || '';
    svc.bytesTotal = p.bytes_total || 0;
    svc.bytesCompleted = p.bytes_completed || 0;
    svc.filesTotal = p.files_total || 0;
    svc.filesCompleted = p.files_completed || 0;
    renderPhaseBadge();
    renderConnection();
    const total = p.bytes_total || 0;
    const done = p.bytes_completed || 0;
    // Download is "done" only after the lifecycle leaves download
    // (loading/ready). Byte equality is not enough: an unpinned
    // multi-file HF pass fills bytes_total per file, and a second
    // source can still start.
    const complete = svc.phase === 'ready' || svc.phase === 'loading';
    svc.downloadComplete = complete;
    renderModelExists();

    // Auto-collapse once so the services below are visible. The
    // finished totals and file list stay in the body for a re-expand.
    if (!complete) {
      autoCollapsedOnce = false;
    } else if (!autoCollapsedOnce) {
      autoCollapsedOnce = true;
      const c = el('modelCollapse'), body = el('modelCardBody');
      if (c && body) { c.setAttribute('aria-expanded', 'false'); body.hidden = true; }
    }

    // Progress block: "{done} of {total}" on the left
    // with the percentage on the right, the bar, then a meta line of
    // "{speed} · {eta} left · {n} transport retries". bytes_total is only
    // 0 while the system is still starting and hasn't learned the size yet
    // — show an indeterminate bar + "Initializing…" then.
    const block = el('progressLabel') && el('progressLabel').closest('.progress-block');
    const bar = el('bar');
    const label = el('progressLabel');
    const pct = el('progressPct');
    if (total > 0) {
      const filesPending = (p.files_total || 0) > 1 && (p.files_completed || 0) < p.files_total && svc.phase === 'download';
      let ratio = Math.min(1, done / total);
      // 100% / a full bar is reserved for "this pass's files are in".
      // Cap whenever files remain — 99.8% already looks finished.
      if (filesPending) ratio = Math.min(ratio, 0.99);
      bar.classList.remove('indeterminate');
      bar.style.width = (ratio * 100) + '%';
      bar.classList.toggle('done', complete);
      if (block) block.classList.toggle('done', complete);
      const pair = fmtProgressPair(Math.min(done, total), total, !complete);
      if (label) label.textContent = LLM.t('prog.of', { done: pair.done, total: pair.total });
      if (pct) pct.textContent = Math.floor(ratio * 100) + '%';
    } else {
      bar.classList.add('indeterminate');
      bar.classList.remove('done');
      if (block) block.classList.remove('done');
      if (label) label.textContent = phaseLabel(svc.phase, total, done);
      if (pct) pct.textContent = '';
    }

    // Meta line: only the parts we actually have, joined by dots. Shown
    // mainly during an active download; hidden when there's nothing to say.
    // Speed/ETA stay off until a bandwidth-bound file; do not fill that
    // gap with a static "fetching files" — name the file instead, and
    // keep the last name across 1 Hz polls so tiny files do not blank it.
    const filesTotal = p.files_total || 0;
    const filesDone = p.files_completed || 0;
    const fileRows = Array.isArray(p.files) ? p.files : [];
    const multi = filesTotal > 1 || fileRows.length > 1;
    const filesRemain = multi && filesDone < (filesTotal || fileRows.length);
    const listShown = renderFileList(fileRows, complete);
    if (svc.phase === 'download' && p.current_file) {
      svc.lastCurrentFile = p.current_file;
    } else if (svc.phase !== 'download' || (multi && !filesRemain)) {
      svc.lastCurrentFile = '';
    }
    const rawFile = (!listShown && svc.phase === 'download')
      ? (p.current_file || (filesRemain ? svc.lastCurrentFile : ''))
      : '';
    const shownFile = displayFileName(rawFile);
    const hasSpeed = p.speed_bytes_per_sec > 0;

    const metaParts = [];
    if (multi) {
      metaParts.push(LLM.t('prog.files', { done: filesDone, total: filesTotal || fileRows.length }));
    }
    if (shownFile) {
      metaParts.push(hasSpeed
        ? shownFile
        : LLM.t('prog.downloading_file', { file: shownFile }));
    }
    if (hasSpeed) {
      metaParts.push(fmtSpeed(p.speed_bytes_per_sec));
    } else if (svc.phase === 'download' && total <= 0) {
      metaParts.push(LLM.t('prog.estimating'));
    }
    if (p.eta_seconds > 0) metaParts.push(LLM.t('prog.left', { eta: fmtETA(p.eta_seconds) }));
    if (p.transport_retries > 0) {
      metaParts.push(LLM.t(p.transport_retries === 1 ? 'prog.retry' : 'prog.retries', { n: p.transport_retries }));
    }
    const meta = el('progressMeta');
    if (meta) {
      meta.hidden = metaParts.length === 0;
      meta.textContent = metaParts.join(' · ');
    }

    applyActionButton(svc.phase);

    el('lastError').textContent = p.last_error || '—';
    el('retryCount').textContent = p.retry_count || 0;
    el('transportRetries').textContent = p.transport_retries || 0;
    applyErrorState(p);

    // Broadcast the current lifecycle phase so tabs that don't poll
    // /api/progress (gpu.js via config.js) can gate on phase=ready.
    try {
      window.dispatchEvent(new CustomEvent('llm:phase', { detail: { phase: p.phase || '' } }));
    } catch (_) {}
  }

  function applyHealth(h) {
    setLed('ledEngine', h.engine_alive);
    const wasAlive = svc.engineAlive;
    svc.engineAlive = h.engine_alive === true;
    // The catalog gains the engine's own rows once it answers, so re-read it then.
    if (svc.engineAlive && !wasAlive) fetchEndpoints();
    svc.modelExistsHealth = h.model_exists == null ? null : h.model_exists === true;
    svc.ready = h.ready === true;
    renderPhaseBadge();
    renderModelExists();
    renderConnection();
    setStatePill('engineAlive',
      h.engine_alive == null ? LLM.t('st.unknown') : (h.engine_alive ? LLM.t('st.running') : LLM.t('st.waiting')),
      h.engine_alive === true ? 'ok' : h.engine_alive === false ? 'warn' : '');
  }

  function refreshHealth() {
    fetch('/healthz').then((r) => r.json()).then(applyHealth).catch(() => {});
  }
  LLM.refreshHealth = refreshHealth;

  function refreshConfig() {
    fetch('/api/config').then((r) => r.json()).then((c) => {
      // /api/config (config_redacted.go Config.Redacted) marshals every
      // field snake_case: engine.kind / model_name / runtime.public_url.
      const kind = (c && c.engine && c.engine.kind) || '';
      const name = (c && c.model_name) || '';
      const downloadOnly = kind === '';

      const displayName = name || (downloadOnly ? 'download-only' : '?');
      el('modelMeta').textContent = displayName;
      const statusName = el('statusModelName');
      if (statusName) statusName.textContent = displayName;
      el('engineBadge').textContent = kind ? LLM.engineName(kind) : 'DOWNLOAD-ONLY';
      LLM.engineKind = kind;
      LLM.downloadOnly = downloadOnly;
      LLM.config = c;

      // Header mode pill (Chat / …) + model source subtitle, sourced from
      // the redacted config (which embeds model_spec + sources[]). The
      // Config tab refines caps from the live /api/model-spec.
      const spec = (c && c.model_spec) || {};
      LLM.specFromConfig = spec;
      const modeBadge = el('modeBadge');
      if (modeBadge) {
        if (spec.mode) {
          modeBadge.textContent = LLM.modeLabel(spec.mode);
          modeBadge.hidden = false;
        } else {
          modeBadge.hidden = true;
        }
      }
      const srcEl = el('modelSource');
      if (srcEl) {
        const sources = (c && c.sources) || [];
        const main = sources.find((s) => s && s.role === 'main') || sources[0];
        srcEl.textContent = (main && main.redacted_source) || (downloadOnly ? '' : '—');
      }

      // download-only: no engine to probe, no args to launch, no GPU
      // residency. Hide those blocks; keep download + error + spec editor.
      document.body.dataset.downloadOnly = downloadOnly ? '1' : '';
      const engineCard = el('engineCard');
      if (engineCard) engineCard.hidden = downloadOnly;
      // The "Download-Only mode" empty state stands in for Service status.
      const dlCard = el('downloadOnlyCard');
      if (dlCard) dlCard.hidden = !downloadOnly;
      const argsCard = el('argsCard');
      if (argsCard) argsCard.hidden = downloadOnly;
      const gpuBlock = el('gpuBlock');
      if (gpuBlock) gpuBlock.hidden = downloadOnly;

      svc.publicURL = (c && c.runtime && c.runtime.public_url) || '';
      svc.modelName = name;
      svc.configReady = true;
      renderFormatRadios();
      renderConnection();

      if (LLM.onConfig) LLM.onConfig(c);
    }).catch(() => {
      el('modelMeta').textContent = LLM.t('meta.config_unavailable');
    });
  }

  function refreshBuildInfo() {
    fetch('/api/build-info').then((r) => r.json()).then((b) => {
      const parts = [];
      if (b.version) parts.push(b.version);
      if (b.commit && b.commit !== 'none') parts.push('@' + b.commit.slice(0, 7));
      const text = parts.join(' ');
      const title = JSON.stringify(b, null, 2);
      // Both the desktop meta row (#buildInfo) and the mobile app header
      // (#buildInfoBar) carry .js-build; fill whichever is in the DOM.
      document.querySelectorAll('.js-build').forEach((node) => {
        node.textContent = text;
        node.title = title;
      });
      LLM.buildInfo = b;
    }).catch(() => {});
  }

  // ─── actions ──────────────────────────────────────────────────────
  function postRetry(qs, label) {
    if (!confirm(LLM.t('confirm.proceed', { label: label }))) return;
    fetch('/api/retry' + qs, { method: 'POST' }).then(async (r) => {
      if (r.ok) {
        LLM.log(label + ' triggered', 'phase');
      } else {
        const t = await r.text().catch(() => r.statusText);
        LLM.log(label + ' failed: ' + r.status + ' ' + t, 'error');
      }
    }).catch((e) => LLM.log(label + ' error: ' + e.message, 'error'));
  }
  el('retryBtn').onclick = () => postRetry('?force=true', el('retryBtn').textContent || 'Retry');

  // ─── boot ─────────────────────────────────────────────────────────
  refreshBuildInfo();
  refreshConfig();
  refreshHealth();
  fetchEndpoints();
  setInterval(refreshHealth, 5000);

  // ─── /api/progress polling (1 s) ──────────────────────────────────
  let lastPhase = null;
  function loadProgress() {
    fetch('/api/progress')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
      .then((state) => {
        if (state && state.phase && state.phase !== lastPhase) {
          if (lastPhase !== null) {
            LLM.log('phase ' + lastPhase + ' → ' + state.phase, 'phase');
            refreshHealth();
          }
          lastPhase = state.phase;
        }
        applyProgress(state);
      })
      .catch(() => {});
  }
  loadProgress();
  setInterval(loadProgress, 1000);
})();
