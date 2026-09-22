/*
 * charts.js — uPlot wrapper for any tab that needs a time-series chart.
 *
 * uPlot is lazy-loaded the first time a chart is requested so tabs that
 * never draw one pay no script-parse cost. After load every subsequent
 * createChart() returns synchronously.
 *
 * The wrapper exposes:
 *   * LLM.charts.create({el, title, series, opts}) → returns a handle
 *     with setData(rows) where rows is [xArr, ...yArrs]
 *   * LLM.charts.ensureLoaded() → Promise<void>
 *
 * Series are configured with stable colours from the GitHub dark
 * palette so charts read consistently across tabs.
 */
(function () {
  'use strict';
  const LLM = (window.LLM = window.LLM || {});

  // GitHub dark palette, picked for high contrast on #0d1117. Index
  // order matters: chart.series[0] is reserved by uPlot for the X
  // axis, so the first user series uses palette[0].
  const palette = [
    '#79c0ff', // blue
    '#7ee787', // green
    '#ffa657', // orange
    '#d2a8ff', // purple
    '#ff7b72', // red
    '#e3b341', // yellow
    '#a5d6ff', // pale blue
    '#56d364', // pale green
  ];

  let loadPromise = null;
  function ensureLoaded() {
    if (window.uPlot) return Promise.resolve();
    if (loadPromise) return loadPromise;
    loadPromise = new Promise((resolve, reject) => {
      // Inject the stylesheet alongside the script — uPlot crosshair
      // and legend layout depend on it. Both files live under /static
      // (vendor/) so they're served by the same FileServer.
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = '/static/vendor/uPlot.min.css';
      document.head.appendChild(link);

      const s = document.createElement('script');
      s.src = '/static/vendor/uPlot.iife.min.js';
      s.async = true;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error('uPlot failed to load'));
      document.head.appendChild(s);
    });
    return loadPromise;
  }

  // theme overrides shared by every chart. uPlot reads inline styles
  // not CSS variables, so we hard-code the GitHub dark colours; if you
  // ever flip the dashboard to light mode, update these alongside
  // dashboard.css.
  function theme() {
    return {
      background: 'transparent',
      axis: { stroke: '#adadad', grid: { stroke: '#ffffff14' }, ticks: { stroke: '#3d3d3d' } },
    };
  }

  function build(elOrId, cfg, data) {
    const node = typeof elOrId === 'string' ? document.getElementById(elOrId) : elOrId;
    if (!node) throw new Error('chart container missing');
    // Drop the "empty" placeholder if present.
    node.classList.remove('empty');
    node.innerHTML = '';

    // Auto-size to the host card. uPlot needs explicit pixel dims so
    // we observe size changes via ResizeObserver and call setSize().
    const rect = node.getBoundingClientRect();
    const baseOpts = {
      width: Math.max(280, Math.floor(rect.width || 480)),
      height: 180,
      cursor: { drag: { x: true, y: false } },
      legend: { show: true },
      axes: [theme().axis, theme().axis],
      ...cfg,
    };

    // Adopt our palette for every series the caller didn't paint. The
    // first slot belongs to the X axis (uPlot convention).
    baseOpts.series = (cfg.series || []).map((s, i) => {
      if (i === 0) return s; // X
      return Object.assign({ stroke: palette[(i - 1) % palette.length], width: 1.5 }, s);
    });

    const u = new window.uPlot(baseOpts, data, node);

    // Resize on window changes; cheap because uPlot redraws lazily.
    const ro = new ResizeObserver(() => {
      const r = node.getBoundingClientRect();
      if (r.width > 0) u.setSize({ width: Math.floor(r.width), height: 180 });
    });
    ro.observe(node);

    return {
      uplot: u,
      setData(rows) { u.setData(rows); },
      setSeries(seriesCfg) {
        // Rebuild when the set of series changes (e.g. new route
        // appears in /metrics). uPlot can't add series after construction,
        // so we destroy + rebuild — costly but rare.
        u.destroy();
        ro.disconnect();
        const replacement = build(node, { ...cfg, series: seriesCfg }, rows(seriesCfg));
        Object.assign(this, replacement);
      },
    };
  }
  function rows(series) {
    // Helper used by setSeries above when rebuilding with zero data.
    return [[], ...series.slice(1).map(() => [])];
  }

  function createTimeChart(opts) {
    return ensureLoaded().then(() => build(opts.el, {
      title: opts.title || '',
      width: opts.width,
      height: opts.height || 180,
      series: opts.series,
      scales: opts.scales || { x: { time: true } },
    }, opts.data || [[], ...opts.series.slice(1).map(() => [])]));
  }

  LLM.charts = {
    ensureLoaded,
    create: createTimeChart,
    palette,
  };
})();
