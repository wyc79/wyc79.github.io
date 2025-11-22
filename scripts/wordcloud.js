/**
 * wordcloud.js
 * Implementation inspired by Jason Davies' word cloud generator,
 * adapted for the toolbox controls (wc-*) and resilient to both
 * legacy D3 v3 + d3.layout.cloud and the modern CDN bundle.
 */
(function () {
  'use strict';

  const d3 = window.d3;
  const cloudFactory =
    d3 && d3.layout && typeof d3.layout.cloud === 'function'
      ? d3.layout.cloud
      : typeof window.d3Cloud === 'function'
      ? window.d3Cloud
      : null;

  if (!d3 || !cloudFactory) {
    console.warn('[wordcloud] D3 or d3-cloud is not available.');
    return;
  }

  const els = {
    input: document.getElementById('wc-input'),
    perLine: document.getElementById('wc-one-per-line'),
    max: document.getElementById('wc-max'),
    rotate: document.getElementById('wc-rotate'),
    scale: document.getElementById('wc-scale'),
    generate: document.getElementById('wc-generate'),
    downloadSvg: document.getElementById('wc-download'),
    downloadPng: document.getElementById('wc-download-png'),
    container: document.getElementById('wc-container'),
  };

  if (!els.input || !els.generate || !els.container) {
    console.warn('[wordcloud] Required controls are missing.');
    return;
  }

  const STOP_WORDS = new Set([
    'a','an','and','are','as','at','be','but','by','for','from','has','have',
    'in','is','it','its','of','on','or','that','the','to','was','were','will','with'
  ]);

  let currentWords = null;
  let currentBounds = null;
  let currentDims = null;

  function toLinearScale() {
    return typeof d3.scaleLinear === 'function' ? d3.scaleLinear() : d3.scale.linear();
  }
  function toLogScale() {
    return typeof d3.scaleLog === 'function' ? d3.scaleLog() : d3.scale.log();
  }
  function toSqrtScale() {
    return typeof d3.scaleSqrt === 'function' ? d3.scaleSqrt() : d3.scale.sqrt();
  }
  function toColorScale() {
    if (typeof d3.scaleOrdinal === 'function' && d3.schemeCategory10) {
      return d3.scaleOrdinal(d3.schemeCategory10);
    }
    return d3.scale.category10();
  }

  function parseInput(raw) {
    const text = (raw || '').toLowerCase();
    const tokens = els.perLine.checked
      ? text.split(/\r?\n+/).map(s => s.trim()).filter(Boolean)
      : text.split(/[^a-z0-9\-']+/i).map(s => s.trim()).filter(Boolean);

    const map = new Map();
    for (const word of tokens) {
      if (!word || STOP_WORDS.has(word)) continue;
      map.set(word, (map.get(word) || 0) + 1);
    }

    const maxWords = Math.max(10, Math.min(parseInt(els.max.value, 10) || 200, 1000));
    return Array.from(map.entries())
      .map(([text, value]) => ({ text, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, maxWords);
  }

  function measureBounds(words) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const w of words) {
      const halfW =
        w.x0 !== undefined && w.x1 !== undefined
          ? Math.max(Math.abs(w.x0), Math.abs(w.x1))
          : w.width
          ? w.width / 2
          : Math.max(10, (w.size || 10) * (w.text ? w.text.length : 1) * 0.25);
      const halfH =
        w.y0 !== undefined && w.y1 !== undefined
          ? Math.max(Math.abs(w.y0), Math.abs(w.y1))
          : Math.max(10, w.size || 10);
      const left = w.x - halfW;
      const right = w.x + halfW;
      const top = w.y - halfH;
      const bottom = w.y + halfH;
      if (left < x0) x0 = left;
      if (right > x1) x1 = right;
      if (top < y0) y0 = top;
      if (bottom > y1) y1 = bottom;
    }
    if (!isFinite(x0) || !isFinite(y0) || !isFinite(x1) || !isFinite(y1)) {
      return null;
    }
    return [{ x: x0, y: y0 }, { x: x1, y: y1 }];
  }

  function buildTransform(width, height, bounds) {
    if (!bounds) return `translate(${width / 2},${height / 2})`;
    const spanX = Math.max(bounds[1].x - bounds[0].x, 1);
    const spanY = Math.max(bounds[1].y - bounds[0].y, 1);
    const rawScale = Math.min((width * 0.9) / spanX, (height * 0.9) / spanY);
    const scale = Math.min(Math.max(rawScale, 0.5), 5);
    const centerX = (bounds[0].x + bounds[1].x) / 2;
    const centerY = (bounds[0].y + bounds[1].y) / 2;
    // SVG applies transform functions right-to-left: first move content to origin, then scale, then move to viewport center.
    return `translate(${width / 2},${height / 2}) scale(${scale}) translate(${-centerX},${-centerY})`;
  }

  function makeFontScale(words, dims) {
    if (!words.length) return () => 12;
    let min = Infinity;
    let max = -Infinity;
    for (const w of words) {
      if (w.value < min) min = w.value;
      if (w.value > max) max = w.value;
    }
    if (!isFinite(min) || !isFinite(max)) return () => 12;
    if (min === max) max = min + 1;

    const base = Math.max(200, Math.min(dims.width, dims.height));
    const range = [24, Math.max(72, Math.floor(base * 0.3))];
    const mode = (els.scale && els.scale.value) || 'sqrt';

    if (mode === 'linear') {
      return toLinearScale().domain([min, max]).range(range);
    }
    if (mode === 'log') {
      return toLogScale().domain([Math.max(1, min), max]).range(range);
    }
    return toSqrtScale().domain([min, max]).range(range);
  }

  function renderCloud(words, dims, bounds) {
    const container = els.container;
    container.innerHTML = '';

    const svg = d3.select(container)
      .append('svg')
      .attr('id', 'wc-svg')
      .attr('width', dims.width)
      .attr('height', dims.height)
      .attr('viewBox', `0 0 ${dims.width} ${dims.height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    const node = svg.node();
    node.style.setProperty('width', `${dims.width}px`, 'important');
    node.style.setProperty('height', `${dims.height}px`, 'important');
    node.style.setProperty('display', 'block', 'important');

    const actualBounds = bounds || measureBounds(words);
    const transform = buildTransform(dims.width, dims.height, actualBounds);
    const group = svg.append('g').attr('transform', transform);

    const color = toColorScale();

    const texts = group.selectAll('text')
      .data(words)
      .enter()
      .append('text')
      .attr('text-anchor', 'middle')
      .style('font-family', 'Impact, system-ui, sans-serif')
      .style('line-height', '1')
      .style('letter-spacing', '0')
      .style('fill', (d, i) => color(i))
      .attr('transform', d => `translate(${d.x},${d.y}) rotate(${d.rotate || 0})`)
      .each(function (d) {
        this.style.setProperty('font-size', `${d.size}px`, 'important');
      })
      .text(d => d.text);

    texts.append('title').text(d => `${d.text} (${d.value})`);

    currentWords = words;
    currentBounds = actualBounds;
    currentDims = dims;
    els.downloadSvg.disabled = false;
    els.downloadPng.disabled = false;
  }

  function generate() {
    const words = parseInput(els.input.value);
    if (!words.length) {
      els.container.innerHTML = '<p class="meta">No words to display.</p>';
      els.downloadSvg.disabled = true;
      els.downloadPng.disabled = true;
      currentWords = null;
      return;
    }

    const width = Math.max(els.container.clientWidth || 0, 960);
    const height = Math.max(Math.round(width * 0.6), 520);
    els.container.style.minHeight = `${height}px`;

    const fontScale = makeFontScale(words, { width, height });
    const rotateMode = (els.rotate && els.rotate.value) || '0';
    const rotation = rotateMode === '90'
      ? () => (Math.random() < 0.5 ? 0 : 90)
      : () => 0;

    els.container.innerHTML = '<p class="meta">Generating word cloud…</p>';
    els.downloadSvg.disabled = true;
    els.downloadPng.disabled = true;

    const layout = cloudFactory()
      .size([width, height])
      .padding(4)
      .words(words.map(word => ({
        text: word.text,
        value: word.value,
        size: Math.max(8, Math.floor(fontScale(word.value))),
      })))
      .rotate(rotation)
      .font('Impact')
      .fontSize(d => d.size)
      .spiral('archimedean')
      .timeInterval(10);

    layout.on('end', (placed, bounds) => {
      renderCloud(placed, { width, height }, bounds || currentBounds);
    });

    layout.start();
  }

  function serializeSvg() {
    const svg = document.getElementById('wc-svg');
    if (!svg) return null;
    const serializer = new XMLSerializer();
    let markup = serializer.serializeToString(svg);
    if (!/^<svg[^>]+xmlns=/.test(markup)) {
      markup = markup.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"');
    }
    return markup;
  }

  function downloadSvg() {
    const markup = serializeSvg();
    if (!markup) return;
    const blob = new Blob([markup], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'wordcloud.svg';
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function downloadPng() {
    const markup = serializeSvg();
    if (!markup || !currentDims) return;
    const blob = new Blob([markup], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = function () {
      const canvas = document.createElement('canvas');
      canvas.width = currentDims.width;
      canvas.height = currentDims.height;
      const ctx = canvas.getContext('2d');
      const bg = getComputedStyle(document.body).getPropertyValue('--bg') || '#ffffff';
      ctx.fillStyle = bg.trim() || '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      canvas.toBlob(blob => {
        if (!blob) return;
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'wordcloud.png';
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(link.href), 0);
      }, 'image/png');
    };
    img.onerror = () => URL.revokeObjectURL(url);
    img.src = url;
  }

  els.generate.addEventListener('click', generate);
  els.downloadSvg.addEventListener('click', downloadSvg);
  els.downloadPng.addEventListener('click', downloadPng);
  els.input.addEventListener('keydown', evt => {
    if ((evt.ctrlKey || evt.metaKey) && evt.key === 'Enter') {
      evt.preventDefault();
      generate();
    }
  });

  // Initial state
  els.downloadSvg.disabled = true;
  els.downloadPng.disabled = true;
})();
