(function () {
  'use strict';

  var ITEMS = [
    { id: 'projects',     label: 'PROJECTS',     href: 'pages/projects.html',     fontSize: 130, offsetX: 0,  offsetY: 0  },
    { id: 'skills',       label: 'SKILLS',       href: 'pages/skills.html',       fontSize: 118, offsetX: 44, offsetY: -6 },
    { id: 'education',    label: 'EDUCATION',    href: 'pages/education.html',    fontSize: 102, offsetX: 22, offsetY: -6 },
    { id: 'publications', label: 'PUBLICATIONS', href: 'pages/publications.html', fontSize: 82,  offsetX: 60, offsetY: -4 },
    { id: 'toolbox',      label: 'TOOLBOX',      href: 'pages/toolbox.html',      fontSize: 74,  offsetX: 28, offsetY: -4 },
    { id: 'github',       label: 'GITHUB',       href: 'https://github.com/wyc79', fontSize: 66, offsetX: 12, offsetY: -2, external: true }
  ];

  // Polygon clip-path generators — each returns an angular Persona-style shape.
  var CLIP_SHAPES = [
    function (w, h) { return 'polygon(0px ' + (h*0.06) + 'px, ' + (w - h*0.55) + 'px 0px, ' + w + 'px ' + (h*0.42) + 'px, ' + (w - h*0.18) + 'px ' + h + 'px, 0px ' + (h*0.94) + 'px)'; },
    function (w, h) { return 'polygon(' + (h*0.12) + 'px 0px, ' + (w - h*0.30) + 'px ' + (h*0.04) + 'px, ' + w + 'px ' + (h*0.50) + 'px, ' + (w - h*0.08) + 'px ' + h + 'px, 0px ' + (h*0.88) + 'px)'; },
    function (w, h) { return 'polygon(0px ' + (h*0.10) + 'px, ' + (w - h*0.40) + 'px 0px, ' + w + 'px ' + (h*0.45) + 'px, ' + (w - h*0.25) + 'px ' + h + 'px, ' + (h*0.05) + 'px ' + (h*0.90) + 'px)'; },
    function (w, h) { return 'polygon(' + (h*0.08) + 'px ' + (h*0.02) + 'px, ' + (w - h*0.22) + 'px 0px, ' + w + 'px ' + (h*0.55) + 'px, ' + (w - h*0.35) + 'px ' + h + 'px, 0px ' + (h*0.82) + 'px)'; },
    function (w, h) { return 'polygon(0px 0px, ' + (w - h*0.45) + 'px ' + (h*0.05) + 'px, ' + w + 'px ' + (h*0.38) + 'px, ' + (w - h*0.12) + 'px ' + (h*0.98) + 'px, ' + (h*0.10) + 'px ' + h + 'px)'; }
  ];

  function buildMenu(root) {
    var menu = root.querySelector('.p3-menu');
    var hint = root.querySelector('.p3-hint');
    if (!menu) return;

    var rows = [];
    var active = 0;

    ITEMS.forEach(function (item, i) {
      var a = document.createElement('a');
      a.className = 'p3-row';
      a.href = item.href;
      if (item.external) {
        a.target = '_blank';
        a.rel = 'noopener';
      }
      a.style.marginLeft = item.offsetX + 'px';
      a.style.marginTop  = item.offsetY + 'px';

      var hl = document.createElement('div');
      hl.className = 'p3-highlight';

      var label = document.createElement('span');
      label.className = 'p3-label';
      label.style.fontSize = item.fontSize + 'px';
      label.textContent = item.label;

      a.appendChild(hl);
      a.appendChild(label);

      a.addEventListener('mouseenter', function () { setActive(i); });
      a.addEventListener('focus',      function () { setActive(i); });

      menu.appendChild(a);
      rows.push({ row: a, highlight: hl, label: label, item: item });
    });

    function sizeHighlight(entry, idx) {
      // Measure the actual rendered label rather than estimating from text length.
      var rect = entry.label.getBoundingClientRect();
      var w = Math.max(80, rect.width + 80);
      var h = Math.max(20, rect.height * 1.02);
      entry.highlight.style.width  = w + 'px';
      entry.highlight.style.height = h + 'px';
      var clipFn = CLIP_SHAPES[idx] || CLIP_SHAPES[0];
      entry.highlight.style.clipPath = clipFn(w, h);
      entry.highlight.style.webkitClipPath = clipFn(w, h);
    }

    function updateStyles() {
      rows.forEach(function (entry, i) {
        var dist = Math.abs(i - active);
        var opacity = (i === active) ? 1 : Math.max(0.18, 1 - dist * 0.38);
        entry.label.style.opacity = opacity;
        if (i === active) entry.row.classList.add('active');
        else              entry.row.classList.remove('active');
      });
    }

    function setActive(i) {
      if (i < 0) i = 0;
      if (i >= rows.length) i = rows.length - 1;
      if (i === active) return;
      active = i;
      updateStyles();
    }

    function activateCurrent() {
      var entry = rows[active];
      if (!entry) return;
      var href = entry.item.href;
      if (entry.item.external) {
        window.open(href, '_blank', 'noopener');
        return;
      }
      if (href.charAt(0) === '#') {
        var el = document.querySelector(href);
        if (el && el.scrollIntoView) {
          el.scrollIntoView({ behavior: 'smooth', block: 'start' });
          return;
        }
      }
      window.location.href = href;
    }

    document.addEventListener('keydown', function (e) {
      // Only respond to arrow/enter when the hero is roughly in view.
      var heroRect = root.getBoundingClientRect();
      var visible = heroRect.bottom > 120 && heroRect.top < window.innerHeight * 0.6;
      if (!visible) return;
      if (e.key === 'ArrowUp')   { e.preventDefault(); setActive(active - 1); }
      if (e.key === 'ArrowDown') { e.preventDefault(); setActive(active + 1); }
      if (e.key === 'Enter')     { e.preventDefault(); activateCurrent(); }
    });

    function sizeAll() {
      rows.forEach(function (entry, i) { sizeHighlight(entry, i); });
    }

    // Initial sizing (after fonts load if possible).
    sizeAll();
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(sizeAll).catch(function () {});
    }
    window.addEventListener('resize', sizeAll);

    updateStyles();

    // Staggered mount-in.
    setTimeout(function () {
      rows.forEach(function (entry, i) {
        setTimeout(function () {
          entry.row.classList.add('mounted');
        }, i * 80);
      });
      if (hint) hint.classList.add('mounted');
      var brand = document.querySelector('.p3-brand');
      if (brand) brand.classList.add('mounted');
      var brandLinks = document.querySelector('.p3-brand-links');
      if (brandLinks) brandLinks.classList.add('mounted');
    }, 80);
  }

  function init() {
    // Inject Bebas Neue font (once).
    if (!document.getElementById('p3-bebas-font')) {
      var link = document.createElement('link');
      link.id = 'p3-bebas-font';
      link.rel = 'stylesheet';
      link.href = 'https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap';
      document.head.appendChild(link);
    }
    var root = document.querySelector('.p3-root');
    if (root) buildMenu(root);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
