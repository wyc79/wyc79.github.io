(function () {
  'use strict';

  // Per-item design data:
  //   tilt     - fixed rotation (deg) on the VISUAL layer only; the <a>'s hit
  //              area stays axis-aligned.
  //   shift    - fixed translateY (px) on the visual layer so tabs can overlap
  //              visually without overlapping hitboxes.
  //   wedge    - polygon variant params so no two rows share the same silhouette:
  //              baseTop/baseBot set the left base height,
  //              leanBot pushes the bottom-left corner inward (angled left edge),
  //              tipY places the tip, tipScale lengthens the right taper,
  //              backOffX is NEGATIVE so the pink back layer peeks to the LEFT,
  //              backOffY sets its vertical drop.
  var ITEMS = [
    { id: 'projects',     label: 'PROJECTS',     href: 'pages/projects.html',     fontSize: 96, offsetX: 0,  offsetY: 0, tilt: -3.0, shift:  0,
      wedge: { baseTop: -0.04, baseBot: 1.00, leanBot: 0.18, tipY: 0.48, tipScale: 1.45, backOffX: -0.045, backOffY: 0.09 } },
    { id: 'skills',       label: 'SKILLS',       href: 'pages/skills.html',       fontSize: 88, offsetX: 44, offsetY: 0, tilt:  4.0, shift: -4,
      wedge: { baseTop: -0.02, baseBot: 0.95, leanBot: 0.22, tipY: 0.40, tipScale: 1.60, backOffX: -0.055, backOffY: 0.07 } },
    { id: 'education',    label: 'EDUCATION',    href: 'pages/education.html',    fontSize: 76, offsetX: 22, offsetY: 0, tilt: -2.0, shift:  2,
      wedge: { baseTop: -0.06, baseBot: 1.05, leanBot: 0.15, tipY: 0.55, tipScale: 1.30, backOffX: -0.035, backOffY: 0.12 } },
    { id: 'publications', label: 'PUBLICATIONS', href: 'pages/publications.html', fontSize: 62, offsetX: 60, offsetY: 0, tilt:  5.0, shift: -4,
      wedge: { baseTop:  0.00, baseBot: 0.98, leanBot: 0.25, tipY: 0.42, tipScale: 1.55, backOffX: -0.050, backOffY: 0.08 } },
    { id: 'toolbox',      label: 'TOOLBOX',      href: 'pages/toolbox.html',      fontSize: 56, offsetX: 28, offsetY: 0, tilt: -4.0, shift:  3,
      wedge: { baseTop: -0.05, baseBot: 1.04, leanBot: 0.20, tipY: 0.38, tipScale: 1.35, backOffX: -0.045, backOffY: 0.13 } },
    { id: 'github',       label: 'GITHUB',       href: 'https://github.com/wyc79', fontSize: 50, offsetX: 12, offsetY: 0, tilt:  3.0, shift: -2, external: true,
      wedge: { baseTop: -0.02, baseBot: 0.97, leanBot: 0.16, tipY: 0.52, tipScale: 1.50, backOffX: -0.035, backOffY: 0.10 } }
  ];

  var SVG_NS = 'http://www.w3.org/2000/svg';

  // Build the highlighted-item SVG composition for one row:
  //   [shadow text][top red wedge][bottom pink wedge][dark base text][bright text, clipped to top wedge]
  // The top wedge is drawn behind the text but the bright text layer is clipped
  // to the top wedge, so the uncovered portion of each letter shows the dark
  // base fill while the covered portion shows the bright fill - no blend modes.
  function buildHeroSvg(entry) {
    var fs    = entry.item.fontSize;
    var label = entry.item.label;
    var w     = entry.item.wedge;

    // Use the live label box so the wedge length actually matches the font
    // render (Bebas Neue width estimates are lossy).
    var lblRect = entry.label.getBoundingClientRect();
    var labelW  = Math.max(lblRect.width, fs * 0.5 * label.length);

    // Asymmetric padding:
    //   LEFT  - tight, plus a small margin for the pink back wedge to peek out
    //           without getting clipped.
    //   RIGHT - long taper room; tipScale makes some tabs reach farther.
    var padLeft  = Math.max(fs * 0.11, 14);
    var padRight = Math.max(fs * 0.55 * w.tipScale, 48);
    var padY     = Math.max(fs * 0.28, 22);
    var svgW     = labelW + padLeft + padRight;
    var svgH     = fs + padY * 2;

    var textX = padLeft;
    var textY = padY + fs * 0.85;

    // --- Front red wedge: 3-point triangle with an inward-leaning LEFT edge.
    //   top-left corner at x=0
    //   bottom-left corner pushed inward by (fs * leanBot)
    //   tip at the far right
    // The slanted left edge reads as a sharp angled entry rather than a block.
    var topPts = [
      [0,                 padY + fs * w.baseTop],
      [svgW,              padY + fs * w.tipY],
      [fs * w.leanBot,    padY + fs * w.baseBot]
    ];

    // --- Back pink wedge: same silhouette, offset LEFT (negative backOffX) and
    //     slightly DOWN so a larger strip is exposed on the left/back edge.
    //     Rendered BEFORE the red so the red covers most of it.
    var backOffX = fs * w.backOffX;   // negative → left reveal
    var backOffY = fs * w.backOffY;
    var botPts = topPts.map(function (p) {
      return [p[0] + backOffX, p[1] + backOffY];
    });

    var ptsStr = function (pts) {
      return pts.map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join(' ');
    };
    var topPtsStr = ptsStr(topPts);
    var botPtsStr = ptsStr(botPts);

    var clipId = 'p3hero-clip-' + entry.item.id;
    var shadowDx = Math.max(3, fs * 0.04);
    var shadowDy = Math.max(4, fs * 0.06);

    // Build as markup then parse - lets us keep the structure readable and
    // avoids the SVG-innerHTML namespace quirks in older browsers.
    var svgMarkup = [
      '<svg xmlns="', SVG_NS, '" class="p3-hero-svg" ',
        'viewBox="0 0 ', svgW, ' ', svgH, '" ',
        'width="', svgW, '" height="', svgH, '" ',
        'aria-hidden="true">',
        '<defs>',
          '<clipPath id="', clipId, '">',
            '<polygon points="', topPtsStr, '"/>',
          '</clipPath>',
        '</defs>',
        // Shadow text - offset behind everything for depth.
        '<text class="p3-hero-text p3-hero-shadow" ',
          'x="', (textX + shadowDx).toFixed(1), '" ',
          'y="', (textY + shadowDy).toFixed(1), '" ',
          'font-size="', fs, '">', escapeXml(label), '</text>',
        // Back pink wedge - drawn first so it sits BEHIND the red one, with
        // only a small sliver exposed around the lower-right edge.
        '<polygon class="p3-hero-wedge-bottom" points="', botPtsStr, '"/>',
        // Front red wedge - sits on top, covering most of the pink.
        '<polygon class="p3-hero-wedge-top" points="', topPtsStr, '"/>',
        // Dark base text over the wedges (visible where bright layer does not cover).
        '<text class="p3-hero-text p3-hero-dark" ',
          'x="', textX.toFixed(1), '" y="', textY.toFixed(1), '" ',
          'font-size="', fs, '">', escapeXml(label), '</text>',
        // Bright text clipped to the top wedge - only the covered portion reads bright.
        '<g clip-path="url(#', clipId, ')">',
          '<text class="p3-hero-text p3-hero-bright" ',
            'x="', textX.toFixed(1), '" y="', textY.toFixed(1), '" ',
            'font-size="', fs, '">', escapeXml(label), '</text>',
        '</g>',
      '</svg>'
    ].join('');

    var parser = new DOMParser();
    var doc    = parser.parseFromString(svgMarkup, 'image/svg+xml');
    var svgEl  = doc.documentElement;

    entry.hero.innerHTML = '';
    entry.hero.appendChild(svgEl);
    entry.hero.style.width  = svgW + 'px';
    entry.hero.style.height = svgH + 'px';
    // .p3-hero is absolutely positioned inside .p3-rot. The label sits at
    // x=0 of .p3-rot, and the SVG's internal text sits at x=padLeft, so we
    // offset the hero by -padLeft to align the two.
    entry.hero.style.left = (-padLeft) + 'px';
    entry.hero.style.top  = '50%';
    entry.hero.style.transform = 'translate(0, -50%)';
    entry.heroBuilt = true;
  }

  function escapeXml(s) {
    return String(s).replace(/[<>&'"]/g, function (c) {
      return c === '<' ? '&lt;' :
             c === '>' ? '&gt;' :
             c === '&' ? '&amp;' :
             c === "'" ? '&apos;' : '&quot;';
    });
  }

  function buildMenu(root) {
    var menu = root.querySelector('.p3-menu');
    var hint = root.querySelector('.p3-hint');
    if (!menu) return;

    var rows = [];
    var active = 0;

    ITEMS.forEach(function (item, i) {
      // <a> owns the hit area and stays axis-aligned - no rotation applied
      // here, so hitboxes never overlap their neighbors regardless of how
      // much the visual layer leans.
      var a = document.createElement('a');
      a.className = 'p3-row';
      a.href = item.href;
      if (item.external) {
        a.target = '_blank';
        a.rel = 'noopener';
      }
      a.style.marginLeft = item.offsetX + 'px';
      a.style.marginTop  = item.offsetY + 'px';

      // .p3-rot is the VISUAL layer. It carries the tilt and the translateY
      // shift. CSS transforms don't affect layout, so the <a>'s hit area
      // stays the same whether or not .p3-rot leans or slides.
      var rot = document.createElement('div');
      rot.className = 'p3-rot';
      rot.style.setProperty('--tilt',  (item.tilt  || 0) + 'deg');
      rot.style.setProperty('--shift', (item.shift || 0) + 'px');

      // Kept in the DOM for backward-compatible styling - hidden by CSS.
      var hl = document.createElement('div');
      hl.className = 'p3-highlight';

      var label = document.createElement('span');
      label.className = 'p3-label';
      label.style.fontSize = item.fontSize + 'px';
      label.textContent = item.label;

      // SVG composition wrapper. Filled by buildHeroSvg once the label has
      // been measured (after fonts load).
      var hero = document.createElement('div');
      hero.className = 'p3-hero';

      rot.appendChild(hl);
      rot.appendChild(label);
      rot.appendChild(hero);
      a.appendChild(rot);

      // Unified "highlighted index": hover, keyboard focus, click, and arrow
      // keys all drive the same `active` state.
      a.addEventListener('mouseenter', function () { setActive(i); });
      a.addEventListener('focus',      function () { setActive(i); });
      a.addEventListener('click',      function () { setActive(i); });

      menu.appendChild(a);
      rows.push({ row: a, rot: rot, highlight: hl, label: label, hero: hero, item: item, heroBuilt: false });
    });

    function buildAllHeroes() {
      rows.forEach(function (entry) {
        buildHeroSvg(entry);
      });
    }

    function updateStyles() {
      rows.forEach(function (entry, i) {
        var dist = Math.abs(i - active);
        var opacity = (i === active) ? 1 : Math.max(0.18, 1 - dist * 0.38);
        entry.row.style.setProperty('--p3-idle-opacity', opacity);
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
      // Ignore keys typed into form fields (e.g. the chat widget's input) , 
      // Enter there means "send the message", not "open the hovered page".
      var t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable ||
                (t.closest && t.closest('.ycchat-panel')))) return;
      var heroRect = root.getBoundingClientRect();
      var visible  = heroRect.bottom > 120 && heroRect.top < window.innerHeight * 0.6;
      if (!visible) return;
      if (e.key === 'ArrowUp')   { e.preventDefault(); setActive(active - 1); }
      if (e.key === 'ArrowDown') { e.preventDefault(); setActive(active + 1); }
      if (e.key === 'Enter')     { e.preventDefault(); activateCurrent(); }
    });

    updateStyles();

    // Build SVGs after labels have laid out at least once.
    function whenReady() {
      buildAllHeroes();
    }
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(whenReady).catch(whenReady);
    } else {
      setTimeout(whenReady, 60);
    }

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
