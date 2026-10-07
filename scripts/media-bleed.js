// Projects page: where a whole snapshot (object-fit: contain) leaves its frame
// unfilled, each gap gets two layers that styles.css blends three ways (see
// .project-media .bleed): the image's adjacent edge column (or row) alone,
// stretched out from the seam; the far end of the image, carried on
// scroll-style; and transparent, so the plate shows. This script only supplies
// the layers and the image's rendered geometry. Purely presentational: without
// it the plate shows.
(function () {
  'use strict';

  var LAYERS = ['wrap bleed--start', 'clamp bleed--start', 'wrap bleed--end', 'clamp bleed--end'];

  function ensureLayers(box) {
    if (box.querySelector('.bleed')) return;
    LAYERS.forEach(function (cls) {
      var span = document.createElement('span');
      span.className = 'bleed bleed--' + cls;
      span.setAttribute('aria-hidden', 'true');
      box.appendChild(span);
    });
  }

  function fit(box, img) {
    var W = box.clientWidth, H = box.clientHeight;
    var nw = img.naturalWidth, nh = img.naturalHeight;
    if (!W || !H || !nw || !nh) return;
    // contain: a frame wider than the image leaves side gaps, else top/bottom.
    var wide = W / H > nw / nh;
    var w = wide ? H * nw / nh : W;
    var h = wide ? H : W * nh / nw;
    var s = box.style;
    s.setProperty('--img-w', w + 'px');
    s.setProperty('--img-h', h + 'px');
    s.setProperty('--img-nw', nw);
    s.setProperty('--img-nh', nh);
    s.setProperty('--img-left', (W - w) / 2 + 'px');
    s.setProperty('--img-top', (H - h) / 2 + 'px');
    s.setProperty('--bleed-src', 'url("' + (img.currentSrc || img.src).replace(/"/g, '%22') + '")');
    ensureLayers(box);
    box.classList.toggle('bleed-x', wide && W - w >= 1);
    box.classList.toggle('bleed-y', !wide && H - h >= 1);
  }

  function init() {
    var ro = 'ResizeObserver' in window ? new ResizeObserver(function (entries) {
      entries.forEach(function (e) {
        var img = e.target.querySelector('img');
        if (img && img.complete) fit(e.target, img);
      });
    }) : null;
    document.querySelectorAll('.project-media').forEach(function (box) {
      var img = box.querySelector('img');
      if (!img) return;
      if (img.complete) fit(box, img);
      else img.addEventListener('load', function () { fit(box, img); });
      if (ro) ro.observe(box);
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
