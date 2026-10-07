// Projects page: where a whole snapshot (object-fit: contain) leaves its frame
// unfilled, hand the frame the image's rendered geometry so styles.css can
// carry the image on past its edges, scroll-style (the left gap shows its right
// end and vice versa), fading into the plate over --media-bleed (see
// .project-media). Purely presentational: without it the plate shows.
(function () {
  'use strict';

  function fit(box, img) {
    var W = box.clientWidth, H = box.clientHeight;
    var r = img.naturalWidth / img.naturalHeight;
    if (!W || !H || !r) return;
    // contain: a frame wider than the image leaves side gaps, else top/bottom.
    var wide = W / H > r;
    var w = wide ? H * r : W;
    var h = wide ? H : W / r;
    var s = box.style;
    s.setProperty('--img-w', w + 'px');
    s.setProperty('--img-h', h + 'px');
    s.setProperty('--img-left', (W - w) / 2 + 'px');
    s.setProperty('--img-top', (H - h) / 2 + 'px');
    s.setProperty('--bleed-src', 'url("' + (img.currentSrc || img.src).replace(/"/g, '%22') + '")');
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
