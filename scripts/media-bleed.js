// Projects page: where a whole snapshot (object-fit: contain) leaves its frame
// unfilled, paint a short bleed into each gap, on a canvas behind the image.
// From the seam outward, over --media-bleed and eased:
//   1. blend - the image's edge column (row) at the seam, smoothed along it,
//      mixed 0 -> 1 into the image's far end, carried on as if the snapshot
//      were a scrolling strip (the left gap shows its rightmost columns, the
//      right gap its leftmost; top and bottom likewise);
//   2. fade  - the blended result's alpha runs 1 -> 0, so the plate shows.
// The frame clips the bleed where the gap is narrower. Purely presentational:
// without it the plate shows.
(function () {
  'use strict';

  var TUCK = 2;        // px the bleed starts under the image, covering the seam
  var EDGE_STEP = 12;  // px of seam per averaged edge sample (smoother if larger)
  var FAR_SOFT = 4;    // the far end is drawn at 1/FAR_SOFT resolution (softer if larger)

  var scratch = document.createElement('canvas');

  function sized(w, h) {
    var c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    var ctx = c.getContext('2d');
    ctx.imageSmoothingQuality = 'high';
    return c;
  }

  // Gradient along p0 -> p1 whose alpha eases (smoothstep) from a to b.
  function ramp(ctx, p0, p1, a, b) {
    var g = ctx.createLinearGradient(p0[0], p0[1], p1[0], p1[1]);
    for (var i = 0; i <= 10; i++) {
      var s = i / 10, e = s * s * (3 - 2 * s);
      g.addColorStop(s, 'rgba(0,0,0,' + (a + (b - a) * e).toFixed(3) + ')');
    }
    return g;
  }

  // The source's one-pixel edge line (src = [sx, sy, sw, sh], sw or sh is 1),
  // averaged down to n samples by repeated halving.
  function edgeLine(img, src, n) {
    var vertical = src[2] === 1;
    var len = vertical ? src[3] : src[2];
    var c = sized(vertical ? 1 : len, vertical ? len : 1);
    c.getContext('2d').drawImage(img, src[0], src[1], src[2], src[3], 0, 0, c.width, c.height);
    while (len > n) {
      len = Math.max(n, Math.ceil(len / 2));
      var d = sized(vertical ? 1 : len, vertical ? len : 1);
      d.getContext('2d').drawImage(c, 0, 0, d.width, d.height);
      c = d;
    }
    return c;
  }

  // The two gaps beside the contained image, in box px: the band each paints
  // (from TUCK under the image out to the bleed's end), the fade's seam and
  // outer end, the source's edge line, and the source's far end with where it
  // lands (it meets the seam with its last column or row).
  function gaps(W, H, nw, nh, E) {
    var wide = W / H > nw / nh;
    var w = wide ? H * nw / nh : W, h = wide ? H : W * nh / nw;
    var L = (W - w) / 2, T = (H - h) / 2;
    if (wide ? W - w < 1 : H - h < 1) return [];
    if (wide) {
      var kx = nw / w, fw = Math.min(E, w);
      return [
        { band: [L - E, T, E + TUCK, h], fade: [[L, 0], [L - E, 0]], edge: [0, 0, 1, nh],
          far: [nw - fw * kx, 0, fw * kx, nh], at: [L - fw, T, fw, h] },
        { band: [L + w - TUCK, T, E + TUCK, h], fade: [[L + w, 0], [L + w + E, 0]], edge: [nw - 1, 0, 1, nh],
          far: [0, 0, fw * kx, nh], at: [L + w, T, fw, h] }
      ];
    }
    var ky = nh / h, fh = Math.min(E, h);
    return [
      { band: [L, T - E, w, E + TUCK], fade: [[0, T], [0, T - E]], edge: [0, 0, nw, 1],
        far: [0, nh - fh * ky, nw, fh * ky], at: [L, T - fh, w, fh] },
      { band: [L, T + h - TUCK, w, E + TUCK], fade: [[0, T + h], [0, T + h + E]], edge: [0, nh - 1, nw, 1],
        far: [0, 0, nw, fh * ky], at: [L, T + h, w, fh] }
    ];
  }

  function paint(box, img, canvas) {
    var W = box.clientWidth, H = box.clientHeight;
    var nw = img.naturalWidth, nh = img.naturalHeight;
    var dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(W * dpr);   // resizing also clears it
    canvas.height = Math.round(H * dpr);
    if (!W || !H || !nw || !nh) return;
    var E = parseFloat(getComputedStyle(box).getPropertyValue('--media-bleed')) || 40;
    var out = canvas.getContext('2d');

    gaps(W, H, nw, nh, E).forEach(function (g) {
      var b = g.band;
      // Device pixels of the band that fall inside the frame.
      var x0 = Math.max(0, Math.floor(b[0] * dpr)), y0 = Math.max(0, Math.floor(b[1] * dpr));
      var x1 = Math.min(canvas.width, Math.ceil((b[0] + b[2]) * dpr));
      var y1 = Math.min(canvas.height, Math.ceil((b[1] + b[3]) * dpr));
      if (x1 - x0 < 1 || y1 - y0 < 1) return;
      scratch.width = x1 - x0;
      scratch.height = y1 - y0;
      var ctx = scratch.getContext('2d');
      ctx.imageSmoothingQuality = 'high';
      ctx.setTransform(dpr, 0, 0, dpr, -x0, -y0);   // draw in box px
      var all = [x0 / dpr, y0 / dpr, scratch.width / dpr, scratch.height / dpr];

      // 1. Blend: the far end weighted 0 -> 1 outward, over the edge line.
      var soft = sized(Math.max(1, Math.round(g.at[2] / FAR_SOFT)), Math.max(1, Math.round(g.at[3] / FAR_SOFT)));
      soft.getContext('2d').drawImage(img, g.far[0], g.far[1], g.far[2], g.far[3], 0, 0, soft.width, soft.height);
      ctx.drawImage(soft, g.at[0], g.at[1], g.at[2], g.at[3]);
      ctx.globalCompositeOperation = 'destination-in';
      ctx.fillStyle = ramp(ctx, g.fade[0], g.fade[1], 0, 1);
      ctx.fillRect(all[0], all[1], all[2], all[3]);
      ctx.globalCompositeOperation = 'destination-over';
      var along = g.edge[2] === 1 ? b[3] : b[2];
      ctx.drawImage(edgeLine(img, g.edge, Math.max(2, Math.round(along / EDGE_STEP))), b[0], b[1], b[2], b[3]);

      // 2. Fade: the blend's alpha 1 -> 0 outward.
      ctx.globalCompositeOperation = 'destination-in';
      ctx.fillStyle = ramp(ctx, g.fade[0], g.fade[1], 1, 0);
      ctx.fillRect(all[0], all[1], all[2], all[3]);

      out.drawImage(scratch, x0, y0);
    });
  }

  function canvasFor(box) {
    var c = box.querySelector('canvas.bleed');
    if (!c) {
      c = document.createElement('canvas');
      c.className = 'bleed';
      c.setAttribute('aria-hidden', 'true');
      box.insertBefore(c, box.firstChild);
    }
    return c;
  }

  function fit(box, img) {
    if (img.complete && img.naturalWidth) paint(box, img, canvasFor(box));
  }

  function init() {
    var ro = 'ResizeObserver' in window ? new ResizeObserver(function (entries) {
      entries.forEach(function (e) {
        var img = e.target.querySelector('img');
        if (img) fit(e.target, img);
      });
    }) : null;
    document.querySelectorAll('.project-media').forEach(function (box) {
      var img = box.querySelector('img');
      if (!img) return;
      fit(box, img);
      img.addEventListener('load', function () { fit(box, img); });
      if (ro) ro.observe(box);
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
