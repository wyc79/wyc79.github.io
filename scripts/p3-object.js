(function () {
  'use strict';

  // The homepage stone (drawn by stone-renderer.js) and the metadata under
  // it. Three jobs, one control:
  //   - it follows the P3 selection: p3-menu.js announces the highlighted
  //     row with a `p3-select` event and each row has a pose below;
  //   - dragging it sideways is a second way to move the selection: it scrubs
  //     through the poses with a detent at each row and asks the menu to
  //     select (`p3-request`);
  //   - a click or tap on it opens Ask AI (window.YCChat, chat-widget.js).
  // It never navigates; opening a page stays with the menu's click and Enter.
  //
  // Without WebGL the stone is skipped: the metadata still follows the menu
  // and the launcher shows as the plain text Ask AI button.

  var figure = document.querySelector('.p3-object');
  if (!figure) return;
  var launcher = figure.querySelector('.p3-stone');
  var canvas = figure.querySelector('canvas');
  var indexEl = figure.querySelector('.p3-object-index');
  var nameEl  = figure.querySelector('.p3-object-name');

  // ---------------------------------------------------------------- states
  // One pose per menu row, keyed by row id. Neighbours differ by roughly
  // 12-16 deg of yaw, a few degrees of pitch and orbit, and a small shift
  // of the surface and key light.
  //   yaw, pitch  - object rotation (deg)
  //   warp        - phase of the surface deformation
  //   tilt, roll  - orbit plane: lean toward the viewer, then screen roll (deg)
  //   light       - key light rotated about the view axis (deg)
  var STATES = {
    projects:     { yaw:   0, pitch: -6, warp: 0.0, tilt: 27, roll: -10, light:  0 },
    skills:       { yaw:  14, pitch: -2, warp: 0.5, tilt: 29, roll:  -8, light:  3 },
    education:    { yaw:  28, pitch: -8, warp: 1.0, tilt: 26, roll: -11, light: -2 },
    publications: { yaw:  41, pitch: -3, warp: 1.4, tilt: 30, roll:  -7, light:  4 },
    agents:       { yaw:  55, pitch: -7, warp: 1.9, tilt: 27, roll: -10, light: -3 },
    toolbox:      { yaw:  68, pitch: -1, warp: 2.3, tilt: 29, roll:  -8, light:  2 },
    github:       { yaw:  80, pitch: -5, warp: 2.8, tilt: 28, roll: -11, light: -1 }
  };
  // The dot walks the orbit's front arc, left to right, as the selection
  // moves down the menu (deg along the orbit; 90 is nearest the viewer).
  var MARKER_FROM = 150, MARKER_TO = 30;
  var MARKER_SPAN = MARKER_TO - MARKER_FROM;

  // Pose spring: critically damped, velocity kept across retargets, so a
  // hover that changes its mind turns the stone round instead of restarting.
  // ~95% of a step in 2.4 x POSE_SMOOTH_S (~0.36 s).
  var POSE_SMOOTH_S = 0.15;
  var IDLE_ORBIT_S  = 2.5;   // settled this long, the dot starts a slow orbit...
  var HOME_ORBIT_S  = 24;    // ...of this period; any interaction calls it back

  // Drag scrubbing. Distances are CSS px of pointer travel.
  var DRAG_START_PX = 9;     // below this a press is a tap: it opens Ask AI
  var DRAG_INTENT   = 1.25;  // |dx| must be this many times |dy|
  var STEP_PX       = { mouse: 68, touch: 56 };   // per row
  var ADVANCE       = 0.55;  // fraction of a step that switches rows...
  var RETURN        = 0.65;  // ...and that switches back (35% hysteresis)
  var MAGNET_PX     = 12;    // near a detent, the form lags the pointer...
  var MAGNET_MIN    = 0.45;  // ...moving at this rate at the detent itself
  var OVERSCROLL_PX = 20;    // elastic give past the first and last rows
  var MOMENTUM_S    = 0.11;  // release velocity projected this far ahead
  var FLICK_MIN     = 1.5;   // rows/s; slower releases keep the highlighted row
  var SETTLE_S      = 0.22;  // release to rest, no overshoot

  function pad(n) { return (n < 10 ? '0' : '') + n; }

  // ---------------------------------------------------------------- stone
  var stone = null;
  function fallback() {
    if (stone) { stone.destroy(); stone = null; }
    figure.classList.remove('is-ready');
    figure.classList.add('no-stone');
  }
  if (canvas && window.YCStone) stone = window.YCStone.create(canvas, { onLost: fallback });
  if (stone) figure.classList.add('is-ready');
  else fallback();

  // ---------------------------------------------------------------- state
  // The pose is one number, `pos`: a fractional row index. Whole numbers are
  // the rows' own poses and fractions blend neighbours, so moving between
  // rows - sprung, dragged or settling - passes through the poses in between.
  //   spring - pos follows goalPos (the final hover goal or the exact row)
  //   drag   - pos follows the pointer, with detents and elastic ends
  //   settle - pos glides from release to the final row, then springs again
  var ids = null;            // row ids in menu order, from p3-menu.js
  var poses = null;
  var menuIndex = 0;         // the row the menu has highlighted
  var pos = 0, vel = 0, goalPos = 0, mode = 'spring', settle = null;
  var sweptIndex = -1, sweepPending = false;
  var hovered = false, chatOpen = false, orbiting = false, lastActive = 0;

  function now() { return performance.now() / 1000; }
  function still() { return stone ? stone.reducedMotion() : true; }
  function lastIndex() { return ids ? ids.length - 1 : 0; }
  function clampIndex(i) { return Math.max(0, Math.min(lastIndex(), i)); }
  function markerAngle(p) { var n = lastIndex(); return MARKER_FROM + MARKER_SPAN * (n ? p / n : 0); }
  function touch() { lastActive = now(); orbiting = false; }

  window.addEventListener('p3-select', function (e) {
    var d = e.detail || {};
    var first = !ids;
    if (indexEl) indexEl.textContent = d.ordinal ? pad(d.ordinal) + ' / ' + pad(d.sections) : '↗';
    if (nameEl)  nameEl.textContent  = d.label || '';
    if (!stone) return;

    if (first) {
      ids = d.ids || ['projects'];
      poses = ids.map(function (id) { return STATES[id] || STATES.projects; });
      stone.setNotches(ids.map(function (_, i) { return markerAngle(i); }));
    }
    menuIndex = d.index || 0;
    if (d.source === 'lang') return;   // same row, new words

    if (d.source !== 'stone') {
      // Keys, focus, clicks and hover win over the stone, mid-drag included.
      // A hover heads straight for the row under the pointer (d.goal) while
      // the titles step through the rows in between.
      if (drag) endDrag(false, false);
      if (mode !== 'spring') { mode = 'spring'; settle = null; }
      goalPos = typeof d.goal === 'number' ? d.goal : menuIndex;
    }
    if (first || still() && !drag) { pos = goalPos = menuIndex; vel = 0; mode = 'spring'; settle = null; }
    touch();

    // One contour sweep per change, once the stone has reached the row the
    // interaction settled on - not for every row a hover or drag passes.
    if (first) sweptIndex = menuIndex;
    else if (d.settled && menuIndex !== sweptIndex) { sweptIndex = menuIndex; sweepPending = true; }
    stone.requestFrame();
  });

  window.addEventListener('ycchat-change', function (e) {
    chatOpen = !!(e.detail && e.detail.open);
    if (!stone) return;
    // Open: the dot glides to the front and waits, as on every page. Closed:
    // it returns to the selection, and the idle orbit can start again later.
    if (chatOpen) { stone.rest(stone.FRONT); stone.drift(false); }
    else { touch(); stone.drift(true); }
    stone.requestFrame();
  });

  function request(phase, index) {
    window.dispatchEvent(new CustomEvent('p3-request', { detail: { phase: phase, index: index } }));
  }

  function update(dt) {
    if (!ids) return false;
    var busy = false;
    if (mode === 'settle') {
      var u = (now() - settle.t0) / SETTLE_S;
      if (u >= 1) { pos = settle.to; vel = 0; mode = 'spring'; settle = null; }
      else {
        var u2 = u * u, u3 = u2 * u;
        pos = (2 * u3 - 3 * u2 + 1) * settle.from + (u3 - 2 * u2 + u) * settle.m + (3 * u2 - 2 * u3) * settle.to;
      }
      busy = true;
    } else if (mode === 'spring') {
      if (still()) { pos = goalPos; vel = 0; }
      else {
        var s = window.YCStone.smoothDamp(pos, goalPos, vel, POSE_SMOOTH_S, dt);
        pos = s[0]; vel = s[1];
        if (Math.abs(goalPos - pos) < 1e-4 && Math.abs(vel) < 1e-3) { pos = goalPos; vel = 0; }
        else busy = true;
      }
    } else {
      busy = true;   // drag
    }
    if (drag) touch();
    var resting = mode === 'spring' && pos === goalPos && goalPos === menuIndex;

    if (sweepPending && resting) { sweepPending = false; stone.sweep(); }

    // The dot: tracks the selection; after IDLE_ORBIT_S at rest it drifts
    // off on a slow orbit, and the next interaction glides it back.
    if (!chatOpen) {
      if (hovered) touch();
      if (!orbiting) {
        stone.follow(markerAngle(pos));
        if (resting && !still() && now() - lastActive > IDLE_ORBIT_S) { orbiting = true; stone.orbit(HOME_ORBIT_S); }
        else if (!still()) busy = true;
      }
    }
    stone.showNotches(hovered || !!drag);
    stone.setPose(window.YCStone.posePath(poses, pos));
    return busy;
  }
  if (stone) { stone.onUpdate(update); stone.drift(true); }

  // ---------------------------------------------------------------- drag
  // Press on the canvas, then move sideways: left steps down the menu, right
  // steps up. Under DRAG_START_PX it is a tap and opens Ask AI; mostly
  // vertical, it is left to the page (touch-action: pan-y). Only the first
  // pointer counts.
  var press = null;   // { id, x, y, type, gone } between pointerdown and drag start
  var drag = null;    // { id, x0, from, step, raw, sel, dir, samples }
  var swallowClick = 0;   // until then, the launcher's next click is ours

  function rubber(px) { return OVERSCROLL_PX * (1 - Math.exp(-px / OVERSCROLL_PX)); }

  // Pointer position (in rows) to drawn position: elastic past the ends, and
  // slowed within MAGNET_PX of a detent so each row feels like it holds.
  function dragVisual(raw, stepPx) {
    var max = lastIndex();
    if (raw < 0)   return -rubber(-raw * stepPx) / stepPx;
    if (raw > max) return max + rubber((raw - max) * stepPx) / stepPx;
    var n = Math.round(raw), d = (raw - n) * stepPx, a = Math.abs(d);
    if (a < MAGNET_PX) d *= MAGNET_MIN + (1 - MAGNET_MIN) * a / MAGNET_PX;
    return n + d / stepPx;
  }

  if (stone) {
    canvas.addEventListener('pointerdown', function (e) {
      if (press || drag || !e.isPrimary || e.button !== 0 || !ids) return;
      // Mouse: no text selection, native image drag or focus change. Touch
      // keeps its default so a vertical swipe still scrolls.
      if (e.pointerType === 'mouse') e.preventDefault();
      press = { id: e.pointerId, x: e.clientX, y: e.clientY, type: e.pointerType };
    });

    canvas.addEventListener('pointermove', function (e) {
      if (drag) {
        if (e.pointerId === drag.id) moveDrag(e.clientX);
        return;
      }
      if (!press || press.gone || e.pointerId !== press.id) return;
      var dx = e.clientX - press.x, dy = e.clientY - press.y;
      if (Math.hypot(dx, dy) < DRAG_START_PX) return;
      // Mostly vertical: not a drag, and too far to be a tap either.
      if (Math.abs(dx) < DRAG_INTENT * Math.abs(dy)) { press.gone = true; return; }
      startDrag(e);
    });

    canvas.addEventListener('pointerup', function (e) {
      if (drag && e.pointerId === drag.id) endDrag(true, true);
      else if (press && e.pointerId === press.id) {
        // A tap opens (or closes) Ask AI here rather than waiting for the
        // click, which a touch that moved a few px may never produce. The
        // click that does follow is swallowed so it cannot toggle twice -
        // as is the click after a press that wandered off vertically.
        var tap = !press.gone;
        press = null;
        swallowClick = performance.now() + 600;
        if (tap && window.YCChat) window.YCChat.toggle();
      }
    });
    var cancel = function (e) {
      if (drag && e.pointerId === drag.id) endDrag(true, false);
      else if (press && e.pointerId === press.id) press = null;
    };
    canvas.addEventListener('pointercancel', cancel);
    canvas.addEventListener('lostpointercapture', cancel);

    canvas.addEventListener('pointerenter', function (e) { if (e.pointerType === 'mouse') { hovered = true; stone.requestFrame(); } });
    canvas.addEventListener('pointerleave', function (e) { if (e.pointerType === 'mouse') { hovered = false; stone.requestFrame(); } });

    // Clicks on the launcher that a pointer gesture already handled - the
    // tap above, or the compatibility click after a drag - stop here, before
    // the launcher's own click handler (chat-widget.js) sees them. Keyboard
    // activation (Enter / Space) reaches it untouched.
    window.addEventListener('click', function (e) {
      if (performance.now() < swallowClick && launcher && launcher.contains(e.target)) {
        swallowClick = 0;
        e.preventDefault();
        e.stopPropagation();
      }
    }, true);
  }

  function startDrag(e) {
    var stepPx = press.type === 'mouse' ? STEP_PX.mouse : STEP_PX.touch;
    try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* already gone */ }
    drag = { id: e.pointerId, x0: press.x, from: menuIndex, step: stepPx,
             raw: menuIndex, sel: menuIndex, dir: 0, samples: [] };
    press = null;
    mode = 'drag';
    settle = null;
    vel = 0;
    touch();
    figure.classList.add('is-dragging');
    request('start');
    moveDrag(e.clientX);
  }

  function moveDrag(x) {
    var max = lastIndex();
    drag.raw = drag.from - (x - drag.x0) / drag.step;
    var t = now();
    drag.samples.push({ t: t, raw: drag.raw });
    while (drag.samples.length > 2 && t - drag.samples[0].t > 0.1) drag.samples.shift();

    // Discrete row with hysteresis: switching on in the same direction
    // takes ADVANCE of a step, turning back takes RETURN.
    var r = Math.max(0, Math.min(max, drag.raw)), sel = drag.sel;
    for (;;) {
      if (sel < max && r >= sel + (drag.dir < 0 ? RETURN : ADVANCE)) { sel++; drag.dir = 1; }
      else if (sel > 0 && r <= sel - (drag.dir > 0 ? RETURN : ADVANCE)) { sel--; drag.dir = -1; }
      else break;
    }
    if (sel !== drag.sel) { drag.sel = sel; request('move', sel); }

    pos = still() ? drag.sel : dragVisual(drag.raw, drag.step);
    stone.requestFrame();
  }

  function velocity(d) {
    var s = d.samples, a = s[0], b = s[s.length - 1];
    return s.length > 1 && b.t > a.t ? (b.raw - a.raw) / (b.t - a.t) : 0;
  }

  // Release: pick the resting row, glide there, and tell the menu. A flick
  // carries MOMENTUM_S of its speed, but never more than one row past where
  // the pointer let go; a slow release, or a cancelled pointer, keeps the
  // row the menu already shows. Nothing navigates and nothing opens.
  //   release  - false when the menu has already moved on (key, focus)
  //   momentum - false for pointercancel / lost capture
  function endDrag(release, momentum) {
    var d = drag;
    drag = null;
    figure.classList.remove('is-dragging');
    try { canvas.releasePointerCapture(d.id); } catch (err) { /* not captured */ }
    swallowClick = performance.now() + 400;   // the compatibility click after a drag
    touch();
    if (!release) { mode = 'spring'; return; }   // the menu moved on (key, focus): just let go

    var max = lastIndex(), fin = d.sel, v = momentum && !still() ? velocity(d) : 0;
    if (Math.abs(v) > FLICK_MIN) {
      var near = Math.max(0, Math.min(max, Math.round(d.raw)));
      var proj = Math.round(d.raw + v * MOMENTUM_S);
      proj = Math.max(near - 1, Math.min(near + 1, proj));
      if ((proj - d.sel) * v > 0) fin = clampIndex(proj);
    }
    startSettle(fin, v);
    request('end', fin);
  }

  // Cubic Hermite from (pos, velocity) to the row at rest. The start speed
  // is capped so the curve never passes the target: no overshoot, no bounce.
  function startSettle(to, v) {
    var dist = to - pos;
    goalPos = to;
    vel = 0;
    if (still() || Math.abs(dist) < 1e-4) { pos = to; mode = 'spring'; settle = null; stone.requestFrame(); return; }
    var m = v * SETTLE_S / dist;              // start slope, in units of dist
    m = Math.max(0, Math.min(3, m));
    settle = { from: pos, to: to, m: m * dist, t0: now() };
    mode = 'settle';
    stone.requestFrame();
  }
})();
