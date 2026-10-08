(function () {
  'use strict';

  // The stone: one small matte form inside a red orbit, with a red dot on the
  // orbit. It is the site's Ask AI mark - the homepage draws it large and
  // lets it steer the P3 selection (p3-object.js); every other page draws it
  // small as the chat launcher (YCStone.ambient below, created by
  // chat-widget.js). Both go through this one renderer, so the geometry,
  // material, lighting and orbit are the same everywhere.
  //
  //   var stone = YCStone.create(canvas, { onLost: fn });   // null on failure
  //   stone.setPose(pose)          // { yaw, pitch, warp, tilt, roll, light }
  //   stone.follow(angle)          // dot tracks a controller-given angle
  //   stone.orbit(periodS)         // dot circles on its own
  //   stone.rest(angle)            // dot glides to an angle and stops
  //   stone.place(angle)           // dot starts here (no glide)
  //   stone.boost(on)              // orbit a little faster (hover / focus)
  //   stone.drift(on)              // near-imperceptible idle yaw
  //   stone.setNotches(angles); stone.showNotches(on)
  //   stone.sweep()                // one contour pass across the form
  //   stone.onUpdate(fn)           // fn(dt) each frame; return true to keep going
  //   stone.reducedMotion(); stone.requestFrame(); stone.destroy()
  //
  // Frames run only while something is moving: an orbiting dot, the idle
  // drift, a sweep, or a controller that says it is busy. Reduced motion
  // stops all of it - the stone redraws only when its state changes.

  if (window.YCStone) return;

  // ---------------------------------------------------------------- geometry
  var VIEW_HALF = 2.0;     // world half-extent the canvas shows
  var ORBIT_R   = 1.88;    // orbit radius; the form's radius is ~1
  var FRONT     = 90;      // dot angle (deg along the orbit) nearest the viewer
  var SWEEP_S   = 0.75;    // contour sweep
  var BLEND_S   = 0.5;     // dot glide when control changes hands
  var RAMP_S    = 0.8;     // orbit speed ramp (time constant)
  var BOOST     = 1.6;     // orbit speed while hovered / focused
  var DRIFT_DEG = 1.6, DRIFT_S = 16;   // idle yaw: amplitude, period

  // A neutral pose for pages with no selection to show.
  var NEUTRAL = { yaw: 22, pitch: -5, warp: 0.6, tilt: 27, roll: -10, light: 0 };
  var POSE_KEYS = ['yaw', 'pitch', 'warp', 'tilt', 'roll', 'light'];

  // ---------------------------------------------------------------- shaders
  var FORM_VS = [
    'attribute vec3 aPos;',
    'uniform mat4 uProj, uView, uModel;',
    'uniform float uWarp;',
    'varying vec3 vWorld;',
    'varying vec3 vNormal;',
    'varying vec3 vDir;',
    // A unit sphere pushed out by two low-frequency waves and one soft lobe:
    // a rounded, slightly uneven volume rather than a ball.
    'float radius(vec3 n) {',
    '  float r = 1.0;',
    '  r += 0.08 * sin(2.3 * n.x + 1.7 * n.y + uWarp) * cos(1.9 * n.z - 0.8 * n.y + 0.6 * uWarp);',
    '  r += 0.05 * sin(2.9 * n.z + 1.1 * n.x - 0.7 * uWarp);',
    '  float l = max(dot(n, normalize(vec3(0.55, 0.35, 0.75))), 0.0);',
    '  r += 0.2 * l * l * l;',
    '  return r;',
    '}',
    'vec3 surf(vec3 n) { vec3 p = n * radius(n); p.y *= 0.94; return p; }',
    'void main() {',
    '  vec3 n = normalize(aPos);',
    '  vec3 t1 = normalize(cross(n, abs(n.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));',
    '  vec3 t2 = cross(n, t1);',
    '  vec3 p  = surf(n);',
    '  vec3 N  = normalize(cross(surf(normalize(n + 0.01 * t1)) - p, surf(normalize(n + 0.01 * t2)) - p));',
    '  if (dot(N, n) < 0.0) N = -N;',
    '  vec4 w = uModel * vec4(p, 1.0);',
    '  vWorld = w.xyz;',
    '  vNormal = mat3(uModel) * N;',
    '  vDir = n;',
    '  gl_Position = uProj * uView * w;',
    '}'
  ].join('\n');

  // Matte: wrapped diffuse over a three-stop ramp, a faint bounce from below,
  // no specular. During a sweep, a sparse latitude/longitude contour shows in
  // a band that crosses the form, then goes.
  function formFS(lines) {
    return [
      lines ? '#extension GL_OES_standard_derivatives : enable' : '',
      'precision mediump float;',
      'varying vec3 vWorld;',
      'varying vec3 vNormal;',
      'varying vec3 vDir;',
      'uniform vec3 uLight;',
      'uniform vec3 uLit, uMid, uShade, uBounce, uLine;',
      'uniform vec2 uSweep;',   // x: band position, y: strength
      'void main() {',
      '  vec3 N = normalize(vNormal);',
      '  float d = clamp((dot(N, uLight) + 0.3) / 1.3, 0.0, 1.0);',
      '  vec3 col = mix(uShade, uMid, smoothstep(0.0, 0.55, d));',
      '  col = mix(col, uLit, smoothstep(0.45, 1.0, d));',
      '  col += uBounce * max(dot(N, normalize(vec3(0.3, -1.0, 0.35))), 0.0);',
      lines ? [
        '  if (uSweep.y > 0.001) {',
        '    vec2 g = vec2(asin(clamp(vDir.y, -1.0, 1.0)) * 3.183, atan(vDir.z, vDir.x) * 2.546);',
        '    vec2 f = abs(fract(g - 0.5) - 0.5) / fwidth(g);',
        '    float line = 1.0 - min(min(f.x, f.y), 1.0);',
        '    float s = dot(vWorld, vec3(0.8, -0.6, 0.0));',
        '    float band = exp(-pow((s - uSweep.x) / 0.4, 2.0));',
        '    col = mix(col, uLine, line * band * uSweep.y);',
        '  }'
      ].join('\n') : '',
      '  gl_FragColor = vec4(col, 1.0);',
      '}'
    ].join('\n');
  }

  // Flat colour for the orbit, the dot and the resting marks: graphic marks
  // in a rendered scene, depth-tested so the orbit passes behind the form.
  // pos = aA * uScale.x + aB * uScale.y lets the torus keep its stroke width
  // in pixels (aA: point on the orbit, aB: offset across the stroke).
  var FLAT_VS = [
    'attribute vec3 aA;',
    'attribute vec3 aB;',
    'uniform mat4 uProj, uView, uModel;',
    'uniform vec2 uScale;',
    'void main() {',
    '  gl_Position = uProj * uView * uModel * vec4(aA * uScale.x + aB * uScale.y, 1.0);',
    '}'
  ].join('\n');
  var FLAT_FS = [
    'precision mediump float;',
    'uniform vec3 uColor;',
    'uniform float uAlpha;',
    'void main() { gl_FragColor = vec4(uColor * uAlpha, uAlpha); }'   // premultiplied
  ].join('\n');

  // ---------------------------------------------------------------- matrices
  // Column-major 4x4.
  function mul(a, b) {
    var o = new Float32Array(16);
    for (var c = 0; c < 4; c++) {
      for (var r = 0; r < 4; r++) {
        o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
      }
    }
    return o;
  }
  function rotX(d) { var r = d * Math.PI / 180, c = Math.cos(r), s = Math.sin(r);
    return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]); }
  function rotY(d) { var r = d * Math.PI / 180, c = Math.cos(r), s = Math.sin(r);
    return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]); }
  function rotZ(d) { var r = d * Math.PI / 180, c = Math.cos(r), s = Math.sin(r);
    return new Float32Array([c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]); }
  function translate(x, y, z) { return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]); }
  function apply(m, v) {
    return [m[0] * v[0] + m[4] * v[1] + m[8] * v[2],
            m[1] * v[0] + m[5] * v[1] + m[9] * v[2],
            m[2] * v[0] + m[6] * v[1] + m[10] * v[2]];
  }

  // Narrow lens from a distance: near-orthographic, so the orbit's front arc
  // is not exaggerated.
  var FOV = 24 * Math.PI / 180;
  var DIST = VIEW_HALF / Math.tan(FOV / 2);
  var PROJ = (function () {
    var f = 1 / Math.tan(FOV / 2), n = DIST - 4, fa = DIST + 4, nf = 1 / (n - fa);
    return new Float32Array([f, 0, 0, 0, 0, f, 0, 0, 0, 0, (fa + n) * nf, -1, 0, 0, 2 * fa * n * nf, 0]);
  })();
  var VIEW = translate(0, 0, -DIST);

  // ---------------------------------------------------------------- theme
  // Dark: a light form on ink. Light: a mid-grey form on warm paper, darker
  // in shadow, so the volume still reads. The orbit is the brand red in both.
  var PALETTES = {
    dark:  { lit: '#EEEBE3', mid: '#8F8C85', shade: '#2A2926', bounce: '#0E0E0D', line: '#111111' },
    light: { lit: '#D3CFC6', mid: '#8C8981', shade: '#2E2D2A', bounce: '#0B0B0A', line: '#F2F0E9' }
  };
  function rgb(hex) {
    var m = /^#?([0-9a-f]{6})$/i.exec(String(hex).trim());
    var n = m ? parseInt(m[1], 16) : 0xA51C30;
    return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
  }
  function themeColors() {
    var t = document.documentElement.getAttribute('data-theme');
    if (t !== 'light' && t !== 'dark') t = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    var p = PALETTES[t];
    return { lit: rgb(p.lit), mid: rgb(p.mid), shade: rgb(p.shade), bounce: rgb(p.bounce), line: rgb(p.line),
             accent: rgb(getComputedStyle(document.documentElement).getPropertyValue('--accent')) };
  }

  // ---------------------------------------------------------------- helpers
  // Shortest signed difference b - a between two angles, in (-180, 180].
  function angleDiff(a, b) {
    var d = ((b - a) % 360 + 540) % 360 - 180;
    return d === -180 ? 180 : d;
  }
  function easeInOut(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  // Critically damped smoothing toward a moving target that keeps velocity
  // across retargets and never overshoots (the familiar game-engine
  // SmoothDamp). Reaches ~95% of a step in ~2.4 x smoothTime.
  // Returns [value, velocity].
  function smoothDamp(cur, target, vel, smoothTime, dt) {
    if (!(dt > 0)) return [cur, vel];
    var omega = 2 / Math.max(1e-4, smoothTime), x = omega * dt;
    var k = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
    var change = cur - target;
    var temp = (vel + omega * change) * dt;
    var nextVel = (vel - omega * temp) * k;
    var out = target + (change + temp) * k;
    if ((target - cur > 0) === (out > target)) { out = target; nextVel = 0; }
    return [out, nextVel];
  }

  // A pose at fractional position p along a list of poses: whole numbers are
  // the listed poses, fractions blend neighbours, and a little past either
  // end extrapolates (elastic overscroll).
  function posePath(poses, p) {
    var n = poses.length;
    if (n === 1) return poses[0];
    var i0 = clamp(Math.floor(p), 0, n - 2), f = p - i0, a = poses[i0], b = poses[i0 + 1], o = {};
    POSE_KEYS.forEach(function (k) { o[k] = a[k] + (b[k] - a[k]) * f; });
    return o;
  }

  // ---------------------------------------------------------------- create
  function create(canvas, opts) {
    opts = opts || {};
    if (!canvas) return null;
    if (canvas.__ycStone) return canvas.__ycStone;
    if (canvas.__ycStoneFailed) return null;   // never retry a canvas that failed

    var gl = null;
    try {
      gl = canvas.getContext('webgl', { antialias: true, alpha: true, premultipliedAlpha: true });
    } catch (e) { gl = null; }
    if (!gl) { canvas.__ycStoneFailed = true; return null; }
    var lines = !!gl.getExtension('OES_standard_derivatives');

    function compile(type, src) {
      var sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS) && !gl.isContextLost()) {
        console.warn('[stone] shader error:', gl.getShaderInfoLog(sh));
        return null;
      }
      return sh;
    }
    function program(vsSrc, fsSrc, attribs, uniforms) {
      var vs = compile(gl.VERTEX_SHADER, vsSrc), fs = compile(gl.FRAGMENT_SHADER, fsSrc);
      if (!vs || !fs) return null;
      var p = gl.createProgram();
      gl.attachShader(p, vs);
      gl.attachShader(p, fs);
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
        if (!gl.isContextLost()) console.warn('[stone] link error:', gl.getProgramInfoLog(p));
        return null;
      }
      var out = { prog: p, a: {}, u: {} };
      attribs.forEach(function (n) { out.a[n] = gl.getAttribLocation(p, n); });
      uniforms.forEach(function (n) { out.u[n] = gl.getUniformLocation(p, n); });
      return out;
    }
    function release() {
      var ext = gl.getExtension('WEBGL_lose_context');
      if (ext && !gl.isContextLost()) ext.loseContext();
    }

    var form = program(FORM_VS, formFS(lines), ['aPos'],
      ['uProj', 'uView', 'uModel', 'uWarp', 'uLight', 'uLit', 'uMid', 'uShade', 'uBounce', 'uLine', 'uSweep']);
    var flat = program(FLAT_VS, FLAT_FS, ['aA', 'aB'],
      ['uProj', 'uView', 'uModel', 'uScale', 'uColor', 'uAlpha']);
    if (!form || !flat) { release(); canvas.__ycStoneFailed = true; return null; }

    function buffer(target, data) {
      var b = gl.createBuffer();
      gl.bindBuffer(target, b);
      gl.bufferData(target, data, gl.STATIC_DRAW);
      return b;
    }
    // Unit UV sphere: positions double as directions for the form's shader.
    function buildSphere(segments, rings) {
      var pos = [], idx = [];
      for (var y = 0; y <= rings; y++) {
        var phi = y / rings * Math.PI;
        for (var x = 0; x <= segments; x++) {
          var th = x / segments * Math.PI * 2;
          pos.push(Math.sin(phi) * Math.cos(th), Math.cos(phi), Math.sin(phi) * Math.sin(th));
        }
      }
      for (var j = 0; j < rings; j++) {
        for (var i = 0; i < segments; i++) {
          var a = j * (segments + 1) + i, b = a + segments + 1;
          idx.push(a, b, a + 1, b, b + 1, a + 1);
        }
      }
      return { pos: buffer(gl.ARRAY_BUFFER, new Float32Array(pos)),
               idx: buffer(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(idx)), count: idx.length };
    }
    // Torus around the Y axis as (point on unit circle, unit offset across
    // the tube), scaled in the vertex shader.
    function buildOrbit(segments, sides) {
      var a = [], b = [], idx = [];
      for (var i = 0; i <= segments; i++) {
        var t = i / segments * Math.PI * 2, c = Math.cos(t), s = Math.sin(t);
        for (var j = 0; j <= sides; j++) {
          var p = j / sides * Math.PI * 2, cp = Math.cos(p), sp = Math.sin(p);
          a.push(c, 0, s);
          b.push(cp * c, sp, cp * s);
        }
      }
      for (var i2 = 0; i2 < segments; i2++) {
        for (var j2 = 0; j2 < sides; j2++) {
          var k = i2 * (sides + 1) + j2, k2 = k + sides + 1;
          idx.push(k, k2, k + 1, k2, k2 + 1, k + 1);
        }
      }
      return { a: buffer(gl.ARRAY_BUFFER, new Float32Array(a)), b: buffer(gl.ARRAY_BUFFER, new Float32Array(b)),
               idx: buffer(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(idx)), count: idx.length };
    }
    var formMesh   = buildSphere(80, 56);
    var markerMesh = buildSphere(16, 10);
    var orbitMesh  = buildOrbit(192, 6);

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0, 0, 0, 0);

    // ---------------------------------------------------------- state
    var reduceMQ = window.matchMedia ? matchMedia('(prefers-reduced-motion: reduce)') : null;
    function still() { return !!(reduceMQ && reduceMQ.matches); }

    var pose = NEUTRAL;
    var dot = { angle: FRONT, mode: 'rest', target: FRONT, from: FRONT, blend: 1, speed: 0, period: 10, boost: false };
    var notches = [], notchShow = false, notchAlpha = 0;
    var sweepStart = -1, driftOn = false, driftClock = 0;
    var update = null, colors = themeColors();
    var pending = false, inFrame = false, rafId = 0, lastT = 0, dead = false, lost = false;

    // ---------------------------------------------------------- dot
    // follow: the controller supplies the angle every frame. Coming from
    // another mode the dot glides over BLEND_S (shortest way round) instead
    // of jumping, then tracks exactly.
    function takeOver(mode, target) {
      if (dot.mode !== mode) { dot.from = dot.angle; dot.blend = still() ? 1 : 0; dot.mode = mode; }
      dot.target = target;
    }
    function stepDot(dt) {
      var calm = still();
      if (dot.mode === 'orbit') {
        if (calm) return false;   // reduced motion: the dot stays where it is
        dot.speed += ((dot.boost ? BOOST : 1) - dot.speed) * (1 - Math.exp(-dt / RAMP_S));
        dot.angle = ((dot.angle - 360 / dot.period * dot.speed * dt) % 360 + 360) % 360;
        return true;
      }
      if (calm || dot.blend >= 1) { dot.angle = dot.target; dot.blend = 1; return false; }
      dot.blend = Math.min(1, dot.blend + dt / BLEND_S);
      var from = dot.target - angleDiff(dot.from, dot.target);
      dot.angle = from + (dot.target - from) * easeInOut(dot.blend);
      return true;
    }

    // ---------------------------------------------------------- frames
    // Setters ask for a frame; inside a frame they need not - whether to go
    // on is decided by what the frame reports as still moving.
    function requestFrame() {
      if (pending || inFrame || dead || lost || document.hidden) return;
      pending = true;
      rafId = requestAnimationFrame(frame);
    }
    function frame(ms) {
      pending = false;
      if (dead || lost) return;
      var t = ms / 1000, dt = lastT ? Math.min(0.1, t - lastT) : 0;
      lastT = t;
      var calm = still(), busy = false;

      inFrame = true;
      try { if (update && update(dt)) busy = true; } finally { inFrame = false; }
      if (stepDot(dt)) busy = true;

      var notchGoal = notchShow ? 1 : 0;
      if (notchAlpha !== notchGoal) {
        notchAlpha = calm ? notchGoal : notchAlpha + (notchGoal - notchAlpha) * (1 - Math.exp(-dt / 0.06));
        if (Math.abs(notchGoal - notchAlpha) < 0.01) notchAlpha = notchGoal;
        busy = true;
      }
      if (driftOn && !calm) { driftClock += dt; busy = true; }
      var sweep = -1;
      if (sweepStart >= 0 && !calm) {
        sweep = (performance.now() / 1000 - sweepStart) / SWEEP_S;
        if (sweep >= 1) { sweep = -1; sweepStart = -1; } else busy = true;
      }

      draw(sweep);
      if (busy) requestFrame();
    }
    function onVisibility() {
      if (document.hidden) {
        if (pending) cancelAnimationFrame(rafId);
        pending = false;
      } else {
        lastT = 0;   // no catch-up jump after a hidden spell
        requestFrame();
      }
    }
    document.addEventListener('visibilitychange', onVisibility);

    // ---------------------------------------------------------- size
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var cssSize = 1;
    function resize() {
      cssSize = canvas.clientWidth || 1;
      var px = Math.max(1, Math.round(cssSize * dpr));
      if (canvas.width !== px || canvas.height !== px) { canvas.width = px; canvas.height = px; }
      gl.viewport(0, 0, px, px);
      requestFrame();
    }
    var ro = window.ResizeObserver ? new ResizeObserver(resize) : null;
    if (ro) ro.observe(canvas); else window.addEventListener('resize', resize);

    // ---------------------------------------------------------- theme
    function onTheme() { colors = themeColors(); requestFrame(); }
    var mo = new MutationObserver(onTheme);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    var darkMQ = window.matchMedia ? matchMedia('(prefers-color-scheme: dark)') : null;
    function listenMQ(mq, fn, on) {
      if (!mq) return;
      if (mq.addEventListener) mq[on ? 'addEventListener' : 'removeEventListener']('change', fn);
      else if (mq.addListener) mq[on ? 'addListener' : 'removeListener'](fn);
    }
    function onMotion() { lastT = 0; requestFrame(); }
    listenMQ(darkMQ, onTheme, true);
    listenMQ(reduceMQ, onMotion, true);

    // ---------------------------------------------------------- loss
    function onLost() {
      lost = true;
      if (pending) cancelAnimationFrame(rafId);
      pending = false;
      if (typeof opts.onLost === 'function') opts.onLost();
    }
    canvas.addEventListener('webglcontextlost', onLost);

    // ---------------------------------------------------------- draw
    function draw(sweep) {
      var worldPerPx = VIEW_HALF * 2 / cssSize;
      // Marks keep their pixel size on the large stone and thin out a little
      // on the small one, so the launcher does not read as heavier.
      var linePx = clamp(cssSize * 0.0075, 1.1, 1.5);
      var markerPx = clamp(cssSize * 0.018, 2.6, 4.5);
      var notchPx = clamp(cssSize * 0.0067, 1.2, 2);
      var s = pose;

      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      // --- form
      gl.useProgram(form.prog);
      var idle = driftOn || driftClock ? DRIFT_DEG * Math.sin(driftClock * Math.PI * 2 / DRIFT_S) : 0;
      var model = mul(rotX(-s.pitch), rotY(s.yaw + idle));
      var lr = s.light * Math.PI / 180;
      var L = [-0.55, 0.68, 0.48];
      L = [L[0] * Math.cos(lr) - L[1] * Math.sin(lr), L[0] * Math.sin(lr) + L[1] * Math.cos(lr), L[2]];
      var ll = Math.hypot(L[0], L[1], L[2]);
      gl.uniformMatrix4fv(form.u.uProj, false, PROJ);
      gl.uniformMatrix4fv(form.u.uView, false, VIEW);
      gl.uniformMatrix4fv(form.u.uModel, false, model);
      gl.uniform1f(form.u.uWarp, s.warp);
      gl.uniform3f(form.u.uLight, L[0] / ll, L[1] / ll, L[2] / ll);
      gl.uniform3fv(form.u.uLit, colors.lit);
      gl.uniform3fv(form.u.uMid, colors.mid);
      gl.uniform3fv(form.u.uShade, colors.shade);
      gl.uniform3fv(form.u.uBounce, colors.bounce);
      gl.uniform3fv(form.u.uLine, colors.line);
      var sw = sweep < 0 ? [0, 0] : [-1.9 + 3.8 * sweep, 0.32 * Math.sin(Math.PI * sweep)];
      gl.uniform2f(form.u.uSweep, sw[0], sw[1]);
      gl.bindBuffer(gl.ARRAY_BUFFER, formMesh.pos);
      gl.enableVertexAttribArray(form.a.aPos);
      gl.vertexAttribPointer(form.a.aPos, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, formMesh.idx);
      gl.drawElements(gl.TRIANGLES, formMesh.count, gl.UNSIGNED_SHORT, 0);
      gl.disableVertexAttribArray(form.a.aPos);

      // --- orbit, resting marks and dot
      gl.useProgram(flat.prog);
      gl.uniformMatrix4fv(flat.u.uProj, false, PROJ);
      gl.uniformMatrix4fv(flat.u.uView, false, VIEW);
      gl.uniform3fv(flat.u.uColor, colors.accent);
      gl.uniform1f(flat.u.uAlpha, 1);
      gl.enableVertexAttribArray(flat.a.aA);
      gl.enableVertexAttribArray(flat.a.aB);

      var orbit = mul(rotZ(s.roll), rotX(s.tilt));
      gl.uniformMatrix4fv(flat.u.uModel, false, orbit);
      gl.uniform2f(flat.u.uScale, ORBIT_R, linePx / 2 * worldPerPx);
      gl.bindBuffer(gl.ARRAY_BUFFER, orbitMesh.a);
      gl.vertexAttribPointer(flat.a.aA, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, orbitMesh.b);
      gl.vertexAttribPointer(flat.a.aB, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, orbitMesh.idx);
      gl.drawElements(gl.TRIANGLES, orbitMesh.count, gl.UNSIGNED_SHORT, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, markerMesh.pos);
      gl.vertexAttribPointer(flat.a.aA, 3, gl.FLOAT, false, 0, 0);
      gl.vertexAttribPointer(flat.a.aB, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, markerMesh.idx);
      function mark(angle, radiusPx, alpha) {
        var m = angle * Math.PI / 180;
        var at = apply(orbit, [Math.cos(m) * ORBIT_R, 0, Math.sin(m) * ORBIT_R]);
        gl.uniformMatrix4fv(flat.u.uModel, false, translate(at[0], at[1], at[2]));
        gl.uniform2f(flat.u.uScale, radiusPx * worldPerPx, 0);
        gl.uniform1f(flat.u.uAlpha, alpha);
        gl.drawElements(gl.TRIANGLES, markerMesh.count, gl.UNSIGNED_SHORT, 0);
      }
      if (notchAlpha > 0) {
        gl.depthMask(false);
        notches.forEach(function (a) { mark(a, notchPx, 0.85 * notchAlpha); });
        gl.depthMask(true);
      }
      mark(dot.angle, markerPx, 1);

      gl.disableVertexAttribArray(flat.a.aA);
      gl.disableVertexAttribArray(flat.a.aB);
    }

    // ---------------------------------------------------------- api
    var stone = {
      FRONT: FRONT,
      setPose: function (p) { pose = p; requestFrame(); },
      follow: function (angle) { takeOver('follow', angle); requestFrame(); },
      orbit: function (periodS) {
        if (periodS) dot.period = periodS;
        if (dot.mode !== 'orbit') { dot.mode = 'orbit'; dot.speed = 0; }   // ramps up from rest
        requestFrame();
      },
      rest: function (angle) {
        if (dot.mode !== 'rest' || dot.target !== angle) { dot.mode = ''; takeOver('rest', angle); }
        requestFrame();
      },
      place: function (angle) { dot.angle = dot.from = dot.target = angle; dot.blend = 1; requestFrame(); },
      boost: function (on) { dot.boost = !!on; requestFrame(); },
      dotMode: function () { return dot.mode; },
      dotAngle: function () { return dot.angle; },
      drift: function (on) { driftOn = !!on; requestFrame(); },
      setNotches: function (angles) { notches = angles.slice(); requestFrame(); },
      showNotches: function (on) { if (notchShow !== !!on) { notchShow = !!on; requestFrame(); } },
      sweep: function () { if (!still()) { sweepStart = performance.now() / 1000; requestFrame(); } },
      onUpdate: function (fn) { update = fn; requestFrame(); },
      reducedMotion: still,
      requestFrame: requestFrame,
      destroy: function () {
        if (dead) return;
        dead = true;
        if (pending) cancelAnimationFrame(rafId);
        document.removeEventListener('visibilitychange', onVisibility);
        canvas.removeEventListener('webglcontextlost', onLost);
        if (ro) ro.disconnect(); else window.removeEventListener('resize', resize);
        mo.disconnect();
        listenMQ(darkMQ, onTheme, false);
        listenMQ(reduceMQ, onMotion, false);
        release();
        delete canvas.__ycStone;
      }
    };
    canvas.__ycStone = stone;
    resize();
    return stone;
  }

  // ---------------------------------------------------------------- ambient
  // The content-page launcher: the neutral pose, a slow orbiting dot that
  // quickens a little on hover / focus, and a dot that glides to the front
  // and stops while the chat is open (and carries on from there when it
  // closes). Reduced motion: no orbit or drift - the dot sits at HOME while
  // the chat is closed and at the front while it is open, and steps between
  // the two.
  var AMBIENT_PERIOD_S = 10;
  var HOME = 125;   // front-left: where the dot starts, and rests under reduced motion

  function ambient(canvas, opts) {
    opts = opts || {};
    var stone = create(canvas, { onLost: opts.onLost });
    if (!stone) return null;
    stone.setPose(NEUTRAL);
    var open = !!(window.YCChat && window.YCChat.isOpen());
    function apply() {
      if (open) { stone.rest(FRONT); stone.drift(false); }
      else if (stone.reducedMotion()) { stone.rest(HOME); stone.drift(false); }
      else { stone.orbit(AMBIENT_PERIOD_S); stone.drift(true); }
    }
    stone.place(HOME);
    apply();
    window.addEventListener('ycchat-change', function (e) {
      open = !!(e.detail && e.detail.open);
      apply();
    });
    // The OS setting can change while the page is open.
    var mq = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)');
    if (mq && mq.addEventListener) mq.addEventListener('change', apply);
    else if (mq && mq.addListener) mq.addListener(apply);
    var host = opts.hoverTarget;
    if (host) {
      var hovered = false, focused = false;
      var set = function () { stone.boost(hovered || focused); };
      host.addEventListener('pointerenter', function (e) { if (e.pointerType === 'mouse') { hovered = true; set(); } });
      host.addEventListener('pointerleave', function (e) { if (e.pointerType === 'mouse') { hovered = false; set(); } });
      host.addEventListener('focus', function () { focused = true; set(); });
      host.addEventListener('blur', function () { focused = false; set(); });
    }
    return stone;
  }

  window.YCStone = {
    create: create,
    ambient: ambient,
    posePath: posePath,
    smoothDamp: smoothDamp,
    angleDiff: angleDiff,
    NEUTRAL: NEUTRAL,
    FRONT: FRONT
  };
})();
