(function () {
  'use strict';

  // The homepage's secondary accent: a small matte form inside one red orbit,
  // with a marker on the orbit. It does not navigate. p3-menu.js announces
  // the highlighted row with a `p3-select` event, and each row has a target
  // pose below; the object eases toward it, so a selection change reads as
  // the page answering, not as an animation to watch.

  var figure = document.querySelector('.p3-object');
  var canvas = figure && figure.querySelector('canvas');
  if (!canvas) return;
  var indexEl = figure.querySelector('.p3-object-index');
  var nameEl  = figure.querySelector('.p3-object-name');

  var gl = canvas.getContext('webgl', { antialias: true, alpha: true, premultipliedAlpha: true });
  if (!gl) return;   // the figure stays hidden (CSS) without .is-ready
  var lines = !!gl.getExtension('OES_standard_derivatives');

  var reduceMotion = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)');

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
  // The marker walks the orbit's front arc, left to right, as the selection
  // moves down the menu (deg along the orbit; 90 is nearest the viewer).
  var MARKER_FROM = 150, MARKER_TO = 30;

  var VIEW_HALF  = 2.0;    // world half-extent the canvas shows
  var ORBIT_R    = 1.88;   // orbit radius; the form's radius is ~1
  var LINE_PX    = 1.5;    // orbit stroke, CSS px
  var MARKER_PX  = 4.5;    // marker radius, CSS px
  var EASE_S     = 0.13;   // time constant of the pose ease: ~0.6 s to settle
  var SWEEP_S    = 0.75;   // contour sweep after a selection change

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
  var FORM_FS = [
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

  // Flat colour for the orbit and its marker: graphic marks in a rendered
  // scene, depth-tested so the orbit passes behind the form.
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
    'void main() { gl_FragColor = vec4(uColor, 1.0); }'
  ].join('\n');

  function compile(type, src) {
    var sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.warn('[p3-object] shader error:', gl.getShaderInfoLog(sh));
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
      console.warn('[p3-object] link error:', gl.getProgramInfoLog(p));
      return null;
    }
    var out = { prog: p, a: {}, u: {} };
    attribs.forEach(function (n) { out.a[n] = gl.getAttribLocation(p, n); });
    uniforms.forEach(function (n) { out.u[n] = gl.getUniformLocation(p, n); });
    return out;
  }

  var form = program(FORM_VS, FORM_FS, ['aPos'],
    ['uProj', 'uView', 'uModel', 'uWarp', 'uLight', 'uLit', 'uMid', 'uShade', 'uBounce', 'uLine', 'uSweep']);
  var flat = program(FLAT_VS, FLAT_FS, ['aA', 'aB'],
    ['uProj', 'uView', 'uModel', 'uScale', 'uColor']);
  if (!form || !flat) return;

  // ---------------------------------------------------------------- meshes
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

  // Torus around the Y axis as (point on unit circle, unit offset across the
  // tube), scaled in the vertex shader.
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
  var colors;
  function applyTheme() {
    var t = document.documentElement.getAttribute('data-theme');
    if (t !== 'light' && t !== 'dark') t = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    var p = PALETTES[t];
    colors = { lit: rgb(p.lit), mid: rgb(p.mid), shade: rgb(p.shade), bounce: rgb(p.bounce), line: rgb(p.line),
               accent: rgb(getComputedStyle(document.documentElement).getPropertyValue('--accent')) };
    requestFrame();
  }
  new MutationObserver(applyTheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // ---------------------------------------------------------------- state
  var KEYS = ['yaw', 'pitch', 'warp', 'tilt', 'roll', 'light', 'marker'];
  var current = null, target = null;
  var sweepStart = -1;

  function pad(n) { return (n < 10 ? '0' : '') + n; }

  window.addEventListener('p3-select', function (e) {
    var d = e.detail || {};
    var s = STATES[d.id] || STATES.projects;
    var t = d.total > 1 ? d.index / (d.total - 1) : 0;
    var next = { yaw: s.yaw, pitch: s.pitch, warp: s.warp, tilt: s.tilt, roll: s.roll, light: s.light,
                 marker: MARKER_FROM + (MARKER_TO - MARKER_FROM) * t };
    var changed = !target || KEYS.some(function (k) { return next[k] !== target[k]; });
    target = next;
    if (!current || reduceMotion && reduceMotion.matches) current = Object.assign({}, target);
    else if (changed && sweepAt(now()) < 0) sweepStart = now();

    if (indexEl) indexEl.textContent = d.ordinal ? pad(d.ordinal) + ' / ' + pad(d.sections) : '↗';
    if (nameEl)  nameEl.textContent  = d.label || '';
    requestFrame();
  });

  function now() { return performance.now() / 1000; }
  // Progress of the running sweep in [0, 1), or -1 when none is running.
  function sweepAt(t) {
    if (sweepStart < 0) return -1;
    var p = (t - sweepStart) / SWEEP_S;
    return p < 1 ? p : -1;
  }

  // ---------------------------------------------------------------- render
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var cssSize = 0;
  function resize() {
    cssSize = canvas.clientWidth || 1;
    var px = Math.max(1, Math.round(cssSize * dpr));
    if (canvas.width !== px || canvas.height !== px) { canvas.width = px; canvas.height = px; }
    gl.viewport(0, 0, px, px);
    requestFrame();
  }
  if (window.ResizeObserver) new ResizeObserver(resize).observe(canvas);
  else window.addEventListener('resize', resize);

  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0, 0, 0, 0);

  var pending = false, lastT = 0;
  function requestFrame() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(frame);
  }

  function frame(ms) {
    pending = false;
    if (!current || !colors) return;
    var t = ms / 1000, dt = Math.min(0.1, lastT ? t - lastT : 0);
    lastT = t;
    var still = reduceMotion && reduceMotion.matches;

    var moving = false;
    var k = still ? 1 : 1 - Math.exp(-dt / EASE_S);
    KEYS.forEach(function (key) {
      var diff = target[key] - current[key];
      if (Math.abs(diff) > 0.01) moving = true;
      current[key] += diff * k;
    });

    // Idle: a yaw drift of under two degrees over sixteen seconds.
    var idle = still ? 0 : 1.6 * Math.sin(now() * Math.PI * 2 / 16);
    var sweep = still ? -1 : sweepAt(now());

    draw(current, idle, sweep);

    // Reduced motion draws only on change; otherwise the idle drift keeps
    // the loop going (one small canvas, paused by the browser when hidden).
    if (!still || moving) requestFrame();
  }

  function draw(s, idle, sweep) {
    var worldPerPx = VIEW_HALF * 2 / cssSize;

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // --- form
    gl.useProgram(form.prog);
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

    // --- orbit and marker
    gl.useProgram(flat.prog);
    gl.uniformMatrix4fv(flat.u.uProj, false, PROJ);
    gl.uniformMatrix4fv(flat.u.uView, false, VIEW);
    gl.uniform3fv(flat.u.uColor, colors.accent);
    gl.enableVertexAttribArray(flat.a.aA);
    gl.enableVertexAttribArray(flat.a.aB);

    var orbit = mul(rotZ(s.roll), rotX(s.tilt));
    gl.uniformMatrix4fv(flat.u.uModel, false, orbit);
    gl.uniform2f(flat.u.uScale, ORBIT_R, LINE_PX / 2 * worldPerPx);
    gl.bindBuffer(gl.ARRAY_BUFFER, orbitMesh.a);
    gl.vertexAttribPointer(flat.a.aA, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, orbitMesh.b);
    gl.vertexAttribPointer(flat.a.aB, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, orbitMesh.idx);
    gl.drawElements(gl.TRIANGLES, orbitMesh.count, gl.UNSIGNED_SHORT, 0);

    var m = s.marker * Math.PI / 180;
    var at = apply(orbit, [Math.cos(m) * ORBIT_R, 0, Math.sin(m) * ORBIT_R]);
    gl.uniformMatrix4fv(flat.u.uModel, false, translate(at[0], at[1], at[2]));
    gl.uniform2f(flat.u.uScale, MARKER_PX * worldPerPx, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, markerMesh.pos);
    gl.vertexAttribPointer(flat.a.aA, 3, gl.FLOAT, false, 0, 0);
    gl.vertexAttribPointer(flat.a.aB, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, markerMesh.idx);
    gl.drawElements(gl.TRIANGLES, markerMesh.count, gl.UNSIGNED_SHORT, 0);

    gl.disableVertexAttribArray(flat.a.aA);
    gl.disableVertexAttribArray(flat.a.aB);
  }

  applyTheme();
  resize();
  figure.classList.add('is-ready');
})();
