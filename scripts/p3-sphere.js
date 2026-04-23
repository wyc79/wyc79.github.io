(function () {
  'use strict';

  var canvas = document.getElementById('p3-sphere');
  if (!canvas) return;

  var gl = canvas.getContext('webgl', { antialias: true, premultipliedAlpha: false }) ||
           canvas.getContext('experimental-webgl');
  if (!gl) return;

  var VERT_SRC = [
    'attribute vec3 aPos;',
    'attribute vec3 aNormal;',
    'uniform mat4 uProj;',
    'uniform mat4 uView;',
    'uniform mat4 uModel;',
    'varying vec3 vWorld;',
    'varying vec3 vNormal;',
    'void main() {',
    '  vec4 w = uModel * vec4(aPos, 1.0);',
    '  vWorld = w.xyz;',
    '  vNormal = mat3(uModel) * aNormal;',
    '  gl_Position = uProj * uView * w;',
    '}'
  ].join('\n');

  var FRAG_SRC = [
    'precision mediump float;',
    'varying vec3 vWorld;',
    'varying vec3 vNormal;',
    'uniform vec3 uLightPos;',
    'uniform vec3 uViewPos;',
    'uniform vec3 uBaseColor;',
    // 7-band toon palette is supplied as a uniform array so the JS layer can
    // swap the entire palette when the theme changes without recompiling.
    'uniform vec3 uBands[7];',
    'void main() {',
    '  vec3 N = normalize(vNormal);',
    '  vec3 L = normalize(uLightPos - vWorld);',
    '',
    '  float ndl = max(dot(N, L), 0.0);',
    '',
    // Reserve only a thin low-end strip [0, shadowCutoff) for c0; evenly
    // divide [shadowCutoff, 1.0] into 5 equal segments for c1..c5 so they
    // stay evenly separated. c6 (specular tip) still gates on raw ndl, kept
    // unchanged so the white highlight doesn't grow.
    '  float shadowCutoff = 0.12;',
    '  float lit = clamp((ndl - shadowCutoff) / (1.0 - shadowCutoff), 0.0, 1.0);',
    '',
    // Build the lit ramp first (c1..c5), then snap to c0 wherever ndl is
    // below the shadow cutoff. Order matters: the c0 snap must happen after
    // the ramp but before the specular tip.
    '  vec3 diffuseColor = uBands[1];',
    '  diffuseColor = mix(diffuseColor, uBands[2], step(1.0/5.0, lit));',
    '  diffuseColor = mix(diffuseColor, uBands[3], step(2.0/5.0, lit));',
    '  diffuseColor = mix(diffuseColor, uBands[4], step(3.0/5.0, lit));',
    '  diffuseColor = mix(diffuseColor, uBands[5], step(4.0/5.0, lit));',
    '  diffuseColor = mix(uBands[0], diffuseColor, step(shadowCutoff, ndl));',
    '  diffuseColor = mix(diffuseColor, uBands[6], step(19.0/20.0, ndl));',
    '',
    '  gl_FragColor = vec4(diffuseColor, 1.0);',
    '}'
  ].join('\n');

  function compile(type, src) {
    var sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.warn('[p3-sphere] shader error:', gl.getShaderInfoLog(sh));
      gl.deleteShader(sh);
      return null;
    }
    return sh;
  }

  var vs = compile(gl.VERTEX_SHADER, VERT_SRC);
  var fs = compile(gl.FRAGMENT_SHADER, FRAG_SRC);
  if (!vs || !fs) return;

  var prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.warn('[p3-sphere] link error:', gl.getProgramInfoLog(prog));
    return;
  }
  gl.useProgram(prog);

  // --- Build UV sphere ---
  function buildSphere(segments, rings) {
    var positions = [];
    var normals = [];
    var indices = [];
    for (var y = 0; y <= rings; y++) {
      var v = y / rings;
      var phi = v * Math.PI; // 0..PI
      for (var x = 0; x <= segments; x++) {
        var u = x / segments;
        var theta = u * Math.PI * 2; // 0..2PI
        var sx = Math.sin(phi) * Math.cos(theta);
        var sy = Math.cos(phi);
        var sz = Math.sin(phi) * Math.sin(theta);
        positions.push(sx, sy, sz);
        normals.push(sx, sy, sz);
      }
    }
    var stride = segments + 1;
    for (var j = 0; j < rings; j++) {
      for (var i = 0; i < segments; i++) {
        var a = j * stride + i;
        var b = a + stride;
        indices.push(a, b, a + 1, b, b + 1, a + 1);
      }
    }
    return {
      positions: new Float32Array(positions),
      normals: new Float32Array(normals),
      indices: new Uint16Array(indices)
    };
  }

  var sphere = buildSphere(96, 64);

  var aPos = gl.getAttribLocation(prog, 'aPos');
  var aNormal = gl.getAttribLocation(prog, 'aNormal');

  var posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, sphere.positions, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);

  var normBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, normBuf);
  gl.bufferData(gl.ARRAY_BUFFER, sphere.normals, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(aNormal);
  gl.vertexAttribPointer(aNormal, 3, gl.FLOAT, false, 0, 0);

  var idxBuf = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, sphere.indices, gl.STATIC_DRAW);

  var uProj = gl.getUniformLocation(prog, 'uProj');
  var uView = gl.getUniformLocation(prog, 'uView');
  var uModel = gl.getUniformLocation(prog, 'uModel');
  var uLightPos = gl.getUniformLocation(prog, 'uLightPos');
  var uViewPos = gl.getUniformLocation(prog, 'uViewPos');
  var uBaseColor = gl.getUniformLocation(prog, 'uBaseColor');
  var uBands = gl.getUniformLocation(prog, 'uBands[0]');

  // --- Theme-aware palettes ---
  // c0 is the darkest toon band (deepest shadow); c6 is the brightest
  // highlight. Both palettes step smoothly through 7 stops so the banding
  // reads cleanly. The JS reads the live `data-theme` attribute set by
  // scripts/theme.js — same source the rest of the site uses.
  var DARK_BANDS = new Float32Array([
    0.060, 0.075, 0.150,   // c0 deep navy shadow
    0.095, 0.145, 0.330,   // c1 dark blue
    0.155, 0.275, 0.560,   // c2 mid blue
    0.240, 0.435, 0.800,   // c3 brighter blue
    0.350, 0.560, 0.895,   // c4 lit blue
    0.480, 0.700, 0.970,   // c5 saturated blue highlight
    0.920, 0.955, 0.995    // c6 brightest band — almost white
  ]);
  // Light theme: muted dark red base, building toward warm pink highlights.
  // Tuned to stay tasteful on a white page — no neon, no pure red.
  var LIGHT_BANDS = new Float32Array([
    0.220, 0.075, 0.090,   // c0 dark muted red
    0.360, 0.130, 0.155,   // c1 deep red
    0.540, 0.230, 0.270,   // c2 mid warm red
    0.720, 0.380, 0.420,   // c3 dusty rose
    0.840, 0.530, 0.565,   // c4 lit pinkish red
    0.925, 0.685, 0.710,   // c5 soft pink highlight
    0.995, 0.955, 0.945    // c6 brightest — warm near-white
  ]);
  var DARK_CLEAR  = [0.0, 0.0, 0.0];
  var LIGHT_CLEAR = [1.0, 1.0, 1.0];

  var bands = DARK_BANDS;
  var clearRGB = DARK_CLEAR;

  function resolvedTheme() {
    var t = document.documentElement.getAttribute('data-theme');
    if (t === 'light' || t === 'dark') return t;
    return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function applyTheme() {
    if (resolvedTheme() === 'light') {
      bands = LIGHT_BANDS;
      clearRGB = LIGHT_CLEAR;
    } else {
      bands = DARK_BANDS;
      clearRGB = DARK_CLEAR;
    }
  }
  applyTheme();
  // Live updates when the user toggles the theme on this page or comes back
  // from another page that changed it.
  new MutationObserver(applyTheme).observe(document.documentElement, {
    attributes: true, attributeFilter: ['data-theme']
  });

  // --- Matrix helpers (column-major 4x4) ---
  function mat4Identity() {
    return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
  }
  function mat4Perspective(fovy, aspect, near, far) {
    var f = 1 / Math.tan(fovy / 2);
    var nf = 1 / (near - far);
    return new Float32Array([
      f/aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far+near)*nf, -1,
      0, 0, 2*far*near*nf, 0
    ]);
  }
  function mat4LookAt(eye, target, up) {
    var zx = eye[0]-target[0], zy = eye[1]-target[1], zz = eye[2]-target[2];
    var zl = Math.hypot(zx, zy, zz) || 1;
    zx/=zl; zy/=zl; zz/=zl;
    var xx = up[1]*zz - up[2]*zy;
    var xy = up[2]*zx - up[0]*zz;
    var xz = up[0]*zy - up[1]*zx;
    var xl = Math.hypot(xx, xy, xz) || 1;
    xx/=xl; xy/=xl; xz/=xl;
    var yx = zy*xz - zz*xy;
    var yy = zz*xx - zx*xz;
    var yz = zx*xy - zy*xx;
    return new Float32Array([
      xx, yx, zx, 0,
      xy, yy, zy, 0,
      xz, yz, zz, 0,
      -(xx*eye[0]+xy*eye[1]+xz*eye[2]),
      -(yx*eye[0]+yy*eye[1]+yz*eye[2]),
      -(zx*eye[0]+zy*eye[1]+zz*eye[2]),
      1
    ]);
  }
  function mat4TranslateScale(tx, ty, tz, s) {
    return new Float32Array([
      s, 0, 0, 0,
      0, s, 0, 0,
      0, 0, s, 0,
      tx, ty, tz, 1
    ]);
  }

  // --- State ---
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var width = 0, height = 0;
  // Default virtual mouse at the right-edge midpoint (dome center) so the
  // resting state gives centered lighting on the sphere.
  var mouseNX = 1, mouseNY = 0;
  var smoothX = 1, smoothY = 0;
  var eye = [0, 0, 4.2];

  function resize() {
    var rect = canvas.getBoundingClientRect();
    var w = rect.width  || window.innerWidth;
    var h = rect.height || window.innerHeight;
    width = Math.max(1, Math.floor(w * dpr));
    height = Math.max(1, Math.floor(h * dpr));
    canvas.width = width;
    canvas.height = height;
    gl.viewport(0, 0, width, height);
  }

  window.addEventListener('resize', resize);
  window.addEventListener('mousemove', function (e) {
    mouseNX = (e.clientX / window.innerWidth) * 2 - 1;
    mouseNY = -((e.clientY / window.innerHeight) * 2 - 1);
  });
  window.addEventListener('touchmove', function (e) {
    if (!e.touches || !e.touches[0]) return;
    mouseNX = (e.touches[0].clientX / window.innerWidth) * 2 - 1;
    mouseNY = -((e.touches[0].clientY / window.innerHeight) * 2 - 1);
  }, { passive: true });

  resize();

  gl.enable(gl.DEPTH_TEST);

  function render() {
    // smooth the mouse toward target for gentle motion
    smoothX += (mouseNX - smoothX) * 0.06;
    smoothY += (mouseNY - smoothY) * 0.06;

    var aspect = width / Math.max(1, height);
    var fovY = Math.PI / 4;
    var proj = mat4Perspective(fovY, aspect, 0.1, 100);
    var view = mat4LookAt(eye, [0, 0, 0], [0, 1, 0]);

    // World-space half-dimensions of the viewport at z=0. A CSS-pixel distance
    // equal to one viewport dimension corresponds to 2*halfH (vertical) or
    // 2*halfW (horizontal) in world units.
    var halfH = Math.tan(fovY / 2) * eye[2];
    var halfW = halfH * aspect;
    var worldHeight = 2 * halfH;
    var worldWidth  = 2 * halfW;

    // Responsive sizing:
    //   diameter <= viewportHeight * 0.8  =>  radius <= 0.4 * worldHeight
    //   radius   <= viewportWidth  * 0.8  =>  radius <= 0.8 * worldWidth
    var sphereRadius = Math.min(0.4 * worldHeight, 0.8 * worldWidth);

    // Portrait viewports push the sphere center further right (by half a
    // radius) so the sphere stops swallowing the left-side menu. Landscape
    // keeps the original layout — center exactly on the right-edge midpoint,
    // so only the left half is visible.
    var portrait = window.innerHeight > window.innerWidth;
    var centerOffsetWorld = portrait ? sphereRadius * 0.5 : 0;
    var centerX = halfW + centerOffsetWorld;
    var model = mat4TranslateScale(centerX, 0, 0, sphereRadius);

    // --- Clamped dome lighting ---
    // The light rides a hemisphere centered at the screen's right-edge
    // midpoint (domeCenter = (viewportWidth, viewportHeight/2)) with radius
    // slightly smaller than the viewport width. Mouse offsets are measured
    // from THAT point (not from screen center). Offsets are clamped to the
    // dome rim — no wraparound — and z is solved from the sphere equation.
    //
    // The dome's forward axis is aligned with the direction from the sphere
    // center to the camera (NOT world +Z). Because the sphere is offset to
    // the right edge of the viewport, world +Z does not point at the camera;
    // aligning the dome forward with the actual camera axis is what makes
    // mouse-at-dome-center produce a centered highlight on the visible face
    // instead of a glancing rim light.
    var vw = window.innerWidth;
    var vh = window.innerHeight;

    // Square pixel → world-unit ratio (used for both the dome-center shift
    // below and the mouse-offset conversion further down).
    var pxToWorld = worldWidth / vw;

    // Smoothed mouse back in pixel coords. smoothY is already flipped (top = +1).
    var mouseX_px = (smoothX + 1) * 0.5 * vw;
    var mouseY_px = (1 - smoothY) * 0.5 * vh;

    // Dome center follows the (possibly shifted) sphere center so the
    // centered-highlight behavior remains correct on portrait viewports.
    var domeCx_px = vw + centerOffsetWorld / pxToWorld;
    var domeCy_px = vh * 0.5;     // vertical midpoint (Y)
    var domeR_px  = 0.85 * vw;    // slightly smaller than viewport width

    // 1. Mouse offset from dome center, in screen pixels.
    var dx_px = mouseX_px - domeCx_px;
    var dy_px = mouseY_px - domeCy_px;

    // 2. Radial clamp to the dome rim. No wraparound — if the mouse is past
    //    the rim, pin the offset to the rim and let the light sit there.
    var d2 = dx_px * dx_px + dy_px * dy_px;
    var r2 = domeR_px * domeR_px;
    if (d2 > r2) {
      var s = domeR_px / Math.sqrt(d2);
      dx_px *= s;
      dy_px *= s;
    }

    // 3. Reconstruct dome Z (front-facing hemisphere).
    var dz_px = Math.sqrt(Math.max(0, r2 - dx_px * dx_px - dy_px * dy_px));

    // Convert pixel offsets to world units (pxToWorld computed above).
    var lX = dx_px * pxToWorld;   // dome-local +X = screen right
    var lY = -dy_px * pxToWorld;  // dome-local +Y = screen up  (flip Y)
    var lZ = dz_px * pxToWorld;   // dome-local +Z = forward along camera axis

    // 4. Dome basis in world. Forward F is the unit vector from the sphere
    //    center to the camera. Right R and up U complete an orthonormal frame
    //    with world up (0,1,0).
    var fx = -centerX;
    var fz = eye[2];
    var fLen = Math.sqrt(fx * fx + fz * fz);
    fx /= fLen;
    fz /= fLen;
    //   F = ( fx,  0, fz)
    //   R = cross(worldUp, F) = (fz, 0, -fx)
    //   U = (  0,  1,  0)

    // 5. Build world-space light position = sphereCenter + lX*R + lY*U + lZ*F.
    //    Sphere center == dome center == (centerX, 0, 0) in world.
    //
    //    Signs verified:
    //      mouse at right-edge midpoint → (lX,lY,lZ) = (0, 0, +R) → light on
    //        camera axis → centered illumination on visible face. ✓
    //      mouse at left-edge midpoint  → clamp puts (lX,lY) on the rim at
    //        (-R, 0), lZ=0 → light sits in the dome-local XY plane, off to
    //        the screen-left side → grazing light from the left. ✓
    var lx = centerX + lX * fz + lZ * fx;
    var ly =         lY;
    var lz =       - lX * fx + lZ * fz;

    gl.clearColor(clearRGB[0], clearRGB[1], clearRGB[2], 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(prog);
    gl.uniformMatrix4fv(uProj, false, proj);
    gl.uniformMatrix4fv(uView, false, view);
    gl.uniformMatrix4fv(uModel, false, model);
    gl.uniform3f(uLightPos, lx, ly, lz);
    gl.uniform3f(uViewPos, eye[0], eye[1], eye[2]);
    // Legacy single-color base (still bound for back-compat with shader history).
    gl.uniform3f(uBaseColor, 0.14, 0.30, 0.72);
    // Active toon palette — swapped by the theme observer above.
    gl.uniform3fv(uBands, bands);

    gl.drawElements(gl.TRIANGLES, sphere.indices.length, gl.UNSIGNED_SHORT, 0);

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
})();
