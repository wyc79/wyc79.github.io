"""The stone as the site-wide Ask AI launcher.

Static checks cover how pages load the shared renderer. The behavioural
checks run the real scripts in `node` (skipped when node is missing): the
chat widget's open-state handling and text fallback, the renderer's motion
helpers, and the homepage controller's split between dragging the stone and
clicking it.
"""

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PAGES = [ROOT / "index.html", *sorted((ROOT / "pages").glob("*.html"))]
NODE = shutil.which("node")


def script_srcs(html: str) -> list[str]:
    return re.findall(r'<script\b[^>]*\bsrc="([^"]+)"[^>]*>', html)


def script_tag(html: str, name: str) -> str:
    return re.search(r'<script\b[^>]*src="[^"]*' + re.escape(name) + r'"[^>]*>', html).group(0)


def extract_function(src: str, name: str) -> str:
    """The full text of `function name(...) { ... }`, by brace matching."""
    start = src.index("function " + name + "(")
    depth, i = 0, src.index("{", start)
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1


def run_node(script: str) -> dict:
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30, encoding="utf-8")
    if out.returncode != 0:
        raise AssertionError("node failed:\n" + out.stderr)
    return json.loads(out.stdout)


class PagesLoadTheRenderer(unittest.TestCase):
    def test_every_page_loads_the_renderer_once_before_the_widget(self):
        for page in PAGES:
            with self.subTest(page=page.name):
                srcs = script_srcs(page.read_text(encoding="utf-8"))
                names = [s.rsplit("/", 1)[-1] for s in srcs]
                self.assertEqual(names.count("stone-renderer.js"), 1)
                self.assertEqual(names.count("chat-widget.js"), 1)
                self.assertLess(names.index("stone-renderer.js"), names.index("chat-widget.js"))

    def test_renderer_and_widget_are_deferred(self):
        # Deferred scripts run in document order, so the renderer is always
        # defined before the widget or the homepage controller needs it.
        for page in PAGES:
            html = page.read_text(encoding="utf-8")
            for name in ("stone-renderer.js", "chat-widget.js"):
                with self.subTest(page=page.name, script=name):
                    self.assertIn("defer", script_tag(html, name))

    def test_homepage_controllers_load_after_the_renderer(self):
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        names = [s.rsplit("/", 1)[-1] for s in script_srcs(html)]
        self.assertLess(names.index("stone-renderer.js"), names.index("p3-object.js"))
        self.assertLess(names.index("p3-object.js"), names.index("p3-menu.js"))
        for name in ("p3-object.js", "p3-menu.js"):
            self.assertIn("defer", script_tag(html, name))

    def test_content_pages_do_not_load_homepage_navigation(self):
        for page in PAGES[1:]:
            with self.subTest(page=page.name):
                names = [s.rsplit("/", 1)[-1] for s in script_srcs(page.read_text(encoding="utf-8"))]
                self.assertNotIn("p3-menu.js", names)
                self.assertNotIn("p3-object.js", names)

    def test_homepage_launcher_is_a_button_outside_any_aria_hidden_subtree(self):
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        figure = re.search(r'<figure class="p3-object"[^>]*>.*?</figure>', html, re.S).group(0)
        self.assertNotIn('aria-hidden="true">', figure.split("\n", 1)[0])
        button = re.search(r"<button\b[^>]*>", figure).group(0)
        self.assertIn("data-ycchat-launcher", button)
        self.assertIn('type="button"', button)
        self.assertRegex(figure, r'<canvas aria-hidden="true">')

    def test_webgl_lives_only_in_the_shared_renderer(self):
        for path in SCRIPTS.glob("*.js"):
            src = path.read_text(encoding="utf-8")
            with self.subTest(script=path.name):
                if path.name == "stone-renderer.js":
                    self.assertIn("createShader", src)
                else:
                    self.assertNotIn("createShader", src)
                    self.assertNotIn("getContext('webgl'", src)


WIDGET = (SCRIPTS / "chat-widget.js").read_text(encoding="utf-8")


@unittest.skipUnless(NODE, "node not on PATH")
class ChatOpenState(unittest.TestCase):
    """setOpen() is the one place the panel opens or closes."""

    def run_widget(self, steps: str) -> dict:
        script = r"""
var events = [], built = 0, warmed = 0, positioned = 0, focused = 0;
var window = { dispatchEvent: function (e) { events.push(e.detail.open); } };
function CustomEvent(type, init) { this.type = type; this.detail = init.detail; }
var document = { activeElement: null };
var attrs = {};
var els = { btn: { setAttribute: function (k, v) { attrs[k] = v; }, focus: function () { focused++; } } };
var state = { open: false, roles: {} };
function buildPanel() { built++; els.panel = { style: {}, contains: function () { return true; } }; }
function positionPanel() { positioned++; }
function prewarm() { warmed++; }
function restoreView() {}
""" + extract_function(WIDGET, "setOpen") + "\n" + extract_function(WIDGET, "toggle") + "\n" + steps + r"""
process.stdout.write(JSON.stringify({ events: events, expanded: attrs['aria-expanded'], built: built,
  warmed: warmed, positioned: positioned, focused: focused, open: state.open,
  display: els.panel && els.panel.style.display }));
"""
        return run_node(script)

    def test_open_close_keeps_aria_expanded_and_events_in_step(self):
        got = self.run_widget("setOpen(true); setOpen(false); toggle(); toggle();")
        self.assertEqual(got["events"], [True, False, True, False])
        self.assertEqual(got["expanded"], "false")
        self.assertEqual(got["display"], "none")
        self.assertEqual(got["built"], 1, "the panel is built once, on first open")

    def test_repeated_requests_for_the_same_state_do_nothing(self):
        got = self.run_widget("setOpen(true); setOpen(true); setOpen(true);")
        self.assertEqual(got["events"], [True])
        self.assertEqual(got["expanded"], "true")
        self.assertEqual(got["positioned"], 1)
        self.assertEqual(got["warmed"], 1)

    def test_closing_from_inside_the_panel_returns_focus_to_the_launcher(self):
        got = self.run_widget("setOpen(true); setOpen(false);")
        self.assertEqual(got["focused"], 1)

    def test_nothing_is_warmed_until_the_chat_is_opened(self):
        got = self.run_widget("")
        self.assertEqual((got["warmed"], got["built"], got["events"]), (0, 0, []))

    def test_widget_refuses_to_initialise_twice(self):
        head = WIDGET[WIDGET.index("(function () {"):][:400]
        self.assertIn("if (window.YCChat) return;", head)


@unittest.skipUnless(NODE, "node not on PATH")
class LauncherFallback(unittest.TestCase):
    """The content-page launcher is a text button until the stone works."""

    def build(self, stone_js: str) -> dict:
        script = r"""
function el(tag) {
  return { tag: tag, children: [], attrs: {}, className: '', textContent: '',
    classList: { set: {}, add: function (c) { this.set[c] = 1; }, remove: function (c) { delete this.set[c]; },
                 contains: function (c) { return !!this.set[c]; } },
    setAttribute: function (k, v) { this.attrs[k] = v; }, appendChild: function (c) { this.children.push(c); } };
}
var document = { createElement: el, body: el('body') };
function h(tag, cls, text) { var e = el(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
function t(k) { return { askBtn: 'TEXT BUTTON', askLabel: 'STONE LABEL' }[k]; }
var els = {};
var destroyed = 0, lostHook = null;
var window = {};
""" + stone_js + "\n" + extract_function(WIDGET, "buildLauncher") + r"""
buildLauncher();
var before = { stone: els.btn.classList.contains('is-stone'), label: els.btnLabel.textContent };
if (lostHook) lostHook();
process.stdout.write(JSON.stringify({ before: before, launchers: document.body.children.length,
  canvasHidden: els.btn.children.filter(function (c) { return c.tag === 'canvas'; })[0].attrs['aria-hidden'],
  labelFirst: els.btn.children[0] === els.btnLabel,
  after: { stone: els.btn.classList.contains('is-stone'), label: els.btnLabel.textContent }, destroyed: destroyed }));
"""
        return run_node(script)

    def test_without_the_renderer_the_text_button_stays(self):
        got = self.build("")
        self.assertEqual(got["before"], {"stone": False, "label": "TEXT BUTTON"})
        self.assertEqual(got["launchers"], 1)

    def test_when_the_stone_cannot_start_the_text_button_stays(self):
        got = self.build("window.YCStone = { ambient: function () { return null; } };")
        self.assertEqual(got["before"], {"stone": False, "label": "TEXT BUTTON"})

    def test_a_working_stone_replaces_the_text_and_a_lost_one_restores_it(self):
        got = self.build(
            "window.YCStone = { ambient: function (c, o) { lostHook = o.onLost;"
            " return { destroy: function () { destroyed++; } }; } };"
        )
        self.assertEqual(got["before"], {"stone": True, "label": "STONE LABEL"})
        self.assertEqual(got["canvasHidden"], "true")
        self.assertTrue(got["labelFirst"], "the label sits above the stone")
        self.assertEqual(got["after"], {"stone": False, "label": "TEXT BUTTON"})
        self.assertEqual(got["destroyed"], 1)


RENDERER = (SCRIPTS / "stone-renderer.js").read_text(encoding="utf-8")


@unittest.skipUnless(NODE, "node not on PATH")
class StoneMotionHelpers(unittest.TestCase):
    def run_helpers(self, body: str) -> dict:
        return run_node("var window = {};\n" + RENDERER + "\nvar S = window.YCStone;\n" + body)

    def test_smooth_damp_reaches_95_percent_in_about_2_4_smooth_times_without_overshoot(self):
        got = self.run_helpers(r"""
var x = 0, v = 0, t = 0, dt = 1 / 60, t95 = null, max = 0;
while (t < 2) { var r = S.smoothDamp(x, 1, v, 0.15, dt); x = r[0]; v = r[1]; t += dt;
  max = Math.max(max, x); if (t95 === null && x >= 0.95) t95 = t; }
process.stdout.write(JSON.stringify({ t95: t95, max: max }));
""")
        self.assertGreaterEqual(got["t95"], 0.32)
        self.assertLessEqual(got["t95"], 0.40)
        self.assertLessEqual(got["max"], 1.0)

    def test_retargeting_keeps_velocity_and_reversal_is_one_continuous_turn(self):
        got = self.run_helpers(r"""
var x = 0, v = 0, dt = 1 / 60, xs = [], vBefore, vAfter;
for (var i = 0; i < 12; i++) { var r = S.smoothDamp(x, 6, v, 0.15, dt); x = r[0]; v = r[1]; xs.push(x); }
vBefore = v;
var r2 = S.smoothDamp(x, 1, v, 0.15, dt); vAfter = r2[1];
for (var j = 0; j < 120; j++) { var r3 = S.smoothDamp(x, 1, v, 0.15, dt); x = r3[0]; v = r3[1]; xs.push(x); }
var turns = 0;
for (var k = 2; k < xs.length; k++) if ((xs[k] - xs[k-1]) * (xs[k-1] - xs[k-2]) < -1e-12) turns++;
process.stdout.write(JSON.stringify({ vBefore: vBefore, vAfter: vAfter, turns: turns, end: x, min: Math.min.apply(null, xs.slice(12)) }));
""")
        self.assertGreater(got["vBefore"], 0)
        self.assertGreater(got["vAfter"], 0, "retargeting must not zero the velocity")
        self.assertEqual(got["turns"], 1)
        self.assertAlmostEqual(got["end"], 1, places=3)
        self.assertGreaterEqual(got["min"], 1 - 1e-9, "no overshoot past the new target")

    def test_angle_diff_and_pose_path(self):
        got = self.run_helpers(r"""
var a = S.posePath([{yaw:0,pitch:0,warp:0,tilt:0,roll:0,light:0},{yaw:10,pitch:2,warp:1,tilt:4,roll:-2,light:6}], 0.5);
process.stdout.write(JSON.stringify({ d1: S.angleDiff(350, 10), d2: S.angleDiff(10, 350), d3: S.angleDiff(0, 180), mid: a }));
""")
        self.assertEqual((got["d1"], got["d2"], got["d3"]), (20, -20, 180))
        self.assertEqual(got["mid"], {"yaw": 5, "pitch": 1, "warp": 0.5, "tilt": 2, "roll": -1, "light": 3})


CONTROLLER = (SCRIPTS / "p3-object.js").read_text(encoding="utf-8")

# A minimal DOM for p3-object.js: one figure, its launcher and canvas, window
# events, and a stand-in stone. Pointer gestures are fed straight to the
# canvas handlers.
DOM = r"""
var now = 0;
var performance = { now: function () { return now; } };
var winHandlers = {}, requests = [], toggles = 0;
function CustomEvent(type, init) { this.type = type; this.detail = init.detail; }
function Target() { this.h = {}; this.cls = {}; var self = this;
  this.classList = { add: function (c) { self.cls[c] = 1; }, remove: function (c) { delete self.cls[c]; }, contains: function (c) { return !!self.cls[c]; } }; }
Target.prototype.addEventListener = function (t, f) { (this.h[t] = this.h[t] || []).push(f); };
Target.prototype.fire = function (t, e) { (this.h[t] || []).forEach(function (f) { f(e); }); };
var canvas = new Target(); canvas.setPointerCapture = function () {}; canvas.releasePointerCapture = function () {};
var launcher = new Target(); launcher.contains = function (n) { return n === canvas || n === launcher; };
var figure = new Target();
figure.querySelector = function (s) { return { '.p3-stone': launcher, canvas: canvas }[s] || null; };
var document = { querySelector: function (s) { return s === '.p3-object' ? figure : null; } };
var window = {
  addEventListener: function (t, f) { (winHandlers[t] = winHandlers[t] || []).push(f); },
  dispatchEvent: function (e) { if (e.type === 'p3-request') requests.push(e.detail.phase + (e.detail.index !== undefined ? ':' + e.detail.index : ''));
                                (winHandlers[e.type] || []).forEach(function (f) { f(e); }); },
  YCChat: { toggle: function () { toggles++; } },
  YCStone: {
    create: function () { return { FRONT: 90, setPose: function () {}, follow: function () {}, orbit: function () {}, rest: function () {},
      drift: function () {}, setNotches: function () {}, showNotches: function () {}, sweep: function () {}, onUpdate: function () {},
      requestFrame: function () {}, reducedMotion: function () { return false; }, destroy: function () {} }; },
    smoothDamp: function (c, t, v) { return [t, 0]; }, posePath: function (p) { return p[0]; }
  }
};
"""

GESTURES = r"""
window.dispatchEvent(new CustomEvent('p3-select', { detail: { index: 0, goal: 0, total: 7, settled: true, source: 'init',
  ids: ['projects','skills','education','publications','agents','toolbox','github'] } }));
function ev(x, y, type) { return { pointerId: 1, isPrimary: true, button: 0, pointerType: type || 'mouse', clientX: x, clientY: y, preventDefault: function () {} }; }
function gesture(points, type) {
  canvas.fire('pointerdown', ev(points[0][0], points[0][1], type));
  points.slice(1).forEach(function (p) { now += 16; canvas.fire('pointermove', ev(p[0], p[1], type)); });
  now += 16; var last = points[points.length - 1];
  canvas.fire('pointerup', ev(last[0], last[1], type));
}
// The browser's click after the gesture: returns true if it got through to the launcher.
function click() { var stopped = false;
  (winHandlers.click || []).forEach(function (f) { f({ target: canvas, preventDefault: function () {}, stopPropagation: function () { stopped = true; } }); });
  return !stopped; }
function line(dx, dy, n) { var p = []; for (var i = 0; i <= n; i++) p.push([100 + dx * i / n, 100 + dy * i / n]); return p; }
var out = {};
"""


@unittest.skipUnless(NODE, "node not on PATH")
class DragVersusClick(unittest.TestCase):
    def run_gestures(self, body: str) -> dict:
        return run_node(DOM + CONTROLLER + GESTURES + body + "\nprocess.stdout.write(JSON.stringify(out));")

    def test_a_tap_opens_ask_ai_once(self):
        got = self.run_gestures("gesture([[100,100]]); out.toggles = toggles; out.clickThrough = click(); out.requests = requests;")
        self.assertEqual(got["toggles"], 1)
        self.assertFalse(got["clickThrough"], "the click after the tap must not toggle a second time")
        self.assertEqual(got["requests"], [])

    def test_a_small_wobble_is_still_a_tap(self):
        got = self.run_gestures("gesture(line(6, 3, 3)); out.toggles = toggles; out.requests = requests;")
        self.assertEqual((got["toggles"], got["requests"]), (1, []))

    def test_a_horizontal_drag_selects_and_never_opens(self):
        got = self.run_gestures(
            "gesture(line(-150, 0, 30)); out.toggles = toggles; out.clickThrough = click(); out.requests = requests;"
        )
        self.assertEqual(got["toggles"], 0)
        self.assertFalse(got["clickThrough"], "the compatibility click after a drag is swallowed")
        self.assertEqual(got["requests"][0], "start")
        self.assertEqual(got["requests"][-1].split(":")[0], "end")
        self.assertIn("move:1", got["requests"])

    def test_a_vertical_move_neither_drags_nor_opens(self):
        got = self.run_gestures("gesture(line(5, 60, 10)); out.toggles = toggles; out.clickThrough = click(); out.requests = requests;")
        self.assertEqual((got["toggles"], got["requests"], got["clickThrough"]), (0, [], False))

    def test_a_cancelled_drag_settles_without_opening(self):
        got = self.run_gestures(r"""
canvas.fire('pointerdown', ev(100, 100));
for (var i = 1; i <= 20; i++) { now += 16; canvas.fire('pointermove', ev(100 - i * 4, 100)); }
canvas.fire('pointercancel', ev(20, 100));
out.toggles = toggles; out.requests = requests;
""")
        self.assertEqual(got["toggles"], 0)
        self.assertEqual(got["requests"], ["start", "move:1", "end:1"])

    def test_keyboard_activation_is_left_to_the_launcher(self):
        # No pointer gesture: the launcher's own click (Enter / Space) passes.
        got = self.run_gestures("out.clickThrough = click(); out.toggles = toggles;")
        self.assertEqual((got["clickThrough"], got["toggles"]), (True, 0))


if __name__ == "__main__":
    unittest.main()
