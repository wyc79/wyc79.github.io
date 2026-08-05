"""Cross-implementation sync test for the off-topic gate's text handling
(Task 32).

The gate's text normalization -- strip YC's name out of the question unless
what remains is a bio-intent stub ("who is", "介绍", an end-anchored skill
question like "都会什么"), then route by language -- is duplicated in
scripts/chat-widget.js (the widget -- source of truth for visitor-facing
behaviour, per runtime.py's own module docstring) and
chat/src/portfolio_rag/runtime.py (imported normally here). It has already
silently diverged three times in this project's history: a literal
alternation typo in BIO_STUB_RE (`介绍<简介` instead of `介绍|简介`, which made
a whole branch dead), a Python-Unicode-\\w vs JavaScript-ASCII-\\w mismatch
around CJK word boundaries, and the widget's local gate paths scoring the raw
`question` instead of `gateForm()`'s output (fixed in 3c15006). This test
exists so a fourth divergence does not repeat that silently: it EXECUTES both
real implementations -- chat-widget.js via a real `node` subprocess, verbatim
extracted (not retyped, which could drift from the file it mirrors the same
way a fourth copy would), and runtime.py imported normally -- against one
shared probe list and asserts they agree, piece by piece:

  1. NAME_RE (Python) vs NAME_TEST_RE (JS) -- the match decision alone,
     tested directly on the real regex objects (not through any higher-level
     function). This is the layer the Python-Unicode-\\w bug lived in.
  2. strip_name()/stripName() -- the real functions, called for real, not
     reimplemented here. Exercises NAME_RE's substitution AND BIO_STUB_RE's
     keep-vs-strip decision together, end to end.
  3. BIO_STUB_RE (Python) vs BIO_STUB_RE (JS) -- the match decision alone, on
     probes chosen to demonstrate the Chinese stubs match ANYWHERE in the
     string (not just at the start, unlike the English stubs) and that the
     end-anchored skill-stub alternatives really are end-anchored. This is
     the layer the alternation-typo bug lived in.
  4. CJK_RE -- THREE-way, because chat/functions/tencent/index.py keeps its
     own copy too (see the scope note below).
  5. gate_form()/gateForm() -- the full composite, and the one that matters
     most, because it is exactly what reaches the gate. This is the layer
     the "scored the raw question" wiring bug lived in.

Scope note -- this is a three-way comparison for all five items, not just
CJK_RE. It used to be two-way (runtime.py vs chat-widget.js) for items 1-3
and 5, because functions/tencent/index.py received `gate_text` as a
parameter (the widget computed gateForm() and sent the result over the wire)
and never stripped a name itself -- only CJK_RE (used there purely to ROUTE
an already-computed gate_text, not to strip anything) needed the three-way
treatment, via the same lazy-compile-forcing technique
test_retrieval_sync.py's precedent (the reverted Task 28
test_task_request_re_sync.py) used for TASK_REQUEST_RE. Follow-up resolution
changed that: the escalated /chat path now runs an authoritative, server-side
re-gate on a rewrite index.py produces itself, with no client involved, so
index.py computes gate_form() -- and therefore strip_name(), NAME_RE, and
BIO_STUB_RE -- on its own. All five items are live code on a real request
path in all three places now; see the three-way block at the bottom of this
file (items 1-3 and 5, widened) for the tests this added.

The node subprocess plumbing (run a `node -e SCRIPT`, skip -- not fail --
when node is unavailable, extract a `var NAME = ...;` literal or a whole
function body verbatim) is shared with test_retrieval_sync.py and
test_chat_contract_sync.py via chat/tests/_node_harness.py, so this file adds
a third *use* of that harness, not a third *copy* of it.
"""

import importlib.util
import json

import pytest

from portfolio_rag.config import settings
from portfolio_rag.runtime import BIO_STUB_RE, CJK_RE, NAME_RE, gate_form, strip_name
from tests._node_harness import extract_js_function, extract_js_var, node_available, run_node_json

WIDGET_PATH = settings.site_root / "scripts" / "chat-widget.js"
BACKEND_PATH = settings.chat_root / "functions" / "tencent" / "index.py"

# Deliberately spans every shape the brief calls out: no name at all; a name
# plus an off-topic remainder (the strip path, both EN and zh, including the
# widget's OWN "scores 0.61"/"scores high on the name alone" examples from
# its NAME_TEST_RE comment); a name plus a bio stub (the keep path, several
# different stub shapes -- "who is", the empty-remainder case, an
# end-anchored skill stub, and a zh bio marker that is NOT at the start of
# the remainder, to actually exercise "matches anywhere"); a CJK question
# containing 王元辰; a CJK question containing "YC" with NO surrounding
# whitespace (Latin name glued directly to CJK punctuation-free text -- the
# exact case Python's Unicode \w got wrong, verbatim from runtime.py's own
# NAME_RE comment); and the brief's near-miss, which shares "会什么" with the
# skill-stub probe but is not end-anchored there and so must STRIP rather
# than keep the name.
PROBE_STRINGS = [
    # No name at all.
    "what engines has he worked on",
    # Name + off-topic remainder -- EN strip path (chat-widget.js's own
    # NAME_TEST_RE comment: this exact string "scores 0.61").
    "Yuanchen Wang tell me a joke",
    # Name + bio stub ("who is") -- EN keep path.
    "who is Yuanchen Wang",
    # Name + off-topic remainder -- zh strip path (chat-widget.js's own
    # NAME_TEST_RE comment: this exact string "scores high on the name
    # alone").
    "王元辰给我讲个笑话",  # 王元辰给我讲个笑话
    # Name + bio stub ("是谁", matches anywhere -- here it's the whole
    # remainder) -- zh keep path.
    "王元辰是谁",  # 王元辰是谁
    # CJK question containing "YC" with NO space on either side -- the case
    # Python's Unicode \w got wrong without NAME_RE's ASCII-only lookaround
    # (verbatim from runtime.py's own comment). Also a bio stub (介绍) -- keep
    # path.
    "介绍YC这个人",  # 介绍YC这个人
    # End-anchored skill stub -- zh keep path.
    "王元辰都会什么",  # 王元辰都会什么
    # The brief's near-miss: shares "会什么" with the skill stub above, but
    # it is NOT anchored at the end here (more text follows) -- must STRIP,
    # not keep the name.
    "王元辰会什么时候讲笑话",  # 王元辰会什么时候讲笑话
    # Just the name, nothing else -- remainder is the empty string, which
    # BIO_STUB_RE's `^$` alternative treats as a (trivial) bio stub -- EN
    # keep path, and the only probe with no CJK where the name is a bare "YC".
    "YC",
    # Possessive form ("Wang's") + an ordinary non-bio question -- EN strip
    # path, exercises the (?:'s)? group.
    "what has Wang's engine work looked like",
    # zh bio marker (介绍) NOT at the start of the remainder -- demonstrates
    # BIO_STUB_RE's Chinese alternatives match ANYWHERE, unlike the
    # start-anchored English ones.
    "用一段话介绍一下王元辰",  # 用一段话介绍一下王元辰
]

# Standalone probes for BIO_STUB_RE itself (item 3), independent of
# NAME_RE/strip_name -- these feed BIO_STUB_RE directly, the way it is
# actually called (re.search on a remainder-shaped string), to demonstrate
# the "matches anywhere" property explicitly and to probe the end-anchored
# skill-stub alternatives' near-misses directly (the same near-miss shape as
# the brief's headline example, isolated from name-stripping).
BIO_STUB_PROBES = [
    "who is",
    "tell me more about",
    "a total mystery",  # must NOT match -- ordinary text, no stub anywhere
    "介绍一下",  # 介绍一下 -- marker at the start
    "用一段话介绍一下",  # 用一段话介绍一下 -- marker NOT at the start
    "是谁",  # 是谁 -- marker at the end
    "都会什么",  # 都会什么 -- end-anchored skill stub, matches
    "都会什么呢",  # 都会什么呢 -- must NOT match (trailing 呢 isn't in the punct/space class)
    "都会什么的事情",  # 都会什么的事情 -- must NOT match (not end-anchored)
    "",  # explicit ^$ alternative
]


def test_probe_strings_cover_both_strip_and_keep_polarities() -> None:
    """Sanity check on the fixture itself (same precedent test_retrieval_sync.py,
    and the reverted Task 28 test_task_request_re_sync.py, both set): if
    every probe here happened to strip (or none did), the agreement
    assertions below could pass trivially without proving anything."""
    stripped = [s for s in PROBE_STRINGS if strip_name(s) is not None]
    kept = [s for s in PROBE_STRINGS if strip_name(s) is None]
    assert len(stripped) >= 3, "fixture needs several STRIP-path probes to be meaningful"
    assert len(kept) >= 3, "fixture needs several KEEP-path probes to be meaningful"


def test_bio_stub_probes_cover_both_polarities() -> None:
    matched = [s for s in BIO_STUB_PROBES if BIO_STUB_RE.search(s)]
    unmatched = [s for s in BIO_STUB_PROBES if not BIO_STUB_RE.search(s)]
    assert len(matched) >= 3, "fixture needs several matching BIO_STUB_RE probes to be meaningful"
    assert len(unmatched) >= 3, "fixture needs several non-matching BIO_STUB_RE probes to be meaningful"


def test_near_miss_and_skill_stub_probes_go_the_way_the_brief_requires() -> None:
    """The two probes the task brief calls out specifically, checked against
    runtime.py's own strip_name (not reimplemented here): the end-anchored
    skill stub keeps the name, and the near-miss that merely SHARES a word
    with it strips instead -- because BIO_STUB_RE's skill-stub alternative is
    end-anchored and the near-miss's match position is not at the end."""
    assert strip_name("王元辰都会什么") is None, (
        "'王元辰都会什么' is an end-anchored skill stub "
        "-- the name must be KEPT"
    )
    near_miss = strip_name("王元辰会什么时候讲笑话")
    assert near_miss is not None, (
        "'王元辰会什么时候讲笑话' is NOT a bio stub "
        "(会什么 isn't anchored at the end here) -- it must STRIP, not keep the name"
    )
    assert near_miss == "会什么时候讲笑话"


# ── Item 4: CJK_RE, three-way (runtime.py / chat-widget.js / index.py) ─────


def _index_py_cjk_re():
    """Load functions/tencent/index.py fresh and force its lazy CJK_RE
    compile, without a real embedding model -- same technique the reverted
    Task 28 test_task_request_re_sync.py used for TASK_REQUEST_RE: CJK_RE is
    compiled inside gate_decision() before any embedding call happens (see
    that function), so seeding a dummy (non-None) _gates['en'] entry gets
    past the early 'no gate packaged' return, the compile runs, and the
    later `_run_embedding(gate, ...)` call then fails on the dummy entry's
    missing tokenizer/session keys. That failure is expected and swallowed:
    this helper only wants the compiled pattern object the real module code
    produced, not a full gate decision."""
    spec = importlib.util.spec_from_file_location("_scf_index_sync_under_test", str(BACKEND_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._gates["en"] = {
        "task_threshold": None, "threshold": 0.0, "stat": "top",
        "matrix": None, "prefix": "", "pooling": "mean",
    }
    try:
        mod.gate_decision("trigger the lazy compile")
    except KeyError:
        pass  # the dummy entry's missing tokenizer/session, as described above
    assert mod.CJK_RE is not None, (
        "gate_decision did not reach its lazy CJK_RE compile -- index.py's "
        "control flow around it may have changed"
    )
    return mod.CJK_RE


# ── Shared JS execution: one batched node call for items 1, 2, 5 ───────────


def _widget_probe_results(strings: list[str]) -> list[dict]:
    """Execute the widget's own NAME_TEST_RE/CJK_RE (verbatim regex
    literals), stripName() and gateForm() (verbatim function bodies) in a
    real node process against `strings`. Does not reimplement or re-derive
    any of the strip/gate logic -- it runs the widget's own code, extracted,
    not retyped."""
    src = WIDGET_PATH.read_text(encoding="utf-8")
    name_test_re = extract_js_var(src, "NAME_TEST_RE")
    cjk_re = extract_js_var(src, "CJK_RE")
    strip_name_src = extract_js_function(src, "function stripName(")
    gate_form_src = extract_js_function(src, "function gateForm(")
    # gateForm's own source references NAME_STRIP_RE and CJK_RE; stripName's
    # references NAME_TEST_RE, NAME_STRIP_RE and BIO_STUB_RE -- declare all
    # four consts (verbatim) before either function body.
    name_strip_re = extract_js_var(src, "NAME_STRIP_RE")
    bio_stub_re = extract_js_var(src, "BIO_STUB_RE")
    script = (
        f"var NAME_TEST_RE = {name_test_re};\n"
        f"var NAME_STRIP_RE = {name_strip_re};\n"
        f"var BIO_STUB_RE = {bio_stub_re};\n"
        f"var CJK_RE = {cjk_re};\n"
        f"{strip_name_src}\n"
        f"{gate_form_src}\n"
        "const probes = " + json.dumps(strings, ensure_ascii=False) + ";\n"
        "const out = probes.map(function (q) {\n"
        "  var stripped = stripName(q);\n"
        "  return { match: NAME_TEST_RE.test(q), cjk: CJK_RE.test(q), "
        "stripped: stripped, gate: gateForm(q, stripped) };\n"
        "});\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    return run_node_json(script)


def _widget_bio_stub_results(strings: list[str]) -> list[bool]:
    """Execute the widget's own BIO_STUB_RE literal directly against
    `strings` -- isolated from NAME_RE/stripName, mirroring item 3's
    standalone Python-side check."""
    src = WIDGET_PATH.read_text(encoding="utf-8")
    literal = extract_js_var(src, "BIO_STUB_RE")
    script = (
        f"const re = {literal};\n"
        "const strings = " + json.dumps(strings, ensure_ascii=False) + ";\n"
        "process.stdout.write(JSON.stringify(strings.map(s => re.test(s))));\n"
    )
    return run_node_json(script)


# ── Item 1: NAME_RE / NAME_TEST_RE -- match decision only ──────────────────


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_chat_widget_js_agrees_with_runtime_py_on_name_match() -> None:
    widget = _widget_probe_results(PROBE_STRINGS)
    mismatches = [
        (s, bool(NAME_RE.search(s)), w["match"])
        for s, w in zip(PROBE_STRINGS, widget)
        if bool(NAME_RE.search(s)) != w["match"]
    ]
    assert not mismatches, (
        "scripts/chat-widget.js's NAME_TEST_RE disagrees with runtime.py's "
        f"NAME_RE on (string, python_match, js_match): {mismatches!r}"
    )


# ── Item 2: strip_name() / stripName() -- the real functions, called ───────


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_chat_widget_js_agrees_with_runtime_py_on_strip_name() -> None:
    widget = _widget_probe_results(PROBE_STRINGS)
    mismatches = [
        (s, strip_name(s), w["stripped"])
        for s, w in zip(PROBE_STRINGS, widget)
        if strip_name(s) != w["stripped"]
    ]
    assert not mismatches, (
        "scripts/chat-widget.js's stripName() disagrees with runtime.py's "
        f"strip_name() on (string, python_result, js_result): {mismatches!r}"
    )


# ── Item 3: BIO_STUB_RE -- match decision only, "matches anywhere" ─────────


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_chat_widget_js_agrees_with_runtime_py_on_bio_stub_re() -> None:
    js_results = _widget_bio_stub_results(BIO_STUB_PROBES)
    py_results = [bool(BIO_STUB_RE.search(s)) for s in BIO_STUB_PROBES]
    mismatches = [
        (s, py, js) for s, py, js in zip(BIO_STUB_PROBES, py_results, js_results) if py != js
    ]
    assert not mismatches, (
        f"scripts/chat-widget.js's BIO_STUB_RE disagrees with runtime.py's on "
        f"(string, python_result, js_result): {mismatches!r}"
    )


# ── Item 4: CJK_RE -- three-way ─────────────────────────────────────────────


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_chat_widget_js_and_index_py_agree_with_runtime_py_on_cjk_re() -> None:
    widget = _widget_probe_results(PROBE_STRINGS)
    backend_cjk_re = _index_py_cjk_re()
    py_results = [bool(CJK_RE.search(s)) for s in PROBE_STRINGS]
    backend_results = [bool(backend_cjk_re.search(s)) for s in PROBE_STRINGS]
    js_results = [w["cjk"] for w in widget]

    py_vs_js = [
        (s, py, js) for s, py, js in zip(PROBE_STRINGS, py_results, js_results) if py != js
    ]
    assert not py_vs_js, (
        f"scripts/chat-widget.js's CJK_RE disagrees with runtime.py's on "
        f"(string, python_result, js_result): {py_vs_js!r}"
    )
    py_vs_backend = [
        (s, py, be) for s, py, be in zip(PROBE_STRINGS, py_results, backend_results) if py != be
    ]
    assert not py_vs_backend, (
        f"functions/tencent/index.py's CJK_RE disagrees with runtime.py's on "
        f"(string, python_result, backend_result): {py_vs_backend!r}"
    )


# ── Item 5: gate_form() / gateForm() -- the full composite ─────────────────


@pytest.mark.skipif(not node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_chat_widget_js_agrees_with_runtime_py_on_gate_form() -> None:
    """The one that matters most: gate_form()/gateForm()'s output is exactly
    what reaches the off-topic gate (see runtime.py's gate_form docstring
    and chat-widget.js's own comment above gateForm). This is also the layer
    the widget's local-gate wiring bug lived in (fixed in 3c15006) -- a test
    at the NAME_RE/BIO_STUB_RE layer alone would not have caught that."""
    widget = _widget_probe_results(PROBE_STRINGS)
    mismatches = [
        (s, gate_form(s), w["gate"])
        for s, w in zip(PROBE_STRINGS, widget)
        if gate_form(s) != w["gate"]
    ]
    assert not mismatches, (
        "scripts/chat-widget.js's gateForm() disagrees with runtime.py's "
        f"gate_form() on (string, python_result, js_result): {mismatches!r}"
    )


# ── Items 1-3 and 5, widened to three-way (Task: follow-up resolution) ──────
#
# index.py used to receive gate_text as a parameter and never strip a name
# itself, which is why the scope note above says "two implementations, not
# three" for everything except CJK_RE. The authoritative re-gate on the
# escalated /chat path changed that: index.py now computes gate_form() on a
# rewrite it produced itself, with no client involved, so its copies of
# NAME_RE / BIO_STUB_RE / strip_name / gate_form are live code on a real
# request path and get the same treatment every other copy does.


def _index_py_module():
    """Load functions/tencent/index.py fresh. Unlike _index_py_cjk_re above,
    nothing here needs the lazy CJK_RE compile forced by hand -- gate_form()
    calls _ensure_cjk_re() itself."""
    spec = importlib.util.spec_from_file_location("_scf_index_gateform_under_test", str(BACKEND_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_index_py_agrees_with_runtime_py_on_name_match() -> None:
    mod = _index_py_module()
    mismatches = [
        (s, bool(NAME_RE.search(s)), bool(mod.NAME_RE.search(s)))
        for s in PROBE_STRINGS
        if bool(NAME_RE.search(s)) != bool(mod.NAME_RE.search(s))
    ]
    assert not mismatches, (
        "functions/tencent/index.py's NAME_RE disagrees with runtime.py's on "
        f"(string, runtime_match, index_match): {mismatches!r}"
    )


def test_index_py_agrees_with_runtime_py_on_bio_stub_re() -> None:
    mod = _index_py_module()
    mismatches = [
        (s, bool(BIO_STUB_RE.search(s)), bool(mod.BIO_STUB_RE.search(s)))
        for s in BIO_STUB_PROBES
        if bool(BIO_STUB_RE.search(s)) != bool(mod.BIO_STUB_RE.search(s))
    ]
    assert not mismatches, (
        "functions/tencent/index.py's BIO_STUB_RE disagrees with runtime.py's on "
        f"(string, runtime_result, index_result): {mismatches!r}"
    )


def test_index_py_agrees_with_runtime_py_on_strip_name() -> None:
    mod = _index_py_module()
    mismatches = [
        (s, strip_name(s), mod.strip_name(s))
        for s in PROBE_STRINGS
        if strip_name(s) != mod.strip_name(s)
    ]
    assert not mismatches, (
        "functions/tencent/index.py's strip_name() disagrees with runtime.py's on "
        f"(string, runtime_result, index_result): {mismatches!r}"
    )


def test_index_py_agrees_with_runtime_py_on_gate_form() -> None:
    """The composite, and the one that matters: this is exactly the text the
    authoritative re-gate judges on the escalated /chat path. A divergence here
    means the server gates a rewrite differently than the widget would have
    gated the same string -- the failure mode the two-way version of this test
    was written for, now with a third place to go wrong."""
    mod = _index_py_module()
    mismatches = [
        (s, gate_form(s), mod.gate_form(s))
        for s in PROBE_STRINGS
        if gate_form(s) != mod.gate_form(s)
    ]
    assert not mismatches, (
        "functions/tencent/index.py's gate_form() disagrees with runtime.py's on "
        f"(string, runtime_result, index_result): {mismatches!r}"
    )
