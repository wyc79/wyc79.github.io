"""Cross-implementation sync test for TASK_REQUEST_RE (Task 28).

The pattern is deliberately duplicated three times: portfolio_rag.runtime
(imported normally here), scripts/chat-widget.js (the browser widget --
source of truth for visitor-facing behaviour, per runtime.py's own module
docstring), and functions/tencent/index.py (the SCF backend, which must stay
stdlib-only and so cannot import runtime.py). BIO_STUB_RE has already
drifted between implementations twice in this project's history -- once as
a literal alternation typo (`介绍<简介` instead of `|`), once as a Python
Unicode-\\w / JS ASCII-\\w mismatch (see runtime.py's TASK_REQUEST_RE
docstring for both). This test exists so a fourth synchronized regex does
not repeat that silently: it EXECUTES all three real files (not a
reimplementation of the pattern in test code, which could drift from all
three the same way a fourth copy would) against one shared list of strings
and asserts they agree, case by case.

The JS copy is run for real via `node` (present on this machine; the test
SKIPS, not fails, when node is unavailable on PATH -- a missing dev tool is
a different claim than "the regexes disagree," so it must not read as a
failure). The index.py copy is loaded via importlib from its actual file on
disk; its module-level imports are stdlib-only (the ONNX/tokenizers
dependencies are deferred into function bodies), so a plain module load is
safe and cheap here -- see functions/tencent/index.py's own module docstring.
"""

import importlib.util
import json
import shutil
import subprocess

import pytest

from portfolio_rag.config import settings
from portfolio_rag.runtime import TASK_REQUEST_RE

WIDGET_PATH = settings.site_root / "scripts" / "chat-widget.js"
BACKEND_PATH = settings.chat_root / "functions" / "tencent" / "index.py"

# Deliberately spans every shape this task's design cares about: several
# DIFFERENT English/Chinese trigger phrases (not five variations on "give
# me"), near-miss text that shares a WORD with a trigger but must not match
# ("written" vs "write", "tell me about" vs "tell me a", "介绍" bio-stub vs
# "帮我"/"给我"/"麻烦"), and the two real golden shapes that motivated this
# whole design (visitor-en-01's exact wording, and its zh counterpart).
TEST_STRINGS = [
    # Flagged: EN, several distinct trigger phrases.
    "give me a summary",
    "write me a poem",
    "write my homework essay",
    "translate this to french",
    "summarise his engine work for me",
    "tell me a joke",
    "help me understand his research",
    "walk me through his engine work",
    "break down his combat design work for me",
    "reply as YC and tell me a story",
    "act as him and answer my question",
    # NOT flagged: EN, ordinary bio questions -- including near-misses that
    # share a word with a trigger phrase.
    "what did he study",
    "does he know Unity?",
    "tell me about him",
    "who is Yuanchen Wang?",
    "has he written actual shader code himself",
    "Can he implement collision response from scratch?",
    # The real golden case that forced the "second tier, not override" design.
    "What about Yuanchen Wang — could you give me the quick version "
    "before pointing me toward whatever's worth clicking on first?",
    # Flagged: ZH, distinct trigger markers (帮我/给我/麻烦).
    "帮我写作业",  # 帮我写作业
    "给我讲讲他的战斗设计经历",  # 给我讲讲他的战斗设计经历
    "麻烦介绍一下他的科研背景",  # 麻烦介绍一下他的科研背景
    # NOT flagged: ZH bio-stub / ordinary questions.
    "介绍一下他的游戏引擎开发经验",  # 介绍一下他的游戏引擎开发经验
    "他之前是学什么的",  # 他之前是学什么的
    # The zh golden case flagged by 给我 (give me an impression).
    "关于王元辰，可以先给我一个"
    "大概的印象吗，他到底是干哪一行？",
]


def test_test_strings_cover_both_polarities_in_both_languages() -> None:
    """Sanity check on the fixture itself: if every string here happened to
    be flagged (or none were), the agreement tests below would pass
    trivially without proving anything."""
    flagged = [s for s in TEST_STRINGS if TASK_REQUEST_RE.search(s)]
    unflagged = [s for s in TEST_STRINGS if not TASK_REQUEST_RE.search(s)]
    assert len(flagged) >= 4, "fixture needs several flagged strings to be meaningful"
    assert len(unflagged) >= 4, "fixture needs several unflagged strings to be meaningful"


def _index_py_task_request_re():
    """Load functions/tencent/index.py fresh and force its lazy TASK_REQUEST_RE
    compile, without a real embedding model. TASK_REQUEST_RE is compiled
    inside gate_decision() BEFORE any embedding call happens (see that
    function) -- populating _gates["en"] with a dummy (non-None) entry gets
    past the early "no gate packaged" return, the compile runs, and the
    later `gate["matrix"] @ ...` call then fails on the dummy None matrix.
    That failure is expected and swallowed: this helper only wants the
    compiled pattern object the real module code produced, not a full gate
    decision."""
    spec = importlib.util.spec_from_file_location("_scf_index_under_test", str(BACKEND_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._gates["en"] = {
        "task_threshold": None, "threshold": 0.0, "stat": "top",
        "matrix": None, "prefix": "", "pooling": "mean",
    }
    try:
        mod.gate_decision("trigger the lazy compile")
    except Exception:
        pass
    assert mod.TASK_REQUEST_RE is not None, (
        "gate_decision did not reach its lazy TASK_REQUEST_RE compile -- "
        "index.py's control flow around it may have changed"
    )
    return mod.TASK_REQUEST_RE


def test_index_py_agrees_with_runtime_py() -> None:
    backend_re = _index_py_task_request_re()
    mismatches = [
        s for s in TEST_STRINGS
        if bool(TASK_REQUEST_RE.search(s)) != bool(backend_re.search(s))
    ]
    assert not mismatches, (
        "functions/tencent/index.py's TASK_REQUEST_RE disagrees with "
        f"runtime.py's on: {mismatches!r}"
    )


def _node_available() -> bool:
    return shutil.which("node") is not None


def _widget_task_request_re_results(strings: list[str]) -> list[bool]:
    """Extract the `var TASK_REQUEST_RE = /.../i;` literal straight out of
    the widget source and execute it in a real Node process against
    `strings`, returning one bool per string. Does not reimplement or
    re-derive the pattern -- it runs the widget's own line verbatim."""
    src = WIDGET_PATH.read_text(encoding="utf-8")
    marker = "var TASK_REQUEST_RE = "
    start = src.index(marker) + len(marker)
    end = src.index(";\n", start)
    literal = src[start:end]  # e.g. "/\\b(?:give\\s+me|...)\\b|.../i"

    script = (
        "const re = " + literal + ";\n"
        "const strings = " + json.dumps(strings, ensure_ascii=False) + ";\n"
        "process.stdout.write(JSON.stringify(strings.map(s => re.test(s))));\n"
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30
    )
    assert result.returncode == 0, f"node failed: {result.stderr}"
    return json.loads(result.stdout)


@pytest.mark.skipif(not _node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_chat_widget_js_agrees_with_runtime_py() -> None:
    js_results = _widget_task_request_re_results(TEST_STRINGS)
    py_results = [bool(TASK_REQUEST_RE.search(s)) for s in TEST_STRINGS]
    mismatches = [
        (s, py, js) for s, py, js in zip(TEST_STRINGS, py_results, js_results) if py != js
    ]
    assert not mismatches, (
        f"scripts/chat-widget.js's TASK_REQUEST_RE disagrees with runtime.py's on "
        f"(string, python_result, js_result): {mismatches!r}"
    )


@pytest.mark.skipif(not _node_available(), reason="node not on PATH -- cannot execute the JS copy")
def test_chat_widget_js_agrees_with_index_py() -> None:
    """All three, not just JS-vs-Python-mirror: the SCF backend and the
    widget must agree directly too, since a visitor's real request is judged
    by whichever of these two actually runs (widget locally/degraded,
    backend when reachable) -- runtime.py is the test harness's own mirror,
    not something a visitor's request ever touches."""
    backend_re = _index_py_task_request_re()
    js_results = _widget_task_request_re_results(TEST_STRINGS)
    backend_results = [bool(backend_re.search(s)) for s in TEST_STRINGS]
    mismatches = [
        (s, be, js) for s, be, js in zip(TEST_STRINGS, backend_results, js_results) if be != js
    ]
    assert not mismatches, (
        "functions/tencent/index.py's TASK_REQUEST_RE disagrees with "
        f"scripts/chat-widget.js's on (string, backend_result, js_result): {mismatches!r}"
    )
