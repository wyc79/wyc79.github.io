"""Shared node-subprocess plumbing for the cross-implementation sync tests
(test_retrieval_sync.py, test_chat_contract_sync.py,
test_implementation_sync.py).

All three tests execute real, verbatim-extracted fragments of
scripts/chat-widget.js in a real `node` subprocess rather than
reimplementing JS logic in Python -- a reimplementation could drift from the
file it's meant to mirror the same way a fourth copy would (see each test's
own module docstring for the project's history of exactly that kind of silent
divergence). This module is only the "run node -e SCRIPT, get JSON back"
plumbing and the generic verbatim-extraction helpers (grab a `var NAME = ...;`
literal, or a whole named function's body by brace-matching); each test file
still owns the bespoke logic for WHAT it extracts and how it composes the
extracted pieces into a runnable script.

Factored out because this exact plumbing was duplicated three times over
(twice within test_chat_contract_sync.py alone, once more in
test_retrieval_sync.py) -- three copies of the same node harness would itself
become a divergence source.
"""

import json
import shutil
import subprocess


def node_available() -> bool:
    return shutil.which("node") is not None


def run_node(script: str) -> subprocess.CompletedProcess:
    """Run `script` via `node -e`, asserting it exited cleanly."""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30
    )
    assert result.returncode == 0, f"node failed: {result.stderr}"
    return result


def run_node_json(script: str):
    """Run `script` via node and parse its stdout as JSON."""
    return json.loads(run_node(script).stdout)


def extract_js_var(src: str, name: str) -> str:
    """Pull the literal right-hand side out of `var {name} = <literal>;`,
    verbatim -- not retyped. Used for single-line consts/regex literals that
    contain no unescaped ';' of their own (true of every const this project
    extracts this way; a literal that violated it would need the brace
    matcher below instead)."""
    marker = f"var {name} = "
    start = src.index(marker) + len(marker)
    end = src.index(";", start)
    return src[start:end]


def _matching_brace(src: str, at_or_after: int) -> int:
    """Index of the '}' closing the first '{' at or after `at_or_after`.

    Counts braces without tracking string literals, which is sound only because
    every construct this module extracts happens to contain no braces inside
    its own strings or comments. That has always been true of the functions
    extracted here; it is worth rechecking before pointing this at something
    new."""
    brace_start = src.index("{", at_or_after)
    depth, i = 0, brace_start
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1


def extract_js_object(src: str, name: str) -> str:
    """Pull a multi-line `var {name} = {...};` object literal out verbatim, by
    brace matching rather than by scanning for the terminating ';'. Needed for
    literals whose own values contain a ';' (a function body, say) -- STR, the
    widget's i18n table, is the case this exists for; extract_js_var's
    first-semicolon rule truncates it mid-table."""
    marker = f"var {name} = "
    start = src.index(marker) + len(marker)
    return src[start : _matching_brace(src, start) + 1]


def extract_js_function(src: str, fn_marker: str) -> str:
    """Verbatim function-body extraction: find fn_marker (e.g.
    'function scoreChunks(' or 'async function askWorker('), then match
    braces from the first '{' after it to its corresponding close."""
    fn_start = src.index(fn_marker)
    return src[fn_start : _matching_brace(src, fn_start) + 1]
