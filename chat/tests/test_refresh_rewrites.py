"""scripts/refresh_rewrites.py's positive-echoed-unchanged WARNING.

The script itself calls a real LLM (requires LLM_API_KEY) and is never run by
pytest/run_eval.py -- see its own module docstring -- but the comparison that
decides whether a positive case "echoed unchanged" is pure and worth pinning
directly, the same way test_run_eval_cli.py drives run_eval.py's main() with
a stubbed runtime rather than a real one. Loaded via importlib for the same
reason: this is a script, not a package module (mirrors
test_build_package.py / test_run_eval_cli.py).

Motivating bug, caught against the live rewrite API: a zh follow-up positive
whose question was a narrative continuation ("那后来", "and then?") -- with
no referent to resolve -- came back as "那后来？", the same text plus a
trailing full-width question mark. `rewrite_question` reported outcome
"rewritten", and a bare `out[c.id] == c.q` comparison missed the collision
entirely, so the WARNING this file exists to catch never fired for a case
that, in substance, was never actually resolved.
"""

import importlib.util

import pytest

from portfolio_rag.config import settings
from portfolio_rag.evaluation import GoldenCase

REFRESH_REWRITES_PATH = settings.chat_root / "scripts" / "refresh_rewrites.py"


def _load_refresh_rewrites():
    spec = importlib.util.spec_from_file_location(
        "_refresh_rewrites_under_test", str(REFRESH_REWRITES_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_refresh_rewrites()


def _case(id_: str, q: str, lang: str = "zh") -> GoldenCase:
    return GoldenCase(
        id=id_, role="visitor", lang=lang, type="positive", q=q,
        history=(("user", "u"), ("assistant", "a")),
        expected_urls=("pages/education.html",), expected_keywords=("kw",),
    )


class _StubBackend:
    """Only rewrite_question -- the one attribute main() reads off _backend()."""

    def __init__(self, text: str, outcome: str):
        self._text = text
        self._outcome = outcome

    def rewrite_question(self, messages, q):
        return self._text, self._outcome


def _run(mod, monkeypatch, capsys, case: GoldenCase, text: str, outcome: str) -> str:
    monkeypatch.setattr(mod, "_backend", lambda: _StubBackend(text, outcome))
    monkeypatch.setattr(mod, "load_cases", lambda: [case])
    monkeypatch.setattr("sys.argv", ["refresh_rewrites.py", "--dry-run"])
    rc = mod.main()
    assert rc == 0
    return capsys.readouterr().err


# --- _echo_key, the pure normalization the comparison relies on -------------


def test_echo_key_strips_a_trailing_full_width_question_mark(mod) -> None:
    assert mod._echo_key("那后来？") == mod._echo_key("那后来")


def test_echo_key_strips_trailing_ascii_punctuation_and_whitespace(mod) -> None:
    assert mod._echo_key("is there tuning work? ") == mod._echo_key("is there tuning work")


def test_echo_key_strips_trailing_full_width_period_and_exclamation(mod) -> None:
    assert mod._echo_key("再后来。") == mod._echo_key("再后来")
    assert mod._echo_key("再后来！") == mod._echo_key("再后来")


def test_echo_key_does_not_touch_interior_punctuation(mod) -> None:
    """Only the TRAILING edge is normalized -- a genuine rewrite that adds
    real content must not be conflated with an echo just because both
    strings happen to end in the same punctuation."""
    a = mod._echo_key("Is there tuning work on Prime Engine?")
    b = mod._echo_key("what about tuning it")
    assert a != b


# --- the WARNING itself, end to end through main() ---------------------------


def test_punctuation_only_change_is_still_flagged_as_an_echo(mod, monkeypatch, capsys) -> None:
    """The exact live-API symptom that motivated this fix: outcome says
    'rewritten', but the only difference is a trailing '？'. Must still warn."""
    case = _case("followup-zh-02", "那后来")
    err = _run(mod, monkeypatch, capsys, case, text="那后来？", outcome="rewritten")

    assert "WARNING" in err
    assert "followup-zh-02" in err


def test_a_true_unchanged_echo_is_still_flagged(mod, monkeypatch, capsys) -> None:
    """Control: the original (pre-fix) behavior this comparison already
    covered must not regress."""
    case = _case("followup-zh-01", "再后来")
    err = _run(mod, monkeypatch, capsys, case, text="再后来", outcome="echoed")

    assert "WARNING" in err
    assert "followup-zh-01" in err


def test_a_genuine_rewrite_is_not_flagged(mod, monkeypatch, capsys) -> None:
    """Control, the other direction: real resolution -- new content, not just
    trailing punctuation -- must not false-positive."""
    case = _case("followup-en-01", "what about tuning it", lang="en")
    err = _run(
        mod, monkeypatch, capsys, case,
        text="Is there any tuning or optimization work on Prime Engine?",
        outcome="rewritten",
    )

    assert "WARNING" not in err


def test_a_non_positive_case_is_never_flagged_even_if_echoed(mod, monkeypatch, capsys) -> None:
    """The WARNING is scoped to type == 'positive' -- a post-context negative
    is SUPPOSED to echo (see neg-post-zh-01/02's whole premise), so an
    unchanged post-context negative must not trip it."""
    negative = GoldenCase(
        id="neg-post-zh-01", role="shared", lang="zh", type="off_topic", adjacency="easy",
        q="给我写首诗", history=(("user", "u"), ("assistant", "a")),
    )
    err = _run(mod, monkeypatch, capsys, negative, text="给我写首诗", outcome="echoed")

    assert "WARNING" not in err
