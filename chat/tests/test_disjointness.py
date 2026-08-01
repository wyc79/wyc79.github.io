"""calibration <-> corpus and golden <-> corpus disjointness (Task 26).

Three sets, two "scored against" edges (see gate_calibration.py's module
docstring and this project's eval/README.md):

    calibration (gate_calibration.ON_TOPIC/OFF_TOPIC/_ZH) --derives--> threshold
    golden (eval/golden.jsonl)                            --validates-> threshold
    corpus (knowledge/about_en.md, about_zh.md)            scored AGAINST by both

test_golden.py's test_cases_are_disjoint_from_fit_data already guards golden
against reusing calibration/role-starter TEXT, but only by exact normalized
equality -- it passes on a reworded near-paraphrase, which is exactly the
shape of collision measured (elsewhere in this project) to inflate a score by
about +0.2. And nothing at all previously compared either query set against
the corpus. This file closes both gaps.

Comparator: character-level longest common substring (LCS), not word
tokenization. Word tokenization is ASCII-shaped (split on whitespace) and is
blind to Chinese, which has no word-boundary spaces -- exactly the mistake
that produced a false-clean disjointness result earlier in this project (see
gate_calibration.py's module comment). Character LCS needs no tokenizer and
works identically on Latin and CJK text.

Only a section's BODY is ever embedded -- index_builder.py embeds `s.text`
for the gate corpus and chunks built from load_site/load_knowledge; the
`## Heading` (Section.section_title) is metadata carried for display, never
vectorised (see loader.py's Section dataclass and load_knowledge()). So this
compares queries against load_knowledge() bodies specifically, not against
raw markdown (which would also drag in headings and manufacture false
positives out of harmless heading/query wording overlap).

Threshold: a single pair of constants (_ABS_EFFECTIVE_THRESHOLD,
_RATIO_THRESHOLD), justified below where they're defined, applied identically
to Latin and CJK text via loader._effective_length's existing script weight
(CJK characters count for _CJK_WEIGHT=2.5 Latin characters -- the same
weighting load_knowledge() already uses to size-floor a section, reused here
for the same reason: one CJK character carries more of a phrase's identity
than one Latin character, so raw character counts are not comparable across
scripts). This means ONE set of thresholds, not two independently-chosen
raw-character ones, while still behaving as if Latin and CJK had different
cutoffs -- the effective-length weighting *is* what makes them different.
"""

import pytest

from portfolio_rag.config import settings
from portfolio_rag.evaluation import GOLDEN_PATH, load_cases
from portfolio_rag.gate_calibration import OFF_TOPIC, OFF_TOPIC_ZH, ON_TOPIC, ON_TOPIC_ZH
from portfolio_rag.loader import _effective_length, load_knowledge

KNOWLEDGE_DIR = settings.chat_root / "knowledge"


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _lcs(a: str, b: str) -> str:
    """Longest common (contiguous) substring of a and b.

    Standard O(len(a) * len(b)) time, O(len(b)) space DP -- only the running
    row is kept since only the best cell's value (and its text) is needed,
    not the full table. Corpus bodies here are short (well under 300 chars,
    see the survey in task-26-report.md), so this is fast; there is no need
    for a suffix-automaton or other sub-quadratic approach at this scale.
    """
    if not a or not b:
        return ""
    prev = [0] * (len(b) + 1)
    best_len = 0
    best_end = 0  # end index in `a` of the best match found so far
    for i, ca in enumerate(a, start=1):
        curr = [0] * (len(b) + 1)
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len = curr[j]
                    best_end = i
        prev = curr
    return a[best_end - best_len : best_end]


# --- threshold ---------------------------------------------------------
#
# _ABS_EFFECTIVE_THRESHOLD: flags a shared span whose effective length (see
# loader._effective_length) is at least this many Latin-equivalent
# characters, regardless of how long the query itself is. Chosen from a
# direct survey of every current calibration and golden query against its
# language's corpus (chat/eval/README.md-adjacent — see task-26-report.md for
# the full table): the largest share seen for any ALREADY-ACCEPTED pair --
# all of them a shared proper noun or multi-word domain term ("grapple
# traversal", "Prime Engine", "game development", "Yuanchen Wang",
# "游戏引擎开发经验" / "game engine development experience") -- is an
# effective length of 25 (English) and 20 (Chinese, i.e. 8 raw CJK
# characters * 2.5). 30 sits above BOTH with headroom, so it does not fire
# on any term/name every on-topic query about this portfolio must legitimately
# share with the corpus, while still being far short of what a genuinely
# lifted clause measures (a duplicated 6-8 word clause running 40-90+
# effective characters in this project's own historical incidents -- see the
# module docstring above and gate_calibration.py's "Harvard biomedical
# informatics background" / "frustum culling implementation" note).
_ABS_EFFECTIVE_THRESHOLD = 30

# _RATIO_THRESHOLD: flags a shared span covering at least this fraction of
# the QUERY's own effective length, independent of absolute size. This
# catches the other collision shape the absolute rule alone would miss: a
# short calibration query (as little as ~15-20 characters, e.g. "resume
# highlights") mirrored ALMOST WHOLESALE by a corpus entry -- exactly the
# measured 0.5153->0.7381 score-inflation incident in the task brief, where
# the corpus, not a long shared clause, is what got added. The highest ratio
# seen on any currently-accepted pair (again: a shared term, not sentence
# reuse -- "shipped games list" against a section using "shipped games") is
# 0.78; 0.8 sits just above it. Effective length cancels out of this ratio
# for any single-script (pure Latin or pure CJK) pair -- the same weight
# multiplies numerator and denominator alike -- so one ratio threshold
# already behaves correctly for both scripts without further adjustment.
_RATIO_THRESHOLD = 0.8


def _overlap(query: str, body: str) -> tuple[float, float, str]:
    """(effective LCS length, ratio of query's effective length, matched text)."""
    shared = _lcs(_normalize(query), _normalize(body))
    eff_shared = _effective_length(shared)
    eff_query = _effective_length(_normalize(query)) or 1.0
    return eff_shared, eff_shared / eff_query, shared


def _find_collisions(queries, sections) -> list[dict]:
    """sections: iterable of (label, body_text), e.g. from load_knowledge()'s
    Section objects as (s.section_title, s.text) -- label is diagnostic only,
    never compared against (see the module docstring: headings aren't
    embedded, so heading/query overlap is not a real collision)."""
    collisions = []
    for q in queries:
        worst = (0.0, 0.0, "", "")
        for label, body in sections:
            eff_shared, ratio, shared = _overlap(q, body)
            if eff_shared > worst[0]:
                worst = (eff_shared, ratio, shared, label)
        eff_shared, ratio, shared, label = worst
        if eff_shared >= _ABS_EFFECTIVE_THRESHOLD or ratio >= _RATIO_THRESHOLD:
            collisions.append(
                {
                    "query": q,
                    "section": label,
                    "shared": shared,
                    "eff_len": round(eff_shared, 1),
                    "ratio": round(ratio, 3),
                }
            )
    return collisions


def _format(collisions: list[dict]) -> str:
    lines = [
        f"{c['query']!r} <-> section {c['section']!r}: shared={c['shared']!r} "
        f"eff_len={c['eff_len']} ratio={c['ratio']}"
        for c in collisions
    ]
    return "near-duplicate query/corpus pair(s):\n  " + "\n  ".join(lines)


# --- checker self-validation: prove it can actually catch something -----
# (task-26-report.md records this same evidence for the human reader; these
# are the executable form of it, so a future edit to the checker re-proves
# it instead of only trusting a comment.)


def test_checker_flags_a_planted_english_collision() -> None:
    """A near-verbatim clause (not just a shared term) must be caught."""
    body = (
        "He designed and implemented grapple traversal as a core movement "
        "mechanic in Cemented Dreams."
    )
    planted_query = (
        "Tell me whether he designed and implemented grapple traversal as a "
        "core movement mechanic."
    )
    collisions = _find_collisions([planted_query], [("planted-en", body)])
    assert collisions, "checker failed to catch a verbatim-clause collision (EN)"
    assert collisions[0]["eff_len"] >= _ABS_EFFECTIVE_THRESHOLD


def test_checker_does_not_flag_a_bare_shared_proper_noun_english() -> None:
    """A query and a section may legitimately share a proper noun / technical
    term (Prime Engine, Unity, Blueprint, RAG, ...) without that being
    sentence-level reuse -- the checker must not cry wolf on this shape."""
    body = "The Prime Engine handles hierarchical culling for large outdoor scenes."
    query = "Does the Prime Engine support dynamic lighting changes at runtime?"
    collisions = _find_collisions([query], [("term-only-en", body)])
    assert not collisions, f"false positive on a shared-term-only pair (EN): {collisions}"


def test_checker_flags_a_planted_chinese_collision() -> None:
    """Character-level, not word-tokenized: this clause has no ASCII word
    boundaries at all, so a whitespace-splitting checker would report zero
    overlap here regardless of the truth -- the exact failure mode named in
    the module docstring. The checker must still catch it."""
    body = (
        "他在南加州大学攻读游戏开发方向的计算机科学硕士学位，"
        "同时做过战斗设计和引擎编程工作。"
    )
    planted_query = "请问他是不是在南加州大学攻读游戏开发方向的计算机科学硕士学位？"
    collisions = _find_collisions([planted_query], [("planted-zh", body)])
    assert collisions, "checker failed to catch a verbatim-clause collision (ZH)"
    assert collisions[0]["eff_len"] >= _ABS_EFFECTIVE_THRESHOLD


def test_checker_does_not_flag_a_bare_shared_proper_noun_chinese() -> None:
    """王元辰 (his name) is exactly the kind of unavoidable, required shared
    term the brief calls out -- two questions both naming him must not, on
    that basis alone, read as a collision."""
    body = "王元辰目前在做一个自动微分工具项目，也做过关卡设计相关的工作。"
    query = "王元辰最近在忙什么项目？"
    collisions = _find_collisions([query], [("term-only-zh", body)])
    assert not collisions, f"false positive on a shared-term-only pair (ZH): {collisions}"


# --- the real disjointness tests ----------------------------------------


@pytest.fixture(scope="module")
def en_corpus() -> list[tuple[str, str]]:
    return [(s.section_title, s.text) for s in load_knowledge(KNOWLEDGE_DIR, "en")]


@pytest.fixture(scope="module")
def zh_corpus() -> list[tuple[str, str]]:
    return [(s.section_title, s.text) for s in load_knowledge(KNOWLEDGE_DIR, "zh")]


def test_calibration_en_is_disjoint_from_the_en_corpus(en_corpus) -> None:
    collisions = _find_collisions(ON_TOPIC + OFF_TOPIC, en_corpus)
    assert not collisions, _format(collisions)


def test_calibration_zh_is_disjoint_from_the_zh_corpus(zh_corpus) -> None:
    collisions = _find_collisions(ON_TOPIC_ZH + OFF_TOPIC_ZH, zh_corpus)
    assert not collisions, _format(collisions)


def test_golden_en_is_disjoint_from_the_en_corpus(en_corpus) -> None:
    queries = [c.q for c in load_cases(GOLDEN_PATH) if c.lang == "en"]
    collisions = _find_collisions(queries, en_corpus)
    assert not collisions, _format(collisions)


def test_golden_zh_is_disjoint_from_the_zh_corpus(zh_corpus) -> None:
    queries = [c.q for c in load_cases(GOLDEN_PATH) if c.lang == "zh"]
    collisions = _find_collisions(queries, zh_corpus)
    assert not collisions, _format(collisions)
