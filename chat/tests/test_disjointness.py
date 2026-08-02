"""calibration <-> corpus and golden <-> corpus disjointness (Task 26).

Three sets, two "scored against" edges (see gate_calibration.py's module
docstring and this project's eval/README.md):

    calibration (gate_calibration.ON_TOPIC/OFF_TOPIC/_ZH) --derives--> threshold
    golden (eval/golden.jsonl)                            --validates-> threshold
    corpus (knowledge/about_en.md, about_zh.md)            scored AGAINST by both

test_golden.py's test_cases_are_disjoint_from_fit_data already guards golden
against reusing calibration/role-starter TEXT, but only by exact normalized
equality -- it passes on any reworded near-paraphrase, exact or not, which is
exactly the shape of collision measured (elsewhere in this project) to
inflate a score by about +0.2. And nothing at all previously compared either
query set against the corpus. This file adds that comparison for both.

Comparator: character-level longest common substring (LCS), not word
tokenization. Word/whitespace tokenization is ASCII-shaped and is blind to
Chinese, which has no word-boundary spaces -- exactly the mistake that
produced a false-clean disjointness result earlier in this project (see
gate_calibration.py's module comment). Character LCS needs no tokenizer and
works identically on Latin and CJK text. The DP directly maximizes
EFFECTIVE (CJK-weighted, see loader._effective_length) length, not raw
character count -- see _lcs_effective's docstring for why raw-length-first
is an outright wrong answer, not just a cosmetic tie-break, whenever a
shorter, CJK-dense candidate substring competes with a longer, Latin-only one.

**Known, deliberately-not-closed gap:** LCS only ever finds a CONTIGUOUS,
in-order span. A paraphrase that reorders clauses or swaps word order
within a clause (active/passive, subject/object inversion, "X does Y"
turned into "was Y done by X") is not caught, because reordering breaks the
contiguous run itself -- there is no shape of "checked in a different
order" that rescues it, only genuine semantic (embedding) comparison would.
This was checked, not assumed, and closing it was attempted twice:

1. A per-clause LCS variant (split the query on clause boundaries, check
   each clause against the body independently) was implemented, then proven
   INERT and removed: a clause is always a contiguous substring of the whole
   query, so any common substring a clause can find is already a common
   substring the whole-string search finds too -- per-clause LCS can never
   exceed whole-string LCS, by construction. Confirmed empirically: replaying
   the two "clause-reordered, now caught" examples that motivated it through
   plain whole-string LCS gave IDENTICAL scores (64.0 and 40.0) -- both were
   already caught before the per-clause code existed, because each example's
   matching clause happened to be internally undisturbed by the reordering
   and so was already a contiguous run findable by the whole-string search
   alone. The per-clause machinery added computation and a false claim of
   coverage, not capability, and was removed.
2. An order-invariant character/word n-gram bag-overlap signal was tried
   instead and also rejected: it scores an ALREADY-ACCEPTED calibration
   query ("grapple traversal mechanic", a shared technical term) as high as
   or higher than a genuine word-reordered paraphrase at every n-gram size
   tried (4/5/6/8/10 chars, and word 1/2/3-grams), so no single threshold
   separates "reworded reuse" from "a shared domain term in an
   independently-phrased on-topic question" -- see task-26-report.md for
   the full experiment and numbers.

Word- and clause-level paraphrase detection needs semantic (embedding)
similarity, not a surface-text statistic, and stays a human-review
responsibility when authoring new calibration/golden entries -- see
test_checker_still_misses_a_word_reordered_paraphrase_* below, which pins
this down as a known limitation rather than leaving it undocumented.

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
from portfolio_rag.loader import _CJK_RE, _CJK_WEIGHT, _effective_length, load_knowledge

KNOWLEDGE_DIR = settings.chat_root / "knowledge"


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _char_weight(ch: str) -> float:
    """loader._effective_length's CJK weighting, applied per character."""
    return _CJK_WEIGHT if _CJK_RE.match(ch) else 1.0


def _lcs_effective(a: str, b: str) -> tuple[float, str]:
    """Longest common (contiguous) substring of a and b, by EFFECTIVE length.

    Standard O(len(a) * len(b)) time, O(len(b)) space DP -- only the running
    row is kept since only the best cell's value (and its text) is needed,
    not the full table. Corpus bodies here are short (well under 300 chars,
    see the survey in task-26-report.md), so this is fast; there is no need
    for a suffix-automaton or other sub-quadratic approach at this scale.

    Maximizes EFFECTIVE length (loader._effective_length's CJK weighting)
    directly in the DP, rather than finding the RAW-longest substring first
    and weighting it afterward. That two-step version is not just a cosmetic
    tie-break difference, it is an outright wrong answer whenever a SHORTER,
    CJK-dense candidate (higher effective length) competes with a LONGER,
    Latin-only candidate (lower effective length) for the same match: raw-
    length-first always picks the longer one, even when the shorter one is
    the more informationally significant match. Concretely: two disjoint
    candidate substrings "abcde" (raw 5, effective 5.0) and "一二三" (raw 3,
    effective 7.5) -- a raw-length-first search reports "abcde" (wrong, lower
    effective length); this function reports "一二三" (right). Confirmed by a
    review of this task, which traced 234/3180 golden-ZH pairs' small score
    deltas (max +3.0 effective) under an earlier per-clause variant to
    exactly this bug rather than to the variant's own (nonexistent, see the
    module docstring) reordering capability.
    """
    if not a or not b:
        return 0.0, ""
    prev_eff = [0.0] * (len(b) + 1)
    prev_len = [0] * (len(b) + 1)
    best_eff = 0.0
    best_end = 0  # end index in `a` of the best match found so far
    best_len = 0  # its RAW length, needed to slice the substring back out
    for i, ca in enumerate(a, start=1):
        curr_eff = [0.0] * (len(b) + 1)
        curr_len = [0] * (len(b) + 1)
        weight = _char_weight(ca)
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                curr_eff[j] = prev_eff[j - 1] + weight
                curr_len[j] = prev_len[j - 1] + 1
                if curr_eff[j] > best_eff:
                    best_eff, best_end, best_len = curr_eff[j], i, curr_len[j]
        prev_eff, prev_len = curr_eff, curr_len
    return best_eff, a[best_end - best_len : best_end]


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

# Both thresholds were re-verified after _lcs_effective replaced the earlier
# raw-length-then-weight two-step (see its docstring for the bug that fix
# closes): re-surveying all four categories against the corrected,
# effective-length-maximizing DP reproduced the exact same four maxima
# (calibration EN 18.0, calibration ZH 20.0, golden EN 25.0, golden ZH 17.5)
# -- the bug did not happen to affect any of these four TOP entries, only
# some lower-ranked ones further down each survey. Both thresholds still
# hold with the same headroom reported above.


def _overlap(query: str, body: str) -> tuple[float, float, str]:
    """(effective LCS length, ratio of query's effective length, matched text)."""
    nq = _normalize(query)
    nb = _normalize(body)
    eff_query = _effective_length(nq) or 1.0
    eff_shared, shared = _lcs_effective(nq, nb)
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


def test_checker_still_misses_a_word_reordered_paraphrase_a_known_limitation_english() -> None:
    """Documents, rather than silently leaves undiscovered, the comparator's
    real limit: an active/passive (word-order) swap scrambles the
    contiguous run LCS depends on, so it is not caught (measured here: 28.0
    effective length / 0.34 ratio, both under threshold). Two fixes were
    tried and rejected for this -- see the module docstring: (1) a per-
    clause LCS variant, proven mathematically inert (a clause is always a
    substring of the whole query, so it can never find a longer match than
    the whole-string search already would) and removed; (2) an order-
    invariant character/n-gram-bag signal, which scores the ALREADY-ACCEPTED
    calibration query "grapple traversal mechanic" (a shared technical term,
    not reuse) as high as or higher than this very paraphrase, so no
    threshold on that signal separates the two cases. If a future change
    closes this gap, update this assertion (and the module docstring's
    "known gap" section) to describe the new capability instead of quietly
    deleting it."""
    body = (
        "He designed and implemented grapple traversal as a core movement "
        "mechanic in Cemented Dreams."
    )
    word_reordered_paraphrase = (
        "Was grapple traversal designed and implemented by him as a core "
        "movement mechanic?"
    )
    collisions = _find_collisions([word_reordered_paraphrase], [("planted-en-wordswap", body)])
    assert not collisions, (
        "the comparator now catches word-level reordering -- if intentional, "
        f"update this test and the module docstring's known-gap note: {collisions}"
    )


def test_checker_still_misses_a_word_reordered_paraphrase_a_known_limitation_chinese() -> None:
    """Chinese counterpart: a subject/predicate inversion turning a
    statement into a question, same words, different order (measured: 22.5
    effective length / 0.30 ratio, both under threshold). See the English
    version above for the full reasoning."""
    body = "他是一名游戏开发者，正在南加州大学攻读计算机科学硕士（游戏开发方向）。"
    word_reordered_paraphrase = "游戏开发方向的计算机科学硕士，是他正在南加州大学攻读的学位吗？"
    collisions = _find_collisions([word_reordered_paraphrase], [("planted-zh-wordswap", body)])
    assert not collisions, (
        "the comparator now catches word-level reordering -- if intentional, "
        f"update this test and the module docstring's known-gap note: {collisions}"
    )


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
