"""Golden-set tests.

Three tests, not 160: well-formedness and disjointness are fast and need no
models; the regression gate (added in Task 8) runs the whole set once.
"""

from collections import Counter

import pytest

from portfolio_rag.config import settings
from portfolio_rag.evaluation import CASE_TYPES, GOLDEN_PATH, load_cases
from portfolio_rag.roles import ROLES


@pytest.fixture(scope="module")
def cases():
    return load_cases(GOLDEN_PATH)


def test_ids_are_unique(cases) -> None:
    dupes = [cid for cid, n in Counter(c.id for c in cases).items() if n > 1]
    assert not dupes, f"duplicate case ids: {dupes}"


def test_roles_and_types_are_valid(cases) -> None:
    assert {c.role for c in cases} <= set(ROLES), "case names a role absent from roles.json"
    assert {c.type for c in cases} <= set(CASE_TYPES)
    assert {c.lang for c in cases} <= {"en", "zh"}


def test_positives_have_expectations_and_others_do_not(cases) -> None:
    for c in cases:
        if c.type == "positive":
            assert c.expected_urls, f"{c.id}: positive with no expected_urls"
            assert c.expected_keywords, f"{c.id}: positive with no expected_keywords"
            assert len(c.expected_keywords) <= 4, f"{c.id}: more than 4 keywords"
        else:
            assert not c.expected_urls, f"{c.id}: {c.type} must not carry expected_urls"
            assert not c.expected_keywords, f"{c.id}: {c.type} must not carry keywords"


def test_keywords_are_distinctive_enough_to_match_safely(cases) -> None:
    """A one- or two-character Latin keyword matches too much to mean anything.
    CJK is exempt: two Chinese characters are a real word."""
    weak = [
        f"{c.id}: {kw!r}"
        for c in cases
        for kw in c.expected_keywords
        if kw.isascii() and len(kw.strip()) < 3
    ]
    assert not weak, f"keywords too short to be distinctive: {weak}"


def test_expected_urls_point_at_real_pages(cases) -> None:
    missing = [
        f"{c.id} -> {url}"
        for c in cases
        for url in c.expected_urls
        if not (settings.site_root / url).exists()
    ]
    assert not missing, f"expected_urls naming files that do not exist: {missing}"
