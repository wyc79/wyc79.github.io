"""Task 29 Part 2: the four-file split under chat/data/ is only as good as
its enforcement. "One file, one job" (see chat/README.md and
src/portfolio_rag/index_builder.py) is a naming *promise* -- these tests are
what makes breaking that promise a test failure instead of a silent, someday-
rediscovered bug the way the old gate_vectors.json/fallback_vectors.json
conflation was (Task 24 broke degraded-mode source links this way and it took
until Task 29 to notice).

Four things, each required by the brief:
1. Name matches contents, per file: the model preset declared inside matches
   the one in the filename, `lang` matches, a gate file has no chunk records
   and a chunks file has no gate fields, and the section/chunk counts are
   what the name implies (measured against the real corpus on disk, never a
   hardcoded number that would silently go stale as the corpus grows).
2. Alignment: chunks_en_minilm.json's chunk ids equal, in order, the ids of
   chunks_e5.json's lang=="en" chunks -- the exact property whose absence
   let Task 24's bug happen.
3. gate_zh_bge.json is never served: matched by chat/.gitignore (real
   `git check-ignore` semantics, not a string check against the .gitignore
   file's own text) and never referenced by scripts/chat-widget.js.
4. The build code documents, at the point where the zh gate is written, why
   there is no chunks_zh_minilm.json.

Reads the real committed data files directly with json.load (never with a
file-reading tool -- see the task brief's constraints; these files are
multi-MB single-line JSON).
"""

import json
import subprocess

import pytest

from portfolio_rag.config import settings
from portfolio_rag.loader import load_knowledge

CHUNKS_E5_PATH = settings.resolve_chunks_path()
GATE_EN_PATH = settings.resolve_path(settings.gate_en_minilm_path)
GATE_ZH_PATH = settings.resolve_path(settings.gate_zh_bge_path)
CHUNKS_EN_MINILM_PATH = settings.resolve_path(settings.chunks_en_minilm_path)
KNOWLEDGE_DIR = settings.chat_root / "knowledge"
WIDGET_PATH = settings.site_root / "scripts" / "chat-widget.js"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


# --- 1. Name matches contents -----------------------------------------------


def test_chunks_e5_json_name_matches_its_declared_preset_and_carries_no_gate_fields() -> None:
    assert CHUNKS_E5_PATH.name == "chunks_e5.json", (
        "settings.resolve_chunks_path() must derive this exact name from "
        "settings.model_preset ('e5', per chat/.env) -- if this fails, "
        "either .env drifted off e5 or the derivation broke"
    )
    data = _load(CHUNKS_E5_PATH)
    assert data["model_preset"] == "e5"
    for gate_field in ("gate_stat", "gate_threshold", "gate_margin", "gate_remote"):
        assert gate_field not in data, (
            f"chunks_e5.json carries {gate_field!r} -- a chunks file is not a "
            "gate file (Task 29 Part 2 spec 2.2); gate fields belong in "
            "meta.json and the gate_*.json files only"
        )
    chunks = data["chunks"]
    en_count = sum(1 for c in chunks if c.get("lang") == "en")
    zh_count = sum(1 for c in chunks if c.get("lang") == "zh")
    assert en_count + zh_count == len(chunks), (
        "every chunk in a multilingual (e5) build must carry a lang tag"
    )
    assert en_count > 0 and zh_count > 0, (
        "chunks_e5.json must hold both languages (pages + about, per the "
        "brief -- a language split is explicitly out of scope)"
    )


def test_gate_en_minilm_json_name_matches_contents_and_carries_no_chunk_records() -> None:
    data = _load(GATE_EN_PATH)
    assert data["model_preset"] == "minilm"
    assert data["lang"] == "en"
    assert data["corpus"] == "knowledge/about_en.md"
    for retrieval_field in ("chunks", "chunk_size", "chunk_overlap", "schema_version"):
        assert retrieval_field not in data, (
            f"gate_en_minilm.json carries {retrieval_field!r} -- a gate file "
            "is not a retrieval corpus (Task 29 Part 2 spec 2.2), which is "
            "the exact conflation that broke degraded-mode source links"
        )
    assert not any(isinstance(v, dict) and "id" in v for v in data.get("vectors", [])), (
        "gate vectors must be plain vectors, never chunk records with ids"
    )
    # Section count matches the CURRENT knowledge/about_en.md corpus on disk
    # -- measured, never a hardcoded 55 that would silently go stale as the
    # corpus grows (see the brief's own "read it from the file" convention).
    expected = len(load_knowledge(KNOWLEDGE_DIR, "en"))
    assert len(data["vectors"]) == expected, (
        f"gate_en_minilm.json has {len(data['vectors'])} vectors but "
        f"knowledge/about_en.md currently has {expected} sections -- rebuild "
        "with scripts/build_index.py"
    )


@pytest.mark.skipif(not GATE_ZH_PATH.exists(), reason="gate_zh_bge.json is gitignored and absent")
def test_gate_zh_bge_json_name_matches_contents_and_carries_no_chunk_records() -> None:
    data = _load(GATE_ZH_PATH)
    assert data["model_preset"] == "bge_zh"
    assert data["lang"] == "zh"
    assert data["corpus"] == "knowledge/about_zh.md"
    for retrieval_field in ("chunks", "chunk_size", "chunk_overlap", "schema_version"):
        assert retrieval_field not in data
    expected = len(load_knowledge(KNOWLEDGE_DIR, "zh"))
    assert len(data["vectors"]) == expected, (
        f"gate_zh_bge.json has {len(data['vectors'])} vectors but "
        f"knowledge/about_zh.md currently has {expected} sections -- rebuild "
        "with scripts/build_index.py"
    )


def test_chunks_en_minilm_json_name_matches_contents_and_carries_no_gate_fields() -> None:
    data = _load(CHUNKS_EN_MINILM_PATH)
    assert data["model_preset"] == "minilm"
    for gate_field in ("gate_stat", "gate_threshold", "gate_margin", "gate_remote"):
        assert gate_field not in data
    chunks = data["chunks"]
    assert chunks, "chunks_en_minilm.json must not be empty"
    assert all(c.get("lang") == "en" for c in chunks), (
        "every chunk in chunks_en_minilm.json must be English -- this file "
        "has no Chinese counterpart by design (MiniLM cannot embed Chinese)"
    )
    # NOT "all pages": the brief requires 137 page + 55 curated about_en
    # chunks specifically (populating it with pages alone would take
    # degraded-mode hit@4 from 90/96 to 59/96, per the brief's own
    # measurement). The curated-section count is checked against the live
    # corpus, like the gate tests above; the page-chunk count is whatever's
    # left, not independently re-derived here (that would just re-implement
    # the loader/chunker, which the alignment test below already exercises
    # for real via chunks_e5.json).
    about_en_sections = len(load_knowledge(KNOWLEDGE_DIR, "en"))
    curated_chunk_ids = {
        c["id"] for c in chunks
        if c.get("page_title") == c.get("section_title")
        and c["page_title"] in {s.section_title for s in load_knowledge(KNOWLEDGE_DIR, "en")}
    }
    assert len(curated_chunk_ids) == about_en_sections, (
        f"expected {about_en_sections} curated about_en chunks (one chunk "
        f"per section -- these are short and don't get split by chunk_text), "
        f"found {len(curated_chunk_ids)}. If about_en.md's authored sections "
        "have grown long enough to be split into multiple chunks, this "
        "assumption needs revisiting, not silently loosening."
    )
    assert len(chunks) > about_en_sections, (
        "chunks_en_minilm.json must hold page chunks in addition to the "
        "curated about_en ones -- 'all curated, no pages' would be as wrong "
        "as 'all pages, no curated'"
    )


# --- 2. Alignment: the exact property that prevents the Task 24 bug --------


def test_chunks_en_minilm_ids_equal_chunks_e5_english_ids_in_order() -> None:
    e5_data = _load(CHUNKS_E5_PATH)
    minilm_data = _load(CHUNKS_EN_MINILM_PATH)
    e5_en_ids = [c["id"] for c in e5_data["chunks"] if c.get("lang") == "en"]
    minilm_ids = [c["id"] for c in minilm_data["chunks"]]
    assert minilm_ids == e5_en_ids, (
        "chunks_en_minilm.json's chunk ids must equal, in order, "
        "chunks_e5.json's lang=='en' chunk ids -- a mismatch here is exactly "
        "the Task 24 bug class: degraded-mode retrieval resolving a chunk "
        "record that does not correspond to the id/position the widget "
        "thinks it does"
    )
    # Text must agree too, not just ids -- an id collision with different
    # underlying text would defeat the whole point of resolving "by id."
    e5_en_text_by_id = {c["id"]: c["text"] for c in e5_data["chunks"] if c.get("lang") == "en"}
    mismatched_text = [
        c["id"] for c in minilm_data["chunks"] if c["text"] != e5_en_text_by_id[c["id"]]
    ]
    assert not mismatched_text, f"chunk text diverged for ids: {mismatched_text}"


# --- 3. gate_zh_bge.json is never served ------------------------------------


def test_gate_zh_bge_json_is_gitignored_and_never_served_to_a_browser() -> None:
    """MUST NEVER be served to a browser: MiniLM, the only model this
    project ever ships to the browser, cannot embed Chinese at all, and the
    widget already handles CJK questions in degraded mode with `degradedCJK`
    plus a static page-link list (never local Chinese retrieval). Gitignoring
    this file is the enforcement that does not depend on
    scripts/chat-widget.js behaving correctly -- verified here with REAL
    `git check-ignore` semantics (not a string search of .gitignore's own
    text, which could pass on a pattern that doesn't actually match), plus a
    belt-and-braces check that the widget's own source never names the file
    (so a future edit that tried to fetch it would be caught immediately,
    before ever reaching a browser)."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(GATE_ZH_PATH)],
        cwd=settings.chat_root, capture_output=True,
    )
    assert result.returncode == 0, (
        f"git check-ignore did not confirm {GATE_ZH_PATH} is ignored "
        f"(exit {result.returncode}) -- chat/.gitignore's data/gate_zh_bge.json "
        "rule is missing or no longer matches"
    )
    widget_src = WIDGET_PATH.read_text(encoding="utf-8")
    assert "gate_zh_bge" not in widget_src, (
        "scripts/chat-widget.js must never reference gate_zh_bge.json -- "
        "this file must never be fetched by a browser"
    )
    assert "chunks_zh_minilm" not in widget_src, (
        "there is deliberately no chunks_zh_minilm.json (see "
        "index_builder.py's _build_zh_gate docstring) -- a reference here "
        "would mean someone built the tempting-but-wrong symmetric file"
    )


# --- 4. The build code documents why there is no chunks_zh_minilm.json -----


def test_build_code_documents_why_there_is_no_chunks_zh_minilm() -> None:
    """Required by the brief at the point where the zh gate is written, not
    merely somewhere in the file: MiniLM cannot embed Chinese, the widget
    already shows degradedCJK plus static page links, and the symmetric-
    looking file is the tempting mistake someone will eventually reach for."""
    src = (settings.chat_root / "src" / "portfolio_rag" / "index_builder.py").read_text(
        encoding="utf-8"
    )
    zh_gate_fn_start = src.index("def _build_zh_gate")
    zh_gate_fn_end = src.index("\ndef ", zh_gate_fn_start + 1)
    docstring = src[zh_gate_fn_start:zh_gate_fn_end]
    # Normalize whitespace (the docstring is line-wrapped at ~79 columns like
    # the rest of this codebase, so a multi-word phrase can legitimately
    # straddle a line break) before substring-matching.
    normalized = " ".join(docstring.split())
    required_phrases = [
        "chunks_zh_minilm",
        "cannot embed Chinese",
        "degradedCJK",
        "static page links",
        "tempting mistake",
    ]
    missing = [p for p in required_phrases if p not in normalized]
    assert not missing, (
        f"_build_zh_gate's docstring (where the zh gate is written) is "
        f"missing required explanatory phrase(s): {missing}"
    )
