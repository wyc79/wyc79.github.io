import json
from pathlib import Path

import numpy as np
import pytest

from portfolio_rag import index_builder
from portfolio_rag.config import MODEL_PRESETS, settings
from portfolio_rag.embedder import OnnxEmbedder

SITE_PAGE = """<!doctype html>
<html><head><title>Projects — Yuanchen Wang</title></head>
<body><main>
  <section id="prime-engine">
    <h2>Prime Engine</h2>
    <p>%s</p>
  </section>
  <section><h3>Anchorless one</h3><p>First section without an id attribute on this page.</p></section>
  <section><h3>Anchorless two</h3><p>Second section without an id attribute on the same page.</p></section>
</main></body></html>""" % ("Engine programming work in C++ covering rendering and tooling. " * 30)


@pytest.fixture()
def tiny_site(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "projects.html").write_text(SITE_PAGE, encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(settings, "index_path", str(out / "index.json"))
    monkeypatch.setattr(settings, "roles_path", str(out / "roles.json"))
    monkeypatch.setattr(settings, "gate_vectors_path", str(out / "gate_vectors.json"))
    monkeypatch.setattr(settings, "fallback_vectors_path", str(out / "fallback_vectors.json"))
    return tmp_path


def test_tiny_site_fixture_confines_all_build_outputs_to_tmp_path(tiny_site: Path) -> None:
    # build_index() can write to four settings-driven paths. If any of them
    # isn't patched onto tmp_path, a build under this fixture writes through
    # to the real, gitignored, hand-calibrated files under chat/data/ and
    # destroys them irrecoverably. Check this WITHOUT calling build_index —
    # a test that runs the unfixed build and diffs the real files before/after
    # would destroy the data on every failure, which is exactly the bug this
    # guards against.
    for field in ("index_path", "roles_path", "gate_vectors_path", "fallback_vectors_path"):
        resolved = settings.resolve_path(getattr(settings, field))
        assert resolved.is_relative_to(tiny_site), (
            f"settings.{field} resolves to {resolved}, outside the tiny_site "
            f"fixture's tmp_path ({tiny_site}); build_index() would write "
            f"through to the real committed/gitignored file"
        )


def test_builds_schema_with_deterministic_ids_and_vectors(tiny_site: Path, monkeypatch) -> None:
    # dim 384 and an empty query_prefix are minilm's values specifically —
    # pin the preset explicitly instead of inheriting whatever
    # RAG_MODEL_PRESET the environment happens to set (the required config
    # is now "e5"; see .env.example). This test is about the schema minilm
    # produces, not about which preset is configured.
    monkeypatch.setattr(settings, "model_preset", "minilm")
    stats = index_builder.build_index(site_root=tiny_site)
    index = json.loads((tiny_site / "out" / "index.json").read_text(encoding="utf-8"))

    assert index["schema_version"] == index_builder.SCHEMA_VERSION
    assert index["dim"] == 384
    assert index["model_preset"] == "minilm"
    assert index["query_prefix"] == ""
    # Thresholds are stat-dependent (a zscore gate can be ~3), and this tiny
    # one-topic fixture can't calibrate meaningfully — just check the fields.
    assert index["gate_stat"] in {"top", "contrast", "zscore"}
    assert isinstance(index["gate_threshold"], float)
    assert stats["chunks"] == len(index["chunks"]) > 1  # long section got split

    ids = [c["id"] for c in index["chunks"]]
    assert len(set(ids)) == len(ids), "chunk ids must be unique"

    first = index["chunks"][0]
    assert first["id"] == "pages/projects.html#prime-engine:0"
    assert first["page_title"] == "Projects"
    assert first["section_title"] == "Prime Engine"
    vec = np.array(first["vector"])
    assert vec.shape == (384,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-3

    # dim==384 alone doesn't prove build_index actually embedded with minilm
    # (e5 is also 384-dim) — it only proves the metadata build_index copies
    # from settings.model_preset, which is exactly what would keep lying if
    # the retrieval embedder came from somewhere else. Tie the assertion to
    # the vector's actual content: independently re-embed this chunk's own
    # text (read back from the index, not reconstructed) with a dedicated,
    # freshly-built minilm embedder and require the two to match. If
    # build_index ever again resolved its embedder from a process-wide
    # cache that another test/module could have already primed under a
    # different preset (this happened — see index_builder.py's comment on
    # why it builds a dedicated OnnxEmbedder instead of calling
    # get_embedder()), this would embed with the wrong model while the
    # metadata above kept claiming "minilm", and only this assertion would
    # catch it.
    reference = OnnxEmbedder.from_preset(
        MODEL_PRESETS["minilm"],
        settings.resolve_path(MODEL_PRESETS["minilm"]["dir"]),
        settings.embedding_max_tokens,
    )
    expected_vec = reference.embed_documents([first["text"]])[0]
    assert np.allclose(vec, expected_vec, atol=1e-4), (
        "index vector for the first chunk does not match a freshly-built "
        "minilm embedder's output for that chunk's own text — build_index "
        "did not actually embed with the pinned minilm preset"
    )


def test_build_refuses_to_change_preset_of_existing_index(tiny_site: Path, monkeypatch) -> None:
    # Rebuilding over an index built with a different model_preset would
    # silently desync data/gate_vectors.json, data/fallback_vectors.json and
    # the deployed Tencent backend (see .env.example) — the whole bug this
    # guard exists to prevent. No opt-in env var set, so this must raise.
    monkeypatch.delenv("RAG_ALLOW_PRESET_CHANGE", raising=False)
    monkeypatch.setattr(settings, "model_preset", "e5")
    (tiny_site / "out" / "index.json").write_text(
        json.dumps({"model_preset": "minilm"}), encoding="utf-8"
    )

    with pytest.raises(ValueError) as excinfo:
        index_builder.build_index(site_root=tiny_site)
    message = str(excinfo.value)
    assert "minilm" in message, "error must name the existing index's preset"
    assert "e5" in message, "error must name the configured preset"


def test_build_allows_preset_change_with_explicit_opt_in(tiny_site: Path, monkeypatch) -> None:
    # RAG_ALLOW_PRESET_CHANGE=1 is the deliberate-switch escape hatch: a
    # developer who really means to move the index to a new embedding space
    # can still do it, just not by accident.
    monkeypatch.setenv("RAG_ALLOW_PRESET_CHANGE", "1")
    monkeypatch.setattr(settings, "model_preset", "e5")
    (tiny_site / "out" / "index.json").write_text(
        json.dumps({"model_preset": "minilm"}), encoding="utf-8"
    )

    index_builder.build_index(site_root=tiny_site)  # must not raise

    index = json.loads((tiny_site / "out" / "index.json").read_text(encoding="utf-8"))
    assert index["model_preset"] == "e5"


def test_build_with_no_existing_index_does_not_require_opt_in(
    tiny_site: Path, monkeypatch
) -> None:
    # A fresh clone / first build has no index.json to compare against — the
    # guard must not block that case even without RAG_ALLOW_PRESET_CHANGE.
    monkeypatch.delenv("RAG_ALLOW_PRESET_CHANGE", raising=False)
    monkeypatch.setattr(settings, "model_preset", "e5")
    assert not (tiny_site / "out" / "index.json").exists()

    index_builder.build_index(site_root=tiny_site)  # must not raise


def test_writes_roles_json_for_widget_and_worker(tiny_site: Path) -> None:
    index_builder.build_index(site_root=tiny_site)
    roles = json.loads((tiny_site / "out" / "roles.json").read_text(encoding="utf-8"))
    assert roles["default_role"] in roles["roles"]
    for role in roles["roles"].values():
        assert role["label"] and role["system_prompt"] and role["starters"]
