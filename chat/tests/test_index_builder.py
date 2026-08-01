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
    # This one-topic, five-chunk fixture genuinely cannot separate the
    # canonical on-/off-topic query sets (measured margin ~ -15%), so it
    # trips the task-20 gate-margin floor below. That floor is a separate
    # concern from what this test checks (index schema/vectors), so opt out
    # of it explicitly rather than letting an unrelated raise mask what this
    # test is actually about.
    monkeypatch.setenv("RAG_ALLOW_NEGATIVE_MARGIN", "1")
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
    # Orthogonal to the preset guard under test: this tiny fixture's real
    # calibration margin is negative (measured ~ -12%), which would trip the
    # task-20 gate-margin floor before this test ever gets to its own
    # assertion. Opt out of that separate guard explicitly.
    monkeypatch.setenv("RAG_ALLOW_NEGATIVE_MARGIN", "1")
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
    # See the comment in test_build_allows_preset_change_with_explicit_opt_in
    # above — same orthogonal opt-out, same reason.
    monkeypatch.setenv("RAG_ALLOW_NEGATIVE_MARGIN", "1")
    monkeypatch.setattr(settings, "model_preset", "e5")
    assert not (tiny_site / "out" / "index.json").exists()

    index_builder.build_index(site_root=tiny_site)  # must not raise


# --- Task 20: fail the build on a non-separating en gate calibration -------
#
# compute_gate() already logs a WARNING and ships anyway when nothing
# separates on-/off-topic, and for an e5 index the en gate is written
# unconditionally — that asymmetry (zh already refuses to enable itself via
# _build_zh_gate; en had no check at all) is how a live false-refusal bug
# reached production silently. These tests force a deterministic margin via
# a fake compute_gate rather than relying on this tiny fixture's real
# calibration (which happens to be negative today, but that's incidental —
# the guard must fire on ANY margin below the configured floor, not on this
# particular fixture's numbers).

_FAKE_NEGATIVE_GATE = {"stat": "top", "threshold": 0.51, "margin": -0.05, "lo": 0.50, "hi": 0.45}
_FAKE_POSITIVE_GATE = {"stat": "top", "threshold": 0.51, "margin": 0.0537, "lo": 0.50, "hi": 0.53}


def test_build_raises_before_writing_any_artifact_on_a_negative_margin(
    tiny_site: Path, monkeypatch
) -> None:
    monkeypatch.delenv("RAG_ALLOW_NEGATIVE_MARGIN", raising=False)
    monkeypatch.delenv("RAG_MIN_GATE_MARGIN", raising=False)
    # e5 (not minilm): this is the branch that writes gate_vectors.json and
    # fallback_vectors.json BEFORE index.json and roles.json — the exact
    # half-updated-data/ scenario amendment 1 is about. A raise placed after
    # any of those writes would be strictly worse than the silent-ship bug
    # it replaces.
    monkeypatch.setattr(settings, "model_preset", "e5")
    monkeypatch.setattr(index_builder, "compute_gate", lambda *a, **k: _FAKE_NEGATIVE_GATE)

    with pytest.raises(ValueError) as excinfo:
        index_builder.build_index(site_root=tiny_site)
    message = str(excinfo.value)
    assert _FAKE_NEGATIVE_GATE["stat"] in message, "error must name the chosen stat"
    assert f"{_FAKE_NEGATIVE_GATE['lo']:.4f}" in message, "error must name the off-topic bound"
    assert f"{_FAKE_NEGATIVE_GATE['hi']:.4f}" in message, "error must name the on-topic bound"
    assert f"{_FAKE_NEGATIVE_GATE['margin']:.4f}" in message, "error must name the measured margin"

    out = tiny_site / "out"
    written = sorted(p.name for p in out.iterdir())
    assert written == [], (
        f"a refused build must not leave ANY artifact written; found {written} — "
        "a raise placed after even one write leaves data/ half-updated, which "
        "is worse than the silent-ship bug this guard replaces"
    )


def test_build_allows_negative_margin_with_explicit_opt_in(tiny_site: Path, monkeypatch) -> None:
    # RAG_ALLOW_NEGATIVE_MARGIN=1 is the deliberate-ship escape hatch —
    # mirrors RAG_ALLOW_PRESET_CHANGE/RAG_ZH_GATE_FORCE above.
    monkeypatch.setenv("RAG_ALLOW_NEGATIVE_MARGIN", "1")
    monkeypatch.delenv("RAG_MIN_GATE_MARGIN", raising=False)
    monkeypatch.setattr(settings, "model_preset", "e5")
    monkeypatch.setattr(index_builder, "compute_gate", lambda *a, **k: _FAKE_NEGATIVE_GATE)

    index_builder.build_index(site_root=tiny_site)  # must not raise

    out = tiny_site / "out"
    assert (out / "index.json").exists()
    assert (out / "gate_vectors.json").exists()


def test_gate_margin_floor_is_configurable_not_a_hardcoded_sign(
    tiny_site: Path, monkeypatch
) -> None:
    # Amendment 2: the trigger is `margin < RAG_MIN_GATE_MARGIN` (default
    # 0.0), not a hardcoded "negative means bad" check. Lowering the floor
    # below the fake -5% margin must let the same build through WITHOUT the
    # RAG_ALLOW_NEGATIVE_MARGIN opt-out — proving the floor, not the sign, is
    # what the guard actually reads.
    monkeypatch.delenv("RAG_ALLOW_NEGATIVE_MARGIN", raising=False)
    monkeypatch.setenv("RAG_MIN_GATE_MARGIN", "-1.0")
    monkeypatch.setattr(settings, "model_preset", "e5")
    monkeypatch.setattr(index_builder, "compute_gate", lambda *a, **k: _FAKE_NEGATIVE_GATE)

    index_builder.build_index(site_root=tiny_site)  # -0.05 >= -1.0 floor: must not raise


def test_gate_margin_floor_defaults_to_zero(tiny_site: Path, monkeypatch) -> None:
    # No RAG_MIN_GATE_MARGIN set at all -- the default floor (0.0) must still
    # reject a small negative margin. This is the "ship exactly the proposal
    # by default" half of amendment 2.
    monkeypatch.delenv("RAG_ALLOW_NEGATIVE_MARGIN", raising=False)
    monkeypatch.delenv("RAG_MIN_GATE_MARGIN", raising=False)
    monkeypatch.setattr(settings, "model_preset", "e5")
    monkeypatch.setattr(
        index_builder, "compute_gate",
        lambda *a, **k: {"stat": "top", "threshold": 0.51, "margin": -0.0001, "lo": 0.50, "hi": 0.4999},
    )

    with pytest.raises(ValueError):
        index_builder.build_index(site_root=tiny_site)


def test_build_allows_a_margin_exactly_at_the_floor(tiny_site: Path, monkeypatch) -> None:
    # margin == floor must NOT raise: the brief's trigger is `margin <
    # RAG_MIN_GATE_MARGIN`, a strict inequality.
    monkeypatch.delenv("RAG_ALLOW_NEGATIVE_MARGIN", raising=False)
    monkeypatch.delenv("RAG_MIN_GATE_MARGIN", raising=False)
    monkeypatch.setattr(settings, "model_preset", "e5")
    monkeypatch.setattr(
        index_builder, "compute_gate",
        lambda *a, **k: {"stat": "top", "threshold": 0.51, "margin": 0.0, "lo": 0.50, "hi": 0.50},
    )

    index_builder.build_index(site_root=tiny_site)  # must not raise


# --- Task 20 amendment 3: persist the measured margin -----------------------


def test_build_persists_gate_margin_in_artifacts_and_summary(
    tiny_site: Path, monkeypatch
) -> None:
    # A recorded margin shows a gate decaying across builds while it is
    # still positive -- the raise above only fires the instant it crosses
    # the floor, far too late to be the primary signal. Written into
    # gate_vectors.json per language, into index.json, and returned in the
    # summary, using e5 so BOTH gate_vectors.json (en+zh) and index.json are
    # exercised in one build.
    monkeypatch.setattr(settings, "model_preset", "e5")
    monkeypatch.setattr(index_builder, "compute_gate", lambda *a, **k: _FAKE_POSITIVE_GATE)

    summary = index_builder.build_index(site_root=tiny_site)
    assert summary["gate_margin"] == _FAKE_POSITIVE_GATE["margin"]

    index = json.loads((tiny_site / "out" / "index.json").read_text(encoding="utf-8"))
    assert index["gate_margin"] == _FAKE_POSITIVE_GATE["margin"]

    gates = json.loads((tiny_site / "out" / "gate_vectors.json").read_text(encoding="utf-8"))
    assert gates["en"]["gate_margin"] == _FAKE_POSITIVE_GATE["margin"]
    # zh gate: the fake compute_gate is reused for _build_zh_gate's call too
    # (real bge-zh model + real knowledge/about_zh.md corpus), so a positive
    # margin here means the zh gate is ALSO enabled and must carry its own
    # gate_margin -- proving "per language", not just "en only".
    assert gates["zh"]["gate_margin"] == _FAKE_POSITIVE_GATE["margin"]

    fallback = json.loads((tiny_site / "out" / "fallback_vectors.json").read_text(encoding="utf-8"))
    assert fallback["gate_margin"] == _FAKE_POSITIVE_GATE["margin"]


def test_build_persists_gate_margin_for_the_local_minilm_gate_too(
    tiny_site: Path, monkeypatch
) -> None:
    # minilm has no gate_model (it gates itself, no gate_vectors.json at
    # all) -- a different code path than the e5/delegate branch above.
    # index.json's top-level gate_margin must still be populated there.
    monkeypatch.setattr(settings, "model_preset", "minilm")
    monkeypatch.setattr(index_builder, "compute_gate", lambda *a, **k: _FAKE_POSITIVE_GATE)

    summary = index_builder.build_index(site_root=tiny_site)
    assert summary["gate_margin"] == _FAKE_POSITIVE_GATE["margin"]

    index = json.loads((tiny_site / "out" / "index.json").read_text(encoding="utf-8"))
    assert index["gate_margin"] == _FAKE_POSITIVE_GATE["margin"]
    assert not (tiny_site / "out" / "gate_vectors.json").exists()


def test_writes_roles_json_for_widget_and_worker(tiny_site: Path, monkeypatch) -> None:
    # Ambient settings.model_preset is "e5" (chat/.env). Same orthogonal
    # gate-margin opt-out as the preset-guard tests above — this test is
    # about roles.json, not gate calibration quality.
    monkeypatch.setenv("RAG_ALLOW_NEGATIVE_MARGIN", "1")
    index_builder.build_index(site_root=tiny_site)
    roles = json.loads((tiny_site / "out" / "roles.json").read_text(encoding="utf-8"))
    assert roles["default_role"] in roles["roles"]
    for role in roles["roles"].values():
        assert role["label"] and role["system_prompt"] and role["starters"]
