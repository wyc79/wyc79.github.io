import json
from pathlib import Path

import numpy as np
import pytest

from portfolio_rag import index_builder
from portfolio_rag.config import settings

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
    return tmp_path


def test_builds_schema_with_deterministic_ids_and_vectors(tiny_site: Path) -> None:
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


def test_writes_roles_json_for_widget_and_worker(tiny_site: Path) -> None:
    index_builder.build_index(site_root=tiny_site)
    roles = json.loads((tiny_site / "out" / "roles.json").read_text(encoding="utf-8"))
    assert roles["default_role"] in roles["roles"]
    for role in roles["roles"].values():
        assert role["label"] and role["system_prompt"] and role["starters"]
