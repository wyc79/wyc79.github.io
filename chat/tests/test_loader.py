from pathlib import Path

from portfolio_rag.loader import load_page, load_site

PAGE = """<!doctype html>
<html><head><title>Skills — Yuanchen Wang</title>
<meta name="description" content="Technical and research skills of Yuanchen Wang, game developer." />
</head>
<body id="top">
  <nav class="site-nav"><a href="x.html">Nav link that must not be indexed</a></nav>
  <main>
    <section id="skills">
      <h2>Skills</h2>
      <p>UE5, Unity, C++ and C# for gameplay and engine programming work.</p>
    </section>
    <section id="empty"><h2>.</h2></section>
    <section>
      <h3>Research</h3>
      <p>Eye tracking and motion capture research background, HCI and data viz.</p>
    </section>
  </main>
  <footer><p>Footer boilerplate that must not be indexed either.</p></footer>
</body></html>"""

LANDING = """<!doctype html>
<html><head><title>Yuanchen Wang — Portfolio</title></head>
<body>
  <section class="p3-root">
    <canvas id="p3-sphere"></canvas>
    <div class="p3-brand">YUANCHEN WANG — GAME DEVELOPER, USC MSCS portfolio landing.</div>
  </section>
</body></html>"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_extracts_sections_with_anchor_title_and_text(tmp_path: Path) -> None:
    sections = load_page(_write(tmp_path, "skills.html", PAGE), "pages/skills.html")
    assert [s.anchor for s in sections] == ["top", "skills", ""]
    meta, first, second = sections
    assert meta.text == "Technical and research skills of Yuanchen Wang, game developer."
    assert first.page_title == "Skills"
    assert first.section_title == "Skills"
    assert "engine programming" in first.text
    assert second.section_title == "Research"


def test_skips_nav_footer_and_tiny_sections(tmp_path: Path) -> None:
    sections = load_page(_write(tmp_path, "skills.html", PAGE), "pages/skills.html")
    joined = " ".join(s.text for s in sections)
    assert "not be indexed" not in joined
    assert all(s.anchor != "empty" for s in sections)


def test_falls_back_to_whole_page_without_main(tmp_path: Path) -> None:
    sections = load_page(_write(tmp_path, "index.html", LANDING), "index.html")
    assert len(sections) == 1
    assert sections[0].page_title == "Yuanchen Wang"
    assert "GAME DEVELOPER" in sections[0].text


def test_load_knowledge_parses_headings_links_and_text(tmp_path: Path) -> None:
    from portfolio_rag.loader import load_knowledge

    (tmp_path / "about.md").write_text(
        "# comment header, not indexed\n\npreamble is ignored\n\n"
        "## Resume highlights\nlink: pages/projects.html\n"
        "Resume and CV highlights of YC Wang, game developer and engineer.\n\n"
        "## Tiny\nlink: x.html\nshort\n",
        encoding="utf-8",
    )
    sections = load_knowledge(tmp_path)
    assert len(sections) == 1  # tiny block dropped
    sec = sections[0]
    assert sec.section_title == "Resume highlights"
    assert sec.url == "pages/projects.html"
    assert "CV highlights" in sec.text and "link:" not in sec.text


def test_load_site_walks_index_and_pages(tmp_path: Path) -> None:
    _write(tmp_path, "index.html", LANDING)
    (tmp_path / "pages").mkdir()
    _write(tmp_path / "pages", "skills.html", PAGE)
    sections = load_site(tmp_path)
    assert {s.url for s in sections} == {"index.html", "pages/skills.html"}
