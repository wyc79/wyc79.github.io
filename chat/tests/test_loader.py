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


TEXTAREA_PAGE = """<!doctype html>
<html><head><title>Toolbox — Yuanchen Wang</title></head>
<body>
  <main>
    <section id="wc">
      <h2>Word Cloud</h2>
      <p>Generate a word cloud from any text you paste in below.</p>
      <textarea id="wc-input" placeholder="Paste or type text here...">DISTINCTIVE_TEXTAREA_DEMO_FILLER_MUST_NOT_BE_INDEXED</textarea>
    </section>
  </main>
</body></html>"""


def test_strips_textarea_demo_filler(tmp_path: Path) -> None:
    sections = load_page(_write(tmp_path, "toolbox.html", TEXTAREA_PAGE), "pages/toolbox.html")
    joined = " ".join(s.text for s in sections)
    assert "DISTINCTIVE_TEXTAREA_DEMO_FILLER_MUST_NOT_BE_INDEXED" not in joined


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


def test_load_knowledge_short_cjk_section_survives_floor(tmp_path: Path) -> None:
    """A CJK character carries far more information than a Latin one, so a
    raw-character floor tuned for English silently drops short-but-complete
    Chinese entries -- including exactly the identity-question shape
    ("who is YC") a gate most needs to see. The floor must weigh CJK
    characters heavier instead of comparing every script 1-for-1."""
    from portfolio_rag.loader import load_knowledge

    (tmp_path / "about_zh.md").write_text(
        "# comment header, not indexed\n\n"
        "## 王元辰是谁\n"
        "link: index.html\n"
        "王元辰是谁：一名游戏开发者，也是本作品集网站的作者。\n\n"
        "## 太短了\n"
        "link: x.html\n"
        "占位内容\n",
        encoding="utf-8",
    )
    sections = load_knowledge(tmp_path, "zh")
    assert len(sections) == 1  # the 4-character stub is still (correctly) dropped
    sec = sections[0]
    assert sec.section_title == "王元辰是谁"
    assert "游戏开发者" in sec.text


def test_load_site_walks_index_and_pages(tmp_path: Path) -> None:
    _write(tmp_path, "index.html", LANDING)
    (tmp_path / "pages").mkdir()
    _write(tmp_path / "pages", "skills.html", PAGE)
    sections = load_site(tmp_path)
    assert {s.url for s in sections} == {"index.html", "pages/skills.html"}


def test_every_site_page_produces_a_summary_chunk() -> None:
    """load_page indexes <meta name="description"> as the page's summary
    section (anchor "top"). It is what makes broad questions land, and the
    page-context feature places it first in the injected block -- so a page
    without one is a page the chat cannot introduce. Four pages silently had
    none: no tag at all, two stubs, and one that missed the 40-char floor by
    three characters."""
    from portfolio_rag.config import settings
    from portfolio_rag.loader import load_page

    site_root = settings.site_root
    pages = [(site_root / "index.html", "index.html")] + [
        (p, f"pages/{p.name}") for p in sorted((site_root / "pages").glob("*.html"))
    ]

    missing = []
    for path, url in pages:
        sections = load_page(path, url, "en")
        if not any(s.anchor == "top" for s in sections):
            missing.append(url)

    assert missing == [], (
        f"these pages have no <meta name=\"description\"> long enough to index: {missing}"
    )


def test_a_zh_description_becomes_the_zh_summary_chunk(tmp_path) -> None:
    """The Chinese index has never had per-page summary chunks: load_page
    gated the meta description on lang in (None, "en")."""
    from portfolio_rag.loader import load_page

    page = tmp_path / "demo.html"
    page.write_text(
        '<html><head><title>Toolbox — YC</title>'
        '<meta name="description" content="Small interactive tools built directly into this site."'
        ' data-zh="网站里内置的两个小工具：一个可以调整词数和缩放的词云生成器，以及一个可以调整输出尺寸的二维码生成器。">'
        "</head><body><main></main></body></html>",
        encoding="utf-8",
    )

    zh = [s for s in load_page(page, "pages/demo.html", "zh") if s.anchor == "top"]
    assert len(zh) == 1
    assert zh[0].text.startswith("网站里内置的两个小工具")

    en = [s for s in load_page(page, "pages/demo.html", "en") if s.anchor == "top"]
    assert len(en) == 1
    assert en[0].text == "Small interactive tools built directly into this site."


def test_a_page_with_no_zh_description_yields_no_zh_summary(tmp_path) -> None:
    """Falling back to content= would put English text in the Chinese half of
    the index, where it muddies retrieval and the zh gate cannot judge it."""
    from portfolio_rag.loader import load_page

    page = tmp_path / "demo.html"
    page.write_text(
        '<html><head><title>Demo — YC</title>'
        '<meta name="description" content="An English description that is comfortably past the floor.">'
        "</head><body><main></main></body></html>",
        encoding="utf-8",
    )

    assert [s for s in load_page(page, "pages/demo.html", "zh") if s.anchor == "top"] == []


def test_a_short_chinese_description_clears_the_script_weighted_floor(tmp_path) -> None:
    """40 RAW characters is a lot of Chinese. _effective_length already exists
    for exactly this and was only wired into load_knowledge."""
    from portfolio_rag.loader import load_page

    zh_desc = "王元辰在南加州大学攻读计算机科学游戏开发方向硕士学位。"  # 25 ideographs, 26 chars
    page = tmp_path / "demo.html"
    page.write_text(
        '<html><head><title>Education — YC</title>'
        '<meta name="description" content="Academic background of Yuanchen Wang at USC and Harvard."'
        f' data-zh="{zh_desc}">'
        "</head><body><main></main></body></html>",
        encoding="utf-8",
    )

    zh = [s for s in load_page(page, "pages/demo.html", "zh") if s.anchor == "top"]
    assert len(zh) == 1, "a 26-character Chinese description must not be dropped by a 40-char floor"


def test_no_meta_description_carries_data_en() -> None:
    """scripts/i18n.js selects [data-en][data-zh] and assigns textContent,
    which is meaningless on a void element. English lives in content=."""
    import re

    from portfolio_rag.config import settings

    site_root = settings.site_root
    offenders = []
    for path in [site_root / "index.html"] + sorted((site_root / "pages").glob("*.html")):
        for tag in re.findall(r"<meta[^>]*name=\"description\"[^>]*>", path.read_text(encoding="utf-8")):
            if "data-en=" in tag:
                offenders.append(path.name)

    assert offenders == [], f"data-en on a <meta> tag will be textContent-swapped by i18n.js: {offenders}"


def test_every_site_page_produces_a_summary_chunk_in_both_languages() -> None:
    """The English half is covered by
    test_every_site_page_produces_a_summary_chunk; this is the zh half, which
    needs data-zh on each page's meta description. Named per page so the
    failure says which ones are still missing."""
    from portfolio_rag.config import settings
    from portfolio_rag.loader import load_page

    site_root = settings.site_root
    pages = [(site_root / "index.html", "index.html")] + [
        (p, f"pages/{p.name}") for p in sorted((site_root / "pages").glob("*.html"))
    ]

    missing = {"en": [], "zh": []}
    for path, url in pages:
        for lang in ("en", "zh"):
            if not any(s.anchor == "top" for s in load_page(path, url, lang)):
                missing[lang].append(url)

    assert missing == {"en": [], "zh": []}, f"pages with no summary chunk: {missing}"


def test_an_authored_page_title_beats_the_title_tag_heuristic(tmp_path) -> None:
    """<meta name="page-title"> is authoritative. The <title> below it is the
    shape that breaks the heuristic in both directions: a hyphen inside a word
    for English, and no hyphen at all for Chinese."""
    from portfolio_rag.loader import load_page

    page = tmp_path / "demo.html"
    page.write_text(
        '<html><head>'
        '<title data-en="Portfolio Chat Agent: Role-Aware RAG | YC"'
        ' data-zh="作品集聊天助手：角色感知 RAG | 王元辰">Portfolio Chat Agent: Role-Aware RAG | YC</title>'
        '<meta name="description" content="A description comfortably past the length floor."'
        ' data-zh="一句足够长的中文描述，用来越过长度下限，确保这一段真的会被索引进去。">'
        '<meta name="page-title" content="Portfolio Chat Agent" data-zh="作品集聊天助手">'
        "</head><body><main></main></body></html>",
        encoding="utf-8",
    )

    en = [s for s in load_page(page, "pages/demo.html", "en") if s.anchor == "top"][0]
    zh = [s for s in load_page(page, "pages/demo.html", "zh") if s.anchor == "top"][0]

    assert en.page_title == "Portfolio Chat Agent", "the heuristic would cut this at 'Role'"
    assert zh.page_title == "作品集聊天助手", "the heuristic would keep ' | 王元辰'"


def test_a_page_with_no_authored_title_still_falls_back(tmp_path) -> None:
    """The tag is optional and can be added per page. Without it, behaviour is
    exactly what it was before the tag existed."""
    from portfolio_rag.loader import load_page

    page = tmp_path / "demo.html"
    page.write_text(
        '<html><head><title>Gyrotris - Solo Puzzle Game - YC</title>'
        '<meta name="description" content="A description comfortably past the length floor.">'
        "</head><body><main></main></body></html>",
        encoding="utf-8",
    )

    en = [s for s in load_page(page, "pages/demo.html", "en") if s.anchor == "top"][0]
    assert en.page_title == "Gyrotris"


def test_no_page_title_leaks_the_author_name_or_a_separator() -> None:
    """page_title is user-visible in three places -- source cards, /chat's
    page-awareness line, and any recommendation list -- so a title carrying
    "Yuanchen Wang" or a leftover "|" is a defect a visitor sees. index.html is
    exempt: its page genuinely is the author."""
    from portfolio_rag.config import settings
    from portfolio_rag.loader import load_page

    site_root = settings.site_root
    pages = [(p, f"pages/{p.name}") for p in sorted((site_root / "pages").glob("*.html"))]

    bad = []
    for path, url in pages:
        for lang in ("en", "zh"):
            for s in load_page(path, url, lang):
                if s.anchor != "top":
                    continue
                if "Yuanchen Wang" in s.page_title or "王元辰" in s.page_title or "|" in s.page_title:
                    bad.append(f"{url} [{lang}] {s.page_title!r}")

    assert bad == [], f"page titles carrying an author name or separator: {bad}"


def test_no_page_title_meta_carries_data_en() -> None:
    """Same void-element rule as the description tag: i18n.js selects
    [data-en][data-zh] and assigns textContent, which does nothing useful on a
    <meta>. English lives in content=."""
    import re

    from portfolio_rag.config import settings

    site_root = settings.site_root
    offenders = []
    for path in [site_root / "index.html"] + sorted((site_root / "pages").glob("*.html")):
        for tag in re.findall(r"<meta[^>]*name=\"page-title\"[^>]*>", path.read_text(encoding="utf-8")):
            if "data-en=" in tag:
                offenders.append(path.name)

    assert offenders == [], f"data-en on a <meta> tag is swapped by i18n.js: {offenders}"
