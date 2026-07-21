"""HTML → sections. The site's own markup is the metadata schema:

- each page's <title> gives the page title ("Skills — Yuanchen Wang" → "Skills")
- each <section id=...> gives one retrieval unit with a stable #anchor
- nav / header / footer / scripts are boilerplate, never indexed

Bilingual pages: text is authored as data-lang="en"/"zh" element pairs (plus
data-en/data-zh attribute pairs), the same scheme scripts/i18n.js toggles at
runtime. Passing lang="en" or lang="zh" reduces a page to that one language
BEFORE extraction, so the index is monolingual instead of en+zh interleaved.
lang=None (default) keeps everything — the original, backward-compatible path.
"""

from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

_BOILERPLATE_TAGS = ("script", "style", "nav", "header", "footer", "noscript", "canvas")
_LANGS = ("en", "zh")


@dataclass(frozen=True)
class Section:
    url: str  # site-relative, e.g. "pages/skills.html"
    anchor: str  # section id, may be ""
    page_title: str
    section_title: str
    text: str


def _select_language(soup: BeautifulSoup, lang: str) -> None:
    """Collapse a bilingual soup to one language, mirroring i18n.js: drop the
    other language's data-lang nodes (target + unmarked/neutral text stays),
    and fold data-<lang> attribute pairs (title/buttons) into element text."""
    if lang not in _LANGS:
        raise ValueError(f"unknown lang {lang!r}; options: {_LANGS}")
    other = "en" if lang == "zh" else "zh"
    for el in soup.find_all(attrs={"data-lang": other}):
        el.decompose()
    # leaf elements carrying data-en/data-zh (e.g. <title>, buttons) render the
    # active language's attribute at runtime; bake it in for the target lang.
    for el in soup.find_all(attrs={f"data-{lang}": True}):
        if not el.find(True):
            el.string = el.get(f"data-{lang}") or el.get_text()


def _clean_text(node) -> str:
    return " ".join(node.get_text(separator=" ").split())


def _page_title(soup: BeautifulSoup) -> str:
    raw = soup.title.get_text() if soup.title else ""
    return raw.split("—")[0].split("-")[0].strip() or "Untitled"


def load_page(path: Path, url: str, lang: str | None = None) -> list[Section]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    if lang:
        _select_language(soup, lang)
    page_title = _page_title(soup)

    sections: list[Section] = []

    # The authored <meta name="description"> is high-signal summary text
    # ("Portfolio of YC Wang — aspiring game developer with a background
    # in ...") — indexing it makes broad questions like "who is YC" land.
    # Anchor "top" is the body id every page defines, so links stay valid.
    # It is English-only, so skip it when building a non-English index.
    meta = soup.find("meta", attrs={"name": "description"})
    desc = (meta.get("content") or "").strip() if meta else ""
    if len(desc) >= 40 and lang in (None, "en"):
        sections.append(
            Section(url=url, anchor="top", page_title=page_title, section_title=page_title, text=desc)
        )

    for tag in soup.find_all(_BOILERPLATE_TAGS):
        tag.decompose()

    scope = soup.find("main") or soup.body or soup
    content_sections: list[Section] = []
    for sec in scope.find_all("section"):
        heading = sec.find(["h1", "h2", "h3"])
        text = _clean_text(sec)
        if len(text) < 40:  # skip decorative/empty sections
            continue
        content_sections.append(
            Section(
                url=url,
                anchor=sec.get("id", ""),
                page_title=page_title,
                section_title=_clean_text(heading) if heading else page_title,
                text=text,
            )
        )

    if not content_sections:  # pages without <section> (e.g. the landing page)
        text = _clean_text(scope)
        if len(text) >= 40:
            content_sections.append(
                Section(url=url, anchor="", page_title=page_title, section_title=page_title, text=text)
            )
    return sections + content_sections


def load_knowledge(knowledge_dir: Path, lang: str | None = None) -> list[Section]:
    """Curated supplementary chunks from chat/knowledge/*.md.

    Visitors phrase questions in vocabulary the site itself never uses
    ("resume", "CV", "highlights") — these authored chunks bridge that gap.
    Format: each `## Heading` starts a section; an optional `link: <url>` line
    right after the heading sets where its source card points.

    One folder holds both languages as `about_en.md` / `about_zh.md`; lang
    selects `*_{lang}.md` so each consumer gets a single-language corpus (the
    en gate must not ingest zh chunks, and the bge-zh gate must not ingest en —
    mixing either re-introduces off-topic hubness). lang=None reads every
    `*.md` (unfiltered) for callers that manage language themselves.
    """
    pattern = f"*_{lang}.md" if lang else "*.md"
    sections: list[Section] = []
    for md in sorted(knowledge_dir.glob(pattern)):
        for block in md.read_text(encoding="utf-8").split("\n## ")[1:]:
            lines = block.strip().splitlines()
            heading, body = lines[0].strip(), lines[1:]
            url = "index.html"
            if body and body[0].startswith("link:"):
                url = body[0].split(":", 1)[1].strip()
                body = body[1:]
            text = " ".join(" ".join(body).split())
            if len(text) >= 40:
                sections.append(
                    Section(url=url, anchor="", page_title=heading, section_title=heading, text=text)
                )
    return sections


def load_site(site_root: Path, lang: str | None = None) -> list[Section]:
    """Load the whole site as sections. lang=None → bilingual pages (original);
    "en"/"zh" → a monolingual view. Curated chunks come from chat/knowledge/,
    picking about_<lang>.md — None uses the English corpus, matching the
    original single-index behaviour (bilingual pages + English about)."""
    sections: list[Section] = []
    index = site_root / "index.html"
    if index.exists():
        sections.extend(load_page(index, "index.html", lang))
    for page in sorted((site_root / "pages").glob("*.html")):
        sections.extend(load_page(page, f"pages/{page.name}", lang))
    knowledge_dir = site_root / "chat" / "knowledge"
    if knowledge_dir.is_dir():
        sections.extend(load_knowledge(knowledge_dir, lang or "en"))
    return sections
