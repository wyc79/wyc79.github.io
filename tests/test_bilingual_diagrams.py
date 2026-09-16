import re
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile(r"[\u3400-\u9fff]")


class ImageCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.images.append(dict(attrs))


class InlineSvgCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.svgs = []
        self.current_svg = None
        self.in_text = False
        self.text_lang = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "svg":
            self.current_svg = {
                "attrs": attributes,
                "texts": {"en": [], "zh": []},
                "unlocalized": [],
            }
            self.svgs.append(self.current_svg)
        elif tag == "text" and self.current_svg is not None:
            self.in_text = True
            self.text_lang = attributes.get("data-lang")

    def handle_data(self, data):
        text = data.strip()
        if not self.in_text or not text:
            return
        if self.text_lang in {"en", "zh"}:
            self.current_svg["texts"][self.text_lang].append(text)
        else:
            self.current_svg["unlocalized"].append(text)

    def handle_endtag(self, tag):
        if tag == "text":
            self.in_text = False
            self.text_lang = None
        elif tag == "svg":
            self.current_svg = None


class BilingualDiagramTests(unittest.TestCase):
    diagrams = (
        (
            "pages/chat-agent.html",
            "../images/chat-agent/flow.svg",
            "../images/chat-agent/flow-zh.svg",
        ),
        (
            "pages/ai-tarot-projection.html",
            "../images/ai-tarot/key-path.svg",
            "../images/ai-tarot/key-path-zh.svg",
        ),
    )

    def test_pages_expose_one_localized_diagram_per_language(self):
        for page, en_src, zh_src in self.diagrams:
            with self.subTest(page=page):
                parser = ImageCollector()
                parser.feed((ROOT / page).read_text(encoding="utf-8"))
                localized = {
                    image.get("data-lang"): image
                    for image in parser.images
                    if image.get("src") in {en_src, zh_src}
                }

                self.assertEqual(localized["en"]["src"], en_src)
                self.assertEqual(localized["zh"]["src"], zh_src)
                self.assertFalse(CJK.search(localized["en"]["alt"]))
                self.assertTrue(CJK.search(localized["zh"]["alt"]))

    def test_localized_svgs_are_valid_and_share_a_canvas(self):
        for _, en_src, zh_src in self.diagrams:
            en_path = ROOT / en_src.removeprefix("../")
            zh_path = ROOT / zh_src.removeprefix("../")

            with self.subTest(svg=zh_path.name):
                en_root = ET.parse(en_path).getroot()
                zh_root = ET.parse(zh_path).getroot()

                self.assertEqual(en_root.attrib["viewBox"], zh_root.attrib["viewBox"])
                self.assertEqual(
                    en_root.attrib["{http://www.w3.org/XML/1998/namespace}lang"],
                    "en",
                )
                self.assertEqual(
                    zh_root.attrib["{http://www.w3.org/XML/1998/namespace}lang"],
                    "zh-Hans",
                )
                self.assertTrue(CJK.search("".join(zh_root.itertext())))

    def test_prime_engine_inline_diagrams_follow_the_language_toggle(self):
        parser = InlineSvgCollector()
        parser.feed((ROOT / "pages/prime-engine.html").read_text(encoding="utf-8"))

        self.assertEqual(len(parser.svgs), 4)
        for index, svg in enumerate(parser.svgs, start=1):
            with self.subTest(diagram=index):
                self.assertFalse(CJK.search(svg["attrs"]["data-al-en"]))
                self.assertTrue(CJK.search(svg["attrs"]["data-al-zh"]))
                self.assertEqual(svg["unlocalized"], [])
                self.assertTrue(svg["texts"]["en"])
                self.assertTrue(svg["texts"]["zh"])
                self.assertFalse(CJK.search("".join(svg["texts"]["en"])))
                self.assertTrue(CJK.search("".join(svg["texts"]["zh"])))


if __name__ == "__main__":
    unittest.main()
