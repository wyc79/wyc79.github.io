import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRAPH_URL = "https://wyc79.github.io/ai-tarot-projection/graph.html"


class LinkCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.current_link = None
        self.current_lang = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "a":
            self.current_link = {"attrs": attributes, "texts": {"en": [], "zh": []}}
            self.links.append(self.current_link)
        elif tag == "span" and self.current_link is not None:
            self.current_lang = attributes.get("data-lang")

    def handle_data(self, data):
        if self.current_link is not None and self.current_lang in {"en", "zh"}:
            self.current_link["texts"][self.current_lang].append(data.strip())

    def handle_endtag(self, tag):
        if tag == "span":
            self.current_lang = None
        elif tag == "a":
            self.current_link = None


class TarotWorkflowDemoTests(unittest.TestCase):
    def test_project_page_exposes_demo_link_in_both_languages(self):
        parser = LinkCollector()
        parser.feed(
            (ROOT / "pages/ai-tarot-projection.html").read_text(encoding="utf-8")
        )

        workflow_links = [
            link for link in parser.links if link["attrs"].get("href") == GRAPH_URL
        ]
        self.assertEqual(len(workflow_links), 2)

        labels = {
            (
                "".join(link["texts"]["en"]),
                "".join(link["texts"]["zh"]),
            )
            for link in workflow_links
        }

        self.assertEqual(
            labels,
            {
                ("Explore the LangGraph workflow", "查看 LangGraph 工作流演示"),
                (
                    "Open the interactive LangGraph StateGraph demo →",
                    "打开交互式 LangGraph StateGraph 演示 →",
                ),
            },
        )
        for link in workflow_links:
            self.assertEqual(link["attrs"].get("target"), "_blank")
            self.assertIn("noopener", link["attrs"].get("rel", "").split())


if __name__ == "__main__":
    unittest.main()
