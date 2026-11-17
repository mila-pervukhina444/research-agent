import json
import re
import time
from xml.etree import ElementTree

import httpx


ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_API = "https://export.arxiv.org/api/query"
PAPER_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[a-zA-Z][a-zA-Z.-]*/\d{7})(?:v\d+)?")

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": (
                "Search arXiv and return paper metadata with short abstract previews. "
                "Use arXiv query syntax, e.g. all:retrieval AND all:reasoning."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper",
            "description": (
                "Get the complete abstract and metadata for an arXiv paper. "
                "Accepts a paper ID such as 1706.03762, not a URL. "
                "Does not retrieve the full paper text."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"paper_id": {"type": "string"}},
                "required": ["paper_id"],
                "additionalProperties": False,
            },
        },
    },
]


class ArxivTools:
    def __init__(self, client: httpx.Client, max_results: int = 5):
        self.client = client
        self.max_results = max_results
        self._last_request = None
        self._papers = {}

    def _fetch(self, params: dict) -> list[dict]:
        if self._last_request is not None:
            time.sleep(max(0, 3.0 - (time.monotonic() - self._last_request)))
        try:
            response = self.client.get(ARXIV_API, params=params)
            response.raise_for_status()
        finally:
            self._last_request = time.monotonic()

        feed = ElementTree.fromstring(response.content)
        if feed.tag != f"{ATOM}feed":
            raise ValueError("arXiv returned an unexpected response")
        papers = []
        for entry in feed.findall(f"{ATOM}entry"):
            identifier = entry.findtext(f"{ATOM}id", "")
            if identifier.endswith("/api/errors"):
                raise ValueError("arXiv rejected the query; check its search syntax")
            paper_id = identifier.partition("/abs/")[2]
            if not PAPER_ID.fullmatch(paper_id):
                raise ValueError("arXiv returned an invalid paper ID")
            paper = {
                "id": paper_id,
                "title": " ".join(entry.findtext(f"{ATOM}title", "").split()),
                "authors": [
                    author.findtext(f"{ATOM}name", "")
                    for author in entry.findall(f"{ATOM}author")
                ],
                "published": entry.findtext(f"{ATOM}published", "")[:10],
                "abstract": " ".join(entry.findtext(f"{ATOM}summary", "").split()),
                "url": f"https://arxiv.org/abs/{paper_id}",
            }
            self._papers[paper_id] = paper
            papers.append(paper)
        return papers

    def search_papers(self, query: str) -> list[dict]:
        papers = self._fetch(
            {
                "search_query": query,
                "start": 0,
                "max_results": self.max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        return [
            {
                **{key: value for key, value in paper.items() if key != "abstract"},
                "abstract_preview": paper["abstract"][:400],
            }
            for paper in papers
        ]

    def get_paper(self, paper_id: str) -> dict:
        if not PAPER_ID.fullmatch(paper_id):
            raise ValueError("Expected an arXiv paper ID, for example 1706.03762")
        if paper_id in self._papers:
            return self._papers[paper_id]
        papers = self._fetch({"id_list": paper_id, "max_results": 1})
        if not papers:
            raise ValueError("Paper not found")
        return papers[0]

    def execute(self, name: str, arguments: str) -> dict:
        handlers = {
            "search_papers": ("query", self.search_papers),
            "get_paper": ("paper_id", self.get_paper),
        }
        if name not in handlers:
            return {"error": "Unknown tool"}
        try:
            params = json.loads(arguments)
            key, handler = handlers[name]
            if not isinstance(params, dict) or set(params) != {key}:
                return {"error": f"Expected exactly one argument: {key}"}
            value = params[key]
            if not isinstance(value, str) or not value.strip():
                return {"error": f"{key} must be a non-empty string"}
            return {"result": handler(value.strip())}
        except json.JSONDecodeError:
            return {"error": "Tool arguments must be valid JSON"}
        except httpx.HTTPStatusError as exc:
            return {"error": f"arXiv returned HTTP {exc.response.status_code}"}
        except httpx.RequestError:
            return {"error": "Could not reach arXiv"}
        except ElementTree.ParseError:
            return {"error": "arXiv returned an invalid XML response"}
        except ValueError as exc:
            return {"error": str(exc)}
