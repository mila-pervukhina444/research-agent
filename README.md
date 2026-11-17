# Research agent

A lightweight research agent that searches arXiv, explores papers through tool
calling, and produces literature summaries with references.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. For another provider, set `OPENAI_BASE_URL` and
choose a model that supports Chat Completions tool calling.

## Usage

```bash
uv run --env-file .env main.py \
  "Recent approaches to long-context reasoning" \
  --output review.md
```

Override the model with `--model`, or edit `configs/agent.toml` to change the
model, system prompt, search limit, and results per query. Existing output files
are never overwritten.

## How it works

The model chooses between two tools: `search_papers` returns matching papers with
abstract previews; `get_paper` retrieves a complete abstract. Tool results are
added to the conversation until the model writes a review or reaches the search
limit. At the limit, one final model call summarizes the collected evidence.

Paper metadata is cached within a run, and arXiv requests are spaced three
seconds apart. Reviews use metadata and abstracts, not full paper text. They
are a starting point for reading, not an exhaustive literature survey.

`agent/core.py` contains the agent loop, `agent/tools.py` implements the arXiv
tools, and `main.py` handles configuration and Markdown output.

## Development

```bash
uv run ruff check .
uv run ruff format .
```

API references: [tool calling](https://developers.openai.com/api/docs/guides/function-calling)
and the [arXiv API](https://info.arxiv.org/help/api/user-manual.html).

## License

MIT. See [LICENSE](LICENSE).
