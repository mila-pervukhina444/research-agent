import argparse
import logging
import os
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import httpx
from openai import APIError, APIStatusError, OpenAI

from agent.core import ResearchAgent, Settings
from agent.tools import ArxivTools


def main() -> int:
    parser = argparse.ArgumentParser(description="Research a topic using arXiv papers.")
    parser.add_argument("question", help="Research question or topic")
    parser.add_argument("--output", type=Path, default=Path("review.md"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "agent.toml",
    )
    parser.add_argument("--model", help="Override the model in the configuration")
    args = parser.parse_args()

    if not args.question.strip():
        parser.error("the research question cannot be empty")
    if args.output.exists():
        parser.error("output already exists; choose another --output path")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("set OPENAI_API_KEY or use uv run --env-file .env")

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("agent").setLevel(logging.INFO)
    try:
        with args.config.open("rb") as source:
            settings = Settings(**tomllib.load(source)["agent"])
        if args.model:
            settings = replace(settings, model=args.model)
    except (OSError, ValueError, TypeError, KeyError):
        parser.error("invalid configuration; check the [agent] section and its values")

    try:
        with (
            OpenAI(timeout=60.0, max_retries=2) as client,
            httpx.Client(
                timeout=30.0,
                follow_redirects=True,
                headers={"User-Agent": "research-agent/0.1.0"},
            ) as http_client,
        ):
            tools = ArxivTools(http_client, max_results=settings.max_results)
            review = ResearchAgent(client, tools, settings).run(args.question)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as target:
            target.write(review + "\n")
    except APIStatusError as exc:
        print(f"Model API returned HTTP {exc.status_code}.", file=sys.stderr)
        return 1
    except APIError:
        print(
            "Model API request failed; check the endpoint and credentials.",
            file=sys.stderr,
        )
        return 1
    except OSError:
        print("Could not write the review; check the output path.", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"Research failed: {exc}", file=sys.stderr)
        return 1

    print(f"Review saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
