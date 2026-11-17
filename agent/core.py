import json
import logging
from dataclasses import dataclass

from openai import OpenAI

from agent.tools import TOOL_SCHEMAS, ArxivTools


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    model: str
    system_prompt: str
    max_steps: int = 8
    max_results: int = 5

    def __post_init__(self):
        for name in ("model", "system_prompt"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name, upper in (("max_steps", 30), ("max_results", 20)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer between 1 and {upper}")


class ResearchAgent:
    def __init__(self, client: OpenAI, tools: ArxivTools, settings: Settings):
        self.client = client
        self.tools = tools
        self.settings = settings

    def run(self, question: str) -> str:
        if not question.strip():
            raise ValueError("The research question cannot be empty")

        messages = [
            {"role": "system", "content": self.settings.system_prompt},
            {"role": "user", "content": question},
        ]

        for step in range(self.settings.max_steps + 1):
            final_step = step == self.settings.max_steps
            if final_step:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The search budget is exhausted. Write the review using "
                            "only retrieved evidence and state any remaining gaps."
                        ),
                    }
                )

            logger.info("Model step %s", step + 1)
            response = self.client.chat.completions.create(
                model=self.settings.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="none" if final_step else "auto",
                parallel_tool_calls=False,
            )
            if not response.choices:
                raise RuntimeError("The model returned no choices")
            choice = response.choices[0]
            if choice.finish_reason in {"length", "content_filter"}:
                raise RuntimeError(f"The model stopped early: {choice.finish_reason}")
            message = choice.message

            if not message.tool_calls:
                if not message.content or not message.content.strip():
                    raise RuntimeError("The model returned an empty review")
                return message.content.strip()
            if final_step:
                raise RuntimeError("The model requested tools after the search limit")

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        call.model_dump(include={"id", "type", "function"})
                        for call in message.tool_calls
                    ],
                }
            )
            for call in message.tool_calls:
                logger.info("Calling %s", call.function.name)
                result = self.tools.execute(call.function.name, call.function.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        raise RuntimeError("The agent did not produce a review")
