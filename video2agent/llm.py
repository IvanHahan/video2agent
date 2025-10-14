import os
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.outputs import GenerationChunk
from openai import OpenAI
from pydantic import Field


class OpenAIModel(LLM):
    """LangChain-compatible wrapper for OpenAI's gpt-5-nano model."""

    client: OpenAI = Field(default=None, exclude=True)
    model: str = "gpt-5-nano"
    api_key: Optional[str] = None
    verbosity: str = "medium"
    reasoning_effort: str = "minimal"

    def __init__(
        self,
        model: str = "gpt-5-nano",
        api_key: Optional[str] = None,
        verbosity: str = "medium",
        reasoning_effort: str = "minimal",
    ):
        super().__init__()
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing OpenAI API key. Set OPENAI_API_KEY or pass api_key."
            )
        self.client = OpenAI(api_key=self.api_key)
        self.verbosity = verbosity
        self.reasoning_effort = reasoning_effort

    @property
    def _llm_type(self) -> str:
        return self.model

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Send prompt to GPT-5 Nano via OpenAI Responses API."""
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            text={"verbosity": self.verbosity},
            reasoning={"effort": self.reasoning_effort},
        )
        text = response.output_text.strip()
        if stop:
            for s in stop:
                text = text.split(s)[0]
        return text

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream responses from GPT-5 Nano via OpenAI Responses API."""
        stream = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            text={"verbosity": self.verbosity},
            reasoning={"effort": self.reasoning_effort},
            stream=True,
        )

        for event in stream:
            if event.type == "response.created":
                pass
            elif event.type == "response.output_text.delta":
                text = event.delta
                yield GenerationChunk(text=text)

                if run_manager:
                    run_manager.on_llm_new_token(text)
            elif event.type == "response.completed":
                pass
            elif event.type == "error":
                pass

                # Check for stop sequences
