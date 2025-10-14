import base64
import os
from io import BytesIO
from typing import Any, Iterator, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.outputs import GenerationChunk
from openai import OpenAI
from PIL import Image
from pydantic import Field


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode()


def read_image_as_base64(image_path: str) -> str:
    """
    Read an image file and convert it to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    with Image.open(image_path) as image:
        return image_to_base64(image)


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
        image: Optional[Image.Image] = None,
        text_format: Optional[Type] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Send prompt to GPT-5 Nano via OpenAI Responses API."""
        if image is not None:
            if isinstance(image, str):
                image = read_image_as_base64(image)
            else:
                image = image_to_base64(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image}",
                        },
                    ],
                },
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        if text_format is not None:
            response = self.client.responses.parse(
                model=self.model,
                input=messages,
                text={"verbosity": kwargs.pop("verbosity", self.verbosity)},
                reasoning={
                    "effort": kwargs.pop("reasoning_effort", self.reasoning_effort)
                },
                text_format=text_format,
                **kwargs,
            )
        else:
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                text={"verbosity": kwargs.pop("verbosity", self.verbosity)},
                reasoning={
                    "effort": kwargs.pop("reasoning_effort", self.reasoning_effort)
                },
                **kwargs,
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
        image: Optional[Image.Image] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream responses from GPT-5 Nano via OpenAI Responses API."""
        if image is not None:
            if isinstance(image, str):
                image = read_image_as_base64(image)
            else:
                image = image_to_base64(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image}",
                        },
                    ],
                },
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        stream = self.client.responses.create(
            model=self.model,
            input=messages,
            text={"verbosity": kwargs.pop("verbosity", self.verbosity)},
            reasoning={"effort": kwargs.pop("reasoning_effort", self.reasoning_effort)},
            stream=True,
            **kwargs,
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


if __name__ == "__main__":
    import requests

    model = OpenAIModel()

    prompt = "Describe the image in detail."

    image_url = "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d"
    image_data = requests.get(image_url).content
    image = Image.open(BytesIO(image_data))

    response = model.invoke(
        [{"role": "system", "content": "You are a helpful assistant."}]
        + [{"role": "user", "content": prompt}],
        image=image,
    )
    print("Response:", response)
