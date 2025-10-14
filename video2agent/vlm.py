import base64
import os
from io import BytesIO
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.outputs import GenerationChunk
from loguru import logger
from openai import OpenAI, Timeout
from PIL import Image
from pydantic import Field

MODEL_URLS = {
    "qwenvl25_32b": "https://stereologic-vllm-qwen25-32b.hf.space/v1",
    "qwenvl3_30b": "https://stereologic-vllm-qwen3-vl-30b-a3b-instruct.hf.space/v1",
    "qwenvl25_72b": "https://stereologic-vllm-qwen25-72b.hf.space/v1",
    "internvl_14b": "https://stereologic-vllm-ogvl-internvl35-14b.hf.space/v1",
    "internvl_30b": "https://stereologic-vllm-ogvl-internvl3-5-30b-a3b.hf.space/v1",
}


def make_system_message(message):
    return {"role": "system", "content": message}


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


class VisionModel(LLM):
    """Enhanced OpenAI model client with response processing capabilities."""

    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: int = 60
    read_timeout_seconds: int = 300
    max_tokens: int = 1024
    default_temperature: float = 0.0
    client: OpenAI = Field(default=None, exclude=True)

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the OpenAI model client.

        Args:
            model_name: Name of the model to use
            base_url: Base URL for the API
            api_key: API key for authentication
            max_tries: Maximum number of retry attempts
        """
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key or os.getenv("HF_TOKEN")
        self.client = self._create_client()
        self._cached_model_name: Optional[str] = None

    def _create_client(self) -> OpenAI:
        """Create and configure the OpenAI client."""
        client_kwargs = {
            "base_url": self.base_url,
            "timeout": Timeout(
                self.timeout_seconds,
                connect=self.timeout_seconds,
                read=self.read_timeout_seconds,
                write=self.timeout_seconds,
            ),
        }

        if self.api_key is not None:
            client_kwargs["default_headers"] = {
                "Authorization": f"Bearer {self.api_key}"
            }

        return OpenAI(**client_kwargs)

    def _stream(
        self,
        prompt: str,
        images: List[Image.Image],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream responses from GPT-5 Nano via OpenAI Responses API."""
        messages = [make_system_message(prompt)]
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{read_image_as_base64(image) if isinstance(image, str) else image_to_base64(image)}",
                        },
                    }
                    for image in images
                ],
            },
        )

        kwargs["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)
        kwargs["temperature"] = kwargs.get("temperature", self.default_temperature)

        stream = self.client.responses.create(
            model=self.model_name,
            input=messages,
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

    def get_model_name(self) -> Optional[str]:
        """Get the model name, fetching from server if not specified."""
        if self._cached_model_name is None:
            try:
                models = self.client.models.list()
                self._cached_model_name = models.data[0].id if models.data else None
            except Exception as e:
                logger.error(f"Failed to fetch model name: {e}")
                self._cached_model_name = "default"

        return self._cached_model_name

    @property
    def _llm_type(self) -> str:
        return self.model_name

    def _call(
        self,
        prompt: str,
        images: List[Image.Image],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        messages = [make_system_message(prompt)]
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{read_image_as_base64(image) if isinstance(image, str) else image_to_base64(image)}",
                        },
                    }
                    for image in images
                ],
            },
        )

        kwargs["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)
        kwargs["temperature"] = kwargs.get("temperature", self.default_temperature)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs,
        )

        # response = self.client.responses.create(
        #     model=self.get_model_name(),
        #     input=messages,
        # )
        return response.choices[0].message.content


if __name__ == "__main__":
    import requests
    from dotenv import load_dotenv
    from PIL import Image

    load_dotenv()

    model = VisionModel(base_url=MODEL_URLS["qwenvl3_30b"])

    image_url = "https://letsenhance.io/static/73136da51c245e80edc6ccfe44888a99/396e9/MainBefore.jpg"  # Random image from Lorem Picsum
    response_img = requests.get(image_url)
    img = Image.open(BytesIO(response_img.content))

    response = model.invoke("Describe the image.", images=[img])
    for chunk in response:
        print(chunk.text, end="", flush=True)
