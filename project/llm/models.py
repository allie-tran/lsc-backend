import json
import os
from collections.abc import Sequence
from typing import AsyncGenerator, Dict, Generator, List, Literal, Optional

from configs import DEBUG, JSON_END_FLAG, JSON_START_FLAG
from openai import AsyncOpenAI, BaseModel
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from partialjson.json_parser import JSONParser
from pyrate_limiter import BucketFullException, Duration, Limiter, Rate
from rich import print

from llm.prompts import INSTRUCTIONS
from retrieval.async_utils import async_generator_timer, async_timer

parser = JSONParser()
parser.on_extra_token = lambda *_, **__: None

rate = Rate(3, Duration.SECOND)
limiter = Limiter(rate)

# Set up ChatGPT generation model
OPENAI_API = os.environ.get("OPENAI_API", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")


class MixedContent(BaseModel):
    type: Literal["text", "image_url"]
    content: str


class LLM:
    # Set up the template messages to use for the completion
    template_message: ChatCompletionMessageParam = ChatCompletionSystemMessageParam(
        role="system", content=INSTRUCTIONS
    )

    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API)
        self.model_name = MODEL_NAME

    async def generate(self, messages: List[ChatCompletionMessageParam]):  # type: ignore
        """
        Generate completions from a list of messages
        """
        request = await self.client.chat.completions.create(
            model=self.model_name, messages=messages, stream=True,
            temperature=0.2
        )

        async for chunk in request:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def __parse(self, response: str) -> Generator[Dict, None, None]:
        # there might be MULTIPLE JSON objects in the response
        # we need to split them and parse them individually
        while JSON_START_FLAG in response:
            start = response.find(JSON_START_FLAG)
            response = response[start + len(JSON_START_FLAG) :]
            json_object = response
            if JSON_END_FLAG in response:
                end = response.find(JSON_END_FLAG)
                json_object = response[: end + len(JSON_END_FLAG)]
                response = response[end + len(JSON_END_FLAG) :]
            try:
                json_object = parser.parse(json_object)
                yield json_object
            except json.JSONDecodeError:
                pass

    async def __generate_and_parse(
        self, messages: List[ChatCompletionMessageParam], stream=False
    ) -> AsyncGenerator[Dict, None]:
        """
        Generate completions from a list of messages
        Then parse the JSON object from the completion
        If the completion is not a JSON object, return the text
        """
        res = ""
        if DEBUG:
            print("Generating completions...")

        async for completion in self.generate(messages):
            res += completion
            response = res

            if DEBUG:
                print(completion, end="")

            if stream:
                try:
                    limiter.try_acquire("1")
                    all_objects = {}
                    for obj in self.__parse(response):
                        try:
                            all_objects.update(obj)
                        except ValueError:
                            pass
                            print("Error parsing object")
                            print("[ERROR]", obj)
                    yield all_objects
                except BucketFullException:
                    continue

        all_objects = {}
        for obj in self.__parse(res):
            try:
                if obj:
                    all_objects.update(obj)
            except ValueError:
                print("Error parsing object")
                print("[ERROR]", obj)
                pass

        yield all_objects

    @async_timer("generate_from_text")
    async def generate_from_text(self, text: str) -> Optional[Dict]:
        """
        Generate completions from text
        Then parse the JSON object from the completion
        If the completion is not a JSON object, return the text
        """
        messages = [self.template_message]
        messages.append(ChatCompletionUserMessageParam(role="user", content=text))
        res = None
        async for data in self.__generate_and_parse(messages):
            res = data
        return res

    @async_generator_timer("stream_from_text")
    async def stream_from_text(self, text: str) -> AsyncGenerator[Dict, None]:
        """
        Generate completions from text
        Then parse the JSON object from the completion
        If the completion is not a JSON object, return the text
        """
        messages = [self.template_message]
        messages.append(ChatCompletionUserMessageParam(role="user", content=text))
        async for data in self.__generate_and_parse(messages, stream=True):
            yield data

    @async_generator_timer("generate_from_mixed_media")
    async def generate_from_mixed_media(self, data: Sequence[MixedContent])-> AsyncGenerator[Dict, None]:
        messages = [self.template_message]
        content: List[ChatCompletionContentPartParam] = []
        for part in data:
            if part.type == "text":
                content.append(
                    ChatCompletionContentPartTextParam(text=part.content, type="text")
                )
            elif part.type == "image_url":
                content.append(
                    ChatCompletionContentPartImageParam(
                        image_url=ImageURL(url=part.content), type="image_url"
                    )
                )
        messages.append(ChatCompletionUserMessageParam(role="user", content=content))
        async for completion in self.__generate_and_parse(messages):
            if DEBUG:
                print("GPT", completion)
            yield completion


# Load the model and JSON parser
gpt_llm_model = LLM()
