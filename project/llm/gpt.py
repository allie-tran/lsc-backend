import json
import os
from collections.abc import Sequence
from typing import List, Literal

import commentjson
from configs import JSON_END_FLAG, JSON_START_FLAG
from openai import AsyncOpenAI
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
from pydantic import BaseModel

from llm.prompts import INSTRUCTIONS

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

    async def generate(self, messages: List[ChatCompletionMessageParam]):
        """
        Generate completions from a list of messages
        """
        request = await self.client.chat.completions.create(
            model=MODEL_NAME, messages=messages, stream=True
        )
        async for chunk in request:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def __parse(self, response: str):
        if JSON_START_FLAG in response:
            response = response.split(JSON_START_FLAG)[1]
        if JSON_END_FLAG in response:
            response = response.split(JSON_END_FLAG)[0]

        try:
            json_object = parser.parse(response)
            return json_object
        except json.JSONDecodeError:
            try:
                json_object = commentjson.loads(response)
                return json_object
            except:
                pass

    async def __generate_and_parse(self, messages: List[ChatCompletionMessageParam]):
        """
        Generate completions from a list of messages
        Then parse the JSON object from the completion
        If the completion is not a JSON object, return the text
        """
        response = ""
        async for completion in self.generate(messages):
            response += completion

            obj = self.__parse(response)
            if obj:
                yield obj

        obj = self.__parse(response)
        if obj:
            yield obj

    async def generate_from_text(self, response: str):
        """
        Generate completions from text
        Then parse the JSON object from the completion
        If the completion is not a JSON object, return the text
        """
        messages = [self.template_message]
        messages.append(ChatCompletionUserMessageParam(role="user", content=response))
        async for completion in self.__generate_and_parse(messages):
            yield completion

    async def generate_from_mixed_media(self, data: Sequence[MixedContent]):
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
            yield completion


# Load the model and JSON parser
gpt_llm_model = LLM()
parser = JSONParser()
