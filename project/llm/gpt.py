import json
import os
from typing import List

from configs import JSON_END_FLAG, JSON_START_FLAG
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from partialjson.json_parser import JSONParser

from llm.prompts import INSTRUCTIONS

# Set up ChatGPT generation model
OPENAI_API = os.environ.get("OPENAI_API", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")


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

    async def generate_from_text(self, text: str):
        """
        Generate completions from text
        Then parse the JSON object from the completion
        If the completion is not a JSON object, return the text
        """
        messages = [self.template_message]
        messages.append(ChatCompletionUserMessageParam(role="user", content=text))
        text = ""
        async for completion in self.generate(messages):
            text += completion
        if JSON_START_FLAG in text:
            text = text.split(JSON_START_FLAG)[1]
        if JSON_END_FLAG in text:
            text = text.split(JSON_END_FLAG)[0]
        try:
            json_object = parser.parse(text)
            return json_object
        except json.JSONDecodeError:
            return text


# Load the model and JSON parser
llm_model = LLM()
parser = JSONParser()
