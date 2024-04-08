import json
import os
from typing import List

import commentjson
from configs import JSON_END_FLAG, JSON_START_FLAG
from partialjson.json_parser import JSONParser

from groq import Groq
from groq.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from llm.prompts import INSTRUCTIONS

# Set up ChatGPT generation model
GROQ_AI = os.environ.get("GROQ_API", "")
MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "")


class LLM:
    # Set up the template messages to use for the completion
    template_message: ChatCompletionMessageParam = ChatCompletionSystemMessageParam(
        role="system", content=INSTRUCTIONS
    )

    def __init__(self):
        self.client = Groq(api_key=GROQ_AI)

    def generate(self, messages: List[ChatCompletionMessageParam]):
        """
        Generate completions from a list of messages
        """
        request = self.client.chat.completions.create(
            model=MODEL_NAME, messages=messages, stream=True
        )
        for chunk in request:
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

    def __generate_and_parse(
        self, messages: List[ChatCompletionMessageParam], stream=False
    ):
        """
        Generate completions from a list of messages
        Then parse the JSON object from the completion
        If the completion is not a JSON object, return the text
        """
        res = ""
        for completion in self.generate(messages):
            res += completion
            response = res

            if stream:
                obj = self.__parse(response)
                if obj:
                    yield obj

        obj = self.__parse(res)
        if obj:
            yield obj

    async def generate_from_text(self, text: str):
        """
        Generate completions from text
        Then parse the JSON object from the completion
        If the completion is not a JSON object, return the text
        """
        messages = [self.template_message]
        messages.append(ChatCompletionUserMessageParam(role="user", content=text))
        res = ""
        for completion in self.__generate_and_parse(messages):
            res = completion
        return res

    async def stream_from_text(self, text: str):
        """
        Generate completions from text
        Then parse the JSON object from the completion
        If the completion is not a JSON object, return the text
        """
        messages = [self.template_message]
        messages.append(ChatCompletionUserMessageParam(role="user", content=text))
        for completion in self.__generate_and_parse(messages, stream=True):
            yield completion


# Load the model and JSON parser
llm_model = LLM()
parser = JSONParser()
