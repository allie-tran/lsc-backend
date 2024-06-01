import asyncio
import json
import os
from typing import AsyncGenerator, Dict, Generator, List, Optional

from configs import JSON_END_FLAG, JSON_START_FLAG
from partialjson.json_parser import JSONParser

from groq import Groq
from groq.types.chat import (ChatCompletionMessageParam,
                             ChatCompletionSystemMessageParam,
                             ChatCompletionUserMessageParam)
from llm.prompts import INSTRUCTIONS

# Set up ChatGPT generation model
GROQ_AI = os.environ.get("GROQ_API", "")
MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "")
parser = JSONParser()
parser.on_extra_token = lambda text, data, reminding: None  # type: ignore
from pyrate_limiter import BucketFullException, Duration, Limiter, Rate

rate  = Rate(3, Duration.SECOND)
limiter = Limiter(rate)

class LLM:
    # Set up the template messages to use for the completion
    template_message: ChatCompletionMessageParam = ChatCompletionSystemMessageParam(
        role="system", content=INSTRUCTIONS
    )

    def __init__(self):
        self.client = Groq(api_key=GROQ_AI)

    async def generate(self, messages: List[ChatCompletionMessageParam]):
        """
        Generate completions from a list of messages
        """
        request = self.client.chat.completions.create(
            model=MODEL_NAME, messages=messages, stream=True
        )

        buffer = []
        loop = asyncio.get_event_loop()
        done = False

        async def yield_buffer():
            while not done:
                await asyncio.sleep(1)
                if buffer:
                    yield ''.join(buffer)
                    buffer.clear()

        async def accumulate_chunks():
            nonlocal done
            for chunk in request:
                if chunk.choices[0].delta.content is not None:
                    buffer.append(chunk.choices[0].delta.content)
            done = True

        loop.create_task(accumulate_chunks())
        async for completion in yield_buffer():
            yield completion


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
        async for completion in self.generate(messages):
            res += completion
            response = res

            if stream:
                try:
                    limiter.try_acquire("1")
                    all_objects = {}
                    for obj in self.__parse(response):
                        all_objects.update(obj)
                    print("Yielding", all_objects)
                    yield all_objects
                except BucketFullException:
                    continue
        all_objects = {}
        for obj in self.__parse(res):
            all_objects.update(obj)
        yield all_objects

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


# Load the model
llm_model = LLM()
