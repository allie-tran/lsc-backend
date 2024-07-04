import asyncio
import os
from typing import List

from groq import Groq
from groq.types.chat import ChatCompletionMessageParam
from llm.models import LLM

# Set up ChatGPT generation model
GROQ_AI = os.environ.get("GROQ_API", "")
MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "")


class GroqLLM(LLM):
    def __init__(self):
        self.client = Groq(api_key=GROQ_AI)
        self.model_name = MODEL_NAME

    async def generate(self, messages: List[ChatCompletionMessageParam]):
        """
        Generate completions from a list of messages
        """
        request = self.client.chat.completions.create(
            model=self.model_name, messages=messages, stream=True,
            temperature=0.1,
        )

        buffer = []
        loop = asyncio.get_event_loop()
        done = False

        async def yield_buffer():
            while not done:
                await asyncio.sleep(1)
                if buffer:
                    yield "".join(buffer)
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


# Load the model
groq_llm_model = GroqLLM()
