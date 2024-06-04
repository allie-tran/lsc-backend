import asyncio
import heapq
from typing import Any, AsyncGenerator, Coroutine

# ========================= #
# Main utility functions    #
# ========================= #

async def merge_generators(*generators: AsyncGenerator) -> AsyncGenerator[Any, None]:
    """
    Merge multiple async generators into one and yield values in order of appearance
    """
    priority_queue = []
    next_idx = 0

    async def add_to_queue(generator, idx):
        nonlocal next_idx
        async for value in generator:
            heapq.heappush(priority_queue, (next_idx, value, idx))
            next_idx += 1

    tasks = [add_to_queue(generator, idx) for idx, generator in enumerate(generators)]
    await asyncio.gather(*tasks)

    while priority_queue:
        _, value, _ = heapq.heappop(priority_queue)
        yield value

async def merge_coroutines(*coroutines: Coroutine) -> AsyncGenerator[Any, None]:
    """
    Merge multiple coroutines into one and yield values in order of appearance
    """
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    for future in asyncio.as_completed(tasks):
        yield await future

# ========================= #
# Application specific code #
# ========================= #
