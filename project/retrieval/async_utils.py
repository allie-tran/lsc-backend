import asyncio
import heapq
from typing import Any, AsyncGenerator, Coroutine
import time
from rich import print

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
# Decorators                #
# ========================= #
def format_time(time: float) -> str:
    """
    Format time seconds into human readable format
    """
    if time < 1:
        return f"{time * 1000:.0f} ms"
    return f"{time:.2f} s"

# Timing decorator for coroutines
def async_timer(name: str):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            print(f"[orange]{name}[/orange]:start")
            start = time.time()
            result = await func(*args, **kwargs)
            end = time.time()
            print(f"[orange]{name}[/orange]:cost {format_time((end - start))}")

            return result

        return wrapper

    return decorator

# for async generator
def async_generator_timer(name: str, track_yield: bool = False):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            print(f"[orange]{name}[/orange]:start")
            start = time.time()
            async for result in func(*args, **kwargs):
                yield result
                if track_yield:
                    end = time.time()
                    print(f"[orange]{name}[/orange]:cost {format_time((end - start))}")

            end = time.time()
            print(f"[orange]{name}[/orange]:cost {format_time((end - start))}")

        return wrapper

    return decorator

# normal timer
def timer(name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"[orange]{name}[/orange]:start")
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"[orange]{name}[/orange]:cost {format_time((end - start))}")
            return result

        return wrapper

    return decorator

# ========================= #
# Application specific code #
# ========================= #
