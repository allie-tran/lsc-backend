import os

from configs import IMAGE_DIRECTORY
from llm import internlm_model
from results.models import EventResults

async def answer_visual_only(question: str, events: EventResults, k: int = 10):
    """
    Answer the question using the visual information only
    K: number of events to consider
    """
    if not internlm_model.loaded:
        internlm_model.load_model()

    # Get the list of images
    images = []
    answers = []
    for event in events.events[:k]:
        image_paths = [os.path.join(IMAGE_DIRECTORY, img) for img in event.images]
        answer = await internlm_model.answer_question(question, image_paths)
        if "I'm sorry" in answer:
            continue
        yield answer
    # return answers

