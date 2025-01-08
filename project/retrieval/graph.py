import asyncio
import logging
from typing import Optional

import pandas as pd
from llm import gpt_llm_model
from llm.prompt.organize import GRAPH_QUERY
from pydantic import ValidationError
from query_parse.extract_info import create_es_combo_query, create_query
from query_parse.question import detect_question, question_to_retrieval
from query_parse.types.options import FunctionWithArgs, SearchPipeline
from query_parse.types.requests import Data, Step, Task
from results.utils import (
    RelevantFields,
    create_event_label,
    limit_images_per_event,
    merge_events,
)
from rich import print as rprint

from retrieval.async_utils import async_timer
from retrieval.search import get_search_tasks
from retrieval.search_utils import organize_by_relevant_fields

logger = logging.getLogger(__name__)


@async_timer("to_csv")
async def to_csv(
    text: str,
    data: Data,
    pipeline: Optional[SearchPipeline] = None,
    task_type: Task = Task.NONE,
) -> pd.DataFrame:
    """
    Search (and answer) a single query
    """

    if not pipeline:
        pipeline = SearchPipeline()

    step = Step(step=1, total=2)
    # ============================= #
    # 1. Query Parser (no skipping but modifiable)
    # ============================= #
    output = await pipeline.query_parser.async_execute(
        [
            FunctionWithArgs(
                function=detect_question, args=[text], output_name="is_question"
            ),
            FunctionWithArgs(
                function=question_to_retrieval,
                args=[text],
                use_previous_output=True,
                output_name="search_text",
                is_async=True,
            ),
            FunctionWithArgs(
                function=create_query,
                use_previous_output=True,
                output_name="query",
                is_async=True,
            ),
            FunctionWithArgs(  # no skipping
                function=create_es_combo_query,
                use_previous_output=True,
                kwargs={"ignore_limit_score": False},
                output_name="es_query",
                is_async=True,
            ),
        ]
    )

    if output["is_question"]:
        step.total = 4

    configs = output["query"].print_info()
    pipeline.query_parser.add_output(configs)

    # ============================= #
    # 2. Search (Field extractor can be skipped)
    # ============================= #
    # a. Check if we need to extract the relevant fields
    field_extractor = pipeline.field_extractor
    skip_extract = task_type == Task.AD_HOC
    async_tasks = get_search_tasks(
        output["es_query"],
        pipeline.size,
        text,
        data=data,
        tag="single",
        filter_fields=skip_extract,
    )
    # ----------------------------- #
    # b. Start the async tasks
    results = None
    relevant_fields = RelevantFields()

    for future in asyncio.as_completed(async_tasks):
        res = await future
        if res.task_type == "search":
            results = res.results
            step.step += 1
        elif res.task_type == "llm":
            relevant_fields = res.results
            field_extractor.add_output(relevant_fields.model_dump())

    if results is None:
        print("[red]to_csv: No results found[/red]")
        return pd.DataFrame()

    # ============================= #
    # 3. Processing the results
    # ============================= #
    # a. Organize the results by relevant fields
    pipeline.field_organizer.default_output = {"results": results}
    results = pipeline.field_organizer.execute(
        [
            FunctionWithArgs(
                function=organize_by_relevant_fields,
                args=[results, relevant_fields.relevant_fields],
                output_name="results",
            )
        ]
    )["results"]

    # ----------------------------- #
    # b. Merge the events
    pipeline.event_merger.default_output = {"results": results}
    results = pipeline.event_merger.execute(
        [
            FunctionWithArgs(
                function=merge_events,
                args=[results, relevant_fields],
                output_name="results",
            )
        ]
    )["results"]

    # ----------------------------- #
    # c. Limit the images
    pipeline.image_limiter.default_output = {"results": results}
    if task_type != Task.AD_HOC:
        results = pipeline.image_limiter.execute(
            [
                FunctionWithArgs(
                    function=limit_images_per_event,
                    args=[results, text, pipeline.image_limiter.output["max_images"]],
                    output_name="results",
                )
            ]
        )["results"]

    # ----------------------------- #
    # d. Check if anything changed
    # Not actually part of the pipeline
    unchanged = all(
        p.skipped
        for p in [
            pipeline.field_organizer,
            pipeline.event_merger,
            pipeline.image_limiter,
        ]
    )
    if not unchanged:
        print("[blue]Some changes detected[/blue]")
        results = create_event_label(results, relevant_fields.relevant_fields)

    step.step += 1
    # ============================= #
    # 4. To CSV
    # ============================= #
    events = results.events

    # Add number of images
    for event in events:
        event.duration = len(event.images) // 2

    df = pd.json_normalize(
        [
            e.model_dump(
                exclude=["images", "markers"],
                exclude_none=True,
                exclude_defaults=True,
            )
            for e in events
        ]
    )
    print(len(events), "events found")
    return df


@async_timer("get_vegalite")
async def get_vegalite(query: str, data: Data):
    """
    Get the vegalite data for a query
    """
    # Get data
    df = await to_csv(query, data)
    prompt = GRAPH_QUERY.format(question=query, data=df.to_dict(orient="records"))

    graph_data = {}
    while True:
        try:
            graph_data = await gpt_llm_model.generate_from_text(prompt)
            if graph_data:
                rprint("Vegetalite data found", graph_data)
                if "data" not in graph_data:
                    graph_data["data"] = {"values": df.to_dict(orient="records")}
                return graph_data
        except ValidationError as e:
            rprint(e)
