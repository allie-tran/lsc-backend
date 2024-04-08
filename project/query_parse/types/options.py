from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar

from configs import (
    DEFAULT_SIZE,
    FILTER_FIELDS,
    MAX_IMAGES_PER_EVENT,
    MAXIMUM_EVENT_TO_GROUP,
    MERGE_EVENTS,
    QUERY_PARSER,
)
from pydantic import BaseModel, Field, model_validator

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Option(BaseModel):
    """For defining options"""

    name: str = ""
    description: str = ""
    status: bool = True
    output: Optional[Dict[str, Any]] = None


class QueryParser(Option):
    """For parsing the query string"""

    status: bool = QUERY_PARSER
    name: str = "Parse Query"
    description: str = "Parse the query string to get a more specific query"


class QuestionParser(Option):
    """For parsing the question"""

    status: bool = False
    name: str = "Parse Question"
    description: str = "Parse the question to a retrieval statement"


class Merger(Option):
    """For grouping the events"""

    status: bool = MERGE_EVENTS
    group_by: list[str] = []
    max_per_group: int = MAXIMUM_EVENT_TO_GROUP
    name: str = "Merge Events"
    description: str = "Merge the events based on the given fields"


class ImageLimiter(Option):
    """For limiting the images"""

    status: bool = MAX_IMAGES_PER_EVENT > 0
    num_images: int = MAX_IMAGES_PER_EVENT
    name: str = "Limit Images"
    description: str = "Limit the number of images per event"


class FieldFilter(Option):
    """For filtering the fields"""

    status: bool = FILTER_FIELDS
    fields: list[str] = []
    name: str = "Filter Fields"
    description: str = "Filter the fields based on the given fields"


class Filter(Option):
    """For filtering the events"""

    pass


class SortOption(BaseModel):

    field: Optional[str] = None
    order: Literal["asc", "desc"] = "asc"


class Sorter(Option):
    """For sorting the events"""

    sort_by: list[SortOption] = []
    name: str = "Sort By"
    description: str = "Sort the events based on the given fields"


class AnswerComponent(Option):
    """For generating the answer"""

    top_k: int = 10
    name: str = "Generate Answer"
    description: str = "Generate the answer based on the events"


class SearchParams(BaseModel):
    """For defining search parameters"""

    # Simple parameters
    size: int = DEFAULT_SIZE

    # Optional parameters
    query: QueryParser = QueryParser()
    question: QuestionParser = QuestionParser()
    merge: Merger = Merger()
    limit_images: ImageLimiter = ImageLimiter()
    filter_fields: FieldFilter = FieldFilter()
    filter_events: Filter = Filter()
    sort: Sorter = Sorter()
    answer: AnswerComponent = AnswerComponent()


class FunctionWithArgs(BaseModel):
    """For defining a function with arguments"""

    function: Callable
    is_async: bool = False
    use_previous_output: bool = False
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    output_name: Optional[str] = None

    def prepare(self, output: Any):
        if self.use_previous_output:
            self.kwargs.update(output)
            # remove unused arguments
            for key in self.kwargs.copy():
                if key not in self.function.__code__.co_varnames:
                    self.kwargs.pop(key)

    def execute(self, output: Any) -> Any:
        """Execute the function"""
        self.prepare(output)
        return {self.output_name: self.function(*self.args, **self.kwargs)}

    async def async_execute(self, output: Any) -> Any:
        """Execute the function asynchronously"""
        self.prepare(output)
        output = await self.function(*self.args, **self.kwargs)
        return {self.output_name: output}

    @model_validator(mode="after")
    def check_output_name(self):
        """Check if the output name is valid"""
        if self.output_name is None:
            self.output_name = self.function.__name__
        return self


class Pipe(BaseModel):
    """For defining the pipeline"""

    # If the function is executed
    executed: bool = False

    # If the user modified the option
    modified: bool = False

    # Output of the function
    output: Dict[str, Any] = {}

    # Skipped
    skipped: bool = False
    skippable: bool = False
    default_output: Dict[str, Any] = Field(default_factory=dict, exclude=True)

    # Output not to be included
    exclude_output: List[str] = Field(default_factory=lambda: [], exclude=True)

    def clean_output(self, keys: List[str] = []):
        """Clean the output"""
        for key in self.exclude_output + keys:
            self.output.pop(key, None)

    def execute(self, functions: List[FunctionWithArgs]) -> Any:
        """Execute multiple function,
        the output of the previous function will be the input of the next function"""
        if not self.executed:
            for func in functions:
                self.output.update(func.execute(self.output))
            self.executed = True
        return self.output

    async def async_execute(self, functions: List[FunctionWithArgs]) -> Any:
        """Execute multiple function asynchronously,
        the output of the previous function will be the input of the next function"""
        if not self.executed:
            for func in functions:
                if func.is_async:
                    self.output.update(await func.async_execute(self.output))
                else:
                    self.output.update(func.execute(self.output))
            self.executed = True
        return self.output

    def add_output(self, output: Dict[str, Any]):
        """Add outputs without executing the function
        This is useful when the function is executed outside the pipeline
        """
        self.output.update(output)
        self.executed = True


class SearchPipeline(BaseModel):
    """For defining the search pipeline
    The point of this is when a user change the search paremeter,
    only changed part will be re-executed
    """

    # Simple parameters
    size: int = DEFAULT_SIZE
    top_k: int = 10

    # Configurable pipes
    query_parser: Pipe = Pipe(exclude_output=["query", "es_query", "search_text"])
    field_extractor: Pipe = Pipe(skippable=True)
    field_organizer: Pipe = Pipe(skippable=True, exclude_output=["results"])
    event_merger: Pipe = Pipe(skippable=True, exclude_output=["results"])
    image_limiter: Pipe = Pipe(
        skippable=True,
        exclude_output=["results"],
        output={"max_images": MAX_IMAGES_PER_EVENT},
    )

    def export(
        self,
    ) -> Dict[str, Any]:
        """Export the pipeline"""
        # First, clean the unnecessary output of the pipes
        self.query_parser.clean_output()
        self.field_extractor.clean_output()
        self.field_organizer.clean_output()
        self.event_merger.clean_output()
        self.image_limiter.clean_output()

        # Then, export the pipeline
        return self.model_dump()
