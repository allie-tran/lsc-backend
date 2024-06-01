from functools import cmp_to_key
from typing import Iterable, List, Literal, Optional, Set

from configs import DERIVABLE_FIELDS, EXCLUDE_FIELDS, ISEQUAL, MAXIMUM_EVENT_TO_GROUP
from pydantic import BaseModel, InstanceOf, field_validator, validate_call
from query_parse.visual import score_images
from retrieval.common_nn import encode_query
from rich import print

from results.models import Event, EventResults, GenericEventResults, Image


def deriving_fields(
    events: List[InstanceOf[Event]], fields: List[str]
) -> List[InstanceOf[Event]]:
    """
    Derive the fields
    """
    derived_events = []
    for event in events:
        derived_event = event.copy_to_derived_event()
        for field in fields:
            setattr(derived_event, field, DERIVABLE_FIELDS[field](event))
        derived_events.append(derived_event)
    return derived_events


class TimeGap(BaseModel):
    unit: str
    value: int


class LocationGap(BaseModel):
    unit: str
    value: int


class MaxGap(BaseModel):
    time_gap: Optional[TimeGap] = None
    gps_gap: Optional[LocationGap] = None

    @field_validator("time_gap")
    def validate_time_gap(cls, v):
        if v is not None:
            if v.unit not in ["none", "hour", "minute", "day", "week", "month", "year"]:
                return None
        return v

    @field_validator("gps_gap")
    def validate_gps_gap(cls, v):
        if v is not None:
            if v.unit not in ["none", "meter", "km"]:
                return None
        return v


def custom_compare_function(
    event1: Event, event2: Event, fields: Iterable[str], max_gap: MaxGap
) -> int:
    """
    Custom compare function
    """
    # Assert if the two events are the same group first
    # if not then just compare the scene ids
    equal = True
    for field in fields:
        cmp_field = field
        if field not in ISEQUAL:
            cmp_field = "*"
        if not ISEQUAL[cmp_field](getattr(event1, field), getattr(event2, field)):
            equal = False
            break

    # Check the time gap
    if max_gap.time_gap is not None and max_gap.time_gap.unit == "none":
        time_gap = max_gap.time_gap
        match time_gap.unit:
            case "hour":
                if (
                    abs((event1.start_time - event2.start_time).seconds)
                    > time_gap.value * 3600
                ):
                    equal = False
            case "minute":
                if (
                    abs((event1.start_time - event2.start_time).seconds)
                    > time_gap.value * 60
                ):
                    equal = False
            case "day":
                if abs((event1.start_time - event2.start_time).days) > time_gap.value:
                    equal = False
            case "week":
                if (
                    abs((event1.start_time - event2.start_time).days)
                    > time_gap.value * 7
                ):
                    equal = False
            case "month":
                if (
                    abs((event1.start_time - event2.start_time).days)
                    > time_gap.value * 30
                ):
                    equal = False
            case "year":
                if (
                    abs((event1.start_time - event2.start_time).days)
                    > time_gap.value * 365
                ):
                    equal = False
            case _:
                pass

    # Check the location gap
    if max_gap.gps_gap is not None and max_gap.gps_gap.unit == "none":
        pass

    if equal:
        return 0

    # If the two events are not the same group then compare the sceneid
    return 1 if event1.scene > event2.scene else -1


class SortBy(BaseModel):
    field: str
    order: Literal["asc", "desc"]


@validate_call
def merge_events(
    results: EventResults, groupby: Set[str], sortby: List[SortBy], max_gap: MaxGap
) -> EventResults:
    """
    Merge the events
    """
    # Check if any of the groupby should be calculated
    available_fields = set(type(results.events[0]).model_fields)
    derive_fields = []
    to_remove = set()
    for criteria in groupby:
        if criteria not in available_fields:
            # Two cases here:
            # 1. the field is derivable from the schema
            if criteria in DERIVABLE_FIELDS:
                derive_fields.append(criteria)
            # 2. the field is not derivable from the schema
            else:
                print("[red]Cannot derive field[/red]", criteria)
                to_remove.add(criteria)

    # Remove the fields that cannot be derived
    groupby = groupby.difference(to_remove)

    # Derive the fields
    if derive_fields:
        results.events = deriving_fields(results.events, derive_fields)

    if len(results.events) == 1:
        return results

    # If groupby is empty then group by "group"
    if not groupby:
        groupby = set(["group"])

    cmp = lambda x, y: custom_compare_function(x, y, groupby, max_gap)

    # Group the events using the ISEQUAL criteria from configs
    temp = results.events
    sorted(temp, key=cmp_to_key(cmp))

    # Get unique keys
    unique_groups = 0
    events = results.events
    scores = results.scores
    grouped_events = {0: [events[0]]}
    grouped_scores = {0: [scores[0]]}

    print(f"[blue]Grouping {len(events)} events by {groupby}[/blue]")
    for event, score in zip(events[1:], scores[1:]):
        found = False
        for group in grouped_events:
            if cmp(event, grouped_events[group][0]) == 0:
                found = True
                if len(grouped_events[group]) < MAXIMUM_EVENT_TO_GROUP:
                    grouped_events[group].append(event)
                    grouped_scores[group].append(score)
                break
        if not found:
            unique_groups += 1
            grouped_events[unique_groups] = [event]
            grouped_scores[unique_groups] = [score]

    print(f"[blue]{unique_groups + 1} unique groups[/blue]")

    new_results = []
    new_scores = []
    for group in grouped_events:
        # Merge the events
        events = grouped_events[group]
        scores = grouped_scores[group]

        if len(events) == 1:
            new_results.append(events[0])
            new_scores.append(scores[0])
            continue

        new_event = events[0]
        to_merge = events[1 : 1 + MAXIMUM_EVENT_TO_GROUP]
        to_merge_scores = scores[1 : 1 + MAXIMUM_EVENT_TO_GROUP]

        new_event.merge_with_many(scores[0], to_merge, to_merge_scores)
        new_results.append(new_event)
        new_scores.append(grouped_scores[group][0])

    if sortby:
        for sort in sortby:
            field = sort.field
            order = sort.order
            if order == "asc":
                new_results, new_scores = zip(
                    *sorted(
                        zip(new_results, new_scores),
                        key=lambda x: getattr(x[0], field),
                    )
                )
            else:
                new_results, new_scores = zip(
                    *sorted(
                        zip(new_results, new_scores),
                        key=lambda x: getattr(x[0], field),
                        reverse=True,
                    )
                )

    print("[green]Merged into[/green]", len(new_results), "events")
    return EventResults(
        events=new_results,
        scores=new_scores,
        relevant_fields=results.relevant_fields + derive_fields,
        min_score=results.min_score,
        max_score=results.max_score,
    )


def limit_images_per_event(
    results: GenericEventResults, text_query: str, max_images: int
) -> GenericEventResults:
    """
    Limit the number of images per event
    This is achieved by selecting the images with the highest score
    and highest relevance to the query
    """
    for generic_event in results.events:
        for event in generic_event.custom_iter():  # might be a doublet or triplet

            images: List[Image] = event.images
            scores = event.image_scores

            if len(images) <= max_images:
                continue

            encoded_query = encode_query(text_query)
            visual_scores = score_images(images, encoded_query)

            # Sort the images by the visual score and the original score
            ensembled_scores = [
                (visual_score + 1) * (score + 1)
                for visual_score, score in zip(visual_scores, scores)
            ]

            sorted_images = [
                x for _, x in sorted(zip(ensembled_scores, images), reverse=True)
            ]
            chosen_images = sorted_images[:max_images]
            indices = sorted([images.index(image) for image in chosen_images])

            new_images = [images[i] for i in indices]
            new_scores = [scores[i] for i in indices]

            # Update the event
            assert len(images) > 0, "No images"
            event.images = new_images
            event.image_scores = new_scores

    return results


def basic_label(event: Event) -> str:
    """
    Create a basic label for the event
    """
    start_time = event.start_time.strftime("%H:%M")
    end_time = event.end_time.strftime("%H:%M")
    location = event.location
    date = event.start_time.strftime("%d, %b %Y")
    return f"<strong>{location}</strong>\n{date}, {start_time} - {end_time}"


def create_event_label(
    results: GenericEventResults, relevant_fields: List[str] = []
) -> GenericEventResults:
    """
    Create a label for the event
    """
    if not relevant_fields:
        # Get the basic fields: location, time, date
        for generic_event in results.events:
            for event in generic_event.custom_iter():
                event.name = basic_label(event)
    else:
        all_fields = set(relevant_fields)
        done = set()
        for generic_event in results.events:
            for event in generic_event.custom_iter():
                label = {}
                # if both start_time and end_time are in the important fields
                if "start_time" in all_fields and "end_time" in all_fields:
                    start_time = event.start_time.strftime("%H:%M")
                    end_time = event.end_time.strftime("%H:%M")
                    label["time"] = f"{start_time} - {end_time}"
                    done.update(["start_time", "end_time", "hour", "minute", "time"])
                elif "start_time" in all_fields:
                    start_time = event.start_time.strftime("%H:%M")
                    label["time"] = f"{start_time}"
                    done.update(["start_time", "hour", "minute", "time"])
                elif "end_time" in all_fields:
                    end_time = event.end_time.strftime("%H:%M")
                    label["time"] = f"{end_time}"
                    done.update(["end_time", "hour", "minute", "time"])
                elif "time" in all_fields:
                    time = event.time.strftime("%H:%M")
                    label["time"] = f"{time}"
                    done.update(["hour", "minute", "time"])

                if "city" in all_fields and "country" in all_fields:
                    label["city"] = f"{event.city.title()}, {event.country.title()}"
                    done.update(["city", "country"])
                elif "city" in all_fields and "region" in all_fields:
                    label["city"] = event.region.title()
                    done.update(["city", "region"])
                elif "region" in all_fields and "country" in all_fields:
                    label["city"] = event.region.title()
                    done.update(["region", "country"])

                # The rest of the fields
                for field in all_fields:
                    if field in EXCLUDE_FIELDS or field in done:
                        continue
                    value = getattr(event, field, None)
                    if isinstance(value, list):
                        value = ", ".join(value)
                    if value:
                        value = str(value)
                        label[field] = value.title()

                # Make time and date on the same line
                if "time" in label and "date" in label:
                    label["time"] += f", {label['date']}"
                    del label["date"]

                event.name = "\n".join(label)
    return results
