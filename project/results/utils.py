from datetime import datetime
from functools import cmp_to_key
from typing import Dict, Iterable, List
from database.main import image_collection, scene_collection

from configs import DERIVABLE_FIELDS, EXCLUDE_FIELDS, ISEQUAL, MAXIMUM_EVENT_TO_GROUP
from pydantic import BaseModel, InstanceOf, validate_call
from query_parse.types.lifelog import MaxGap, RelevantFields
from query_parse.visual import score_images
from retrieval.async_utils import timer
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


class FakeEvent(BaseModel):
    start_time: datetime
    end_time: datetime
    location: str
    region: List[str]
    country: str
    location_info: str


def index_derived_fields():
    """
    Index the derived fields
    """
    for image in image_collection.find():
        for field in DERIVABLE_FIELDS:
            fake_event = FakeEvent(
                start_time=image["time"],
                end_time=image["time"],
                location=image["location"],
                region=image["region"],
                country=image["country"],
                location_info=image["location_info"],
            )
            if field in image:
                continue
            image[field] = DERIVABLE_FIELDS[field](fake_event)
        image_collection.update_one({"_id": image["_id"]}, {"$set": image})
    print("[green]Indexed derived fields for images[/green]")
    for scene in scene_collection.find():
        for field in DERIVABLE_FIELDS:
            fake_event = FakeEvent(
                start_time=scene["start_time"],
                end_time=scene["end_time"],
                location=scene["location"],
                region=scene["region"],
                country=scene["country"],
                location_info=scene["location_info"],
            )
            if field in scene:
                continue
            scene[field] = DERIVABLE_FIELDS[field](fake_event)
        scene_collection.update_one({"_id": scene["_id"]}, {"$set": scene})
    print("[green]Indexed derived fields for scenes[/green]")

# index_derived_fields()

def custom_compare_function(
    event1: Event, event2: Event, fields: Iterable[str], max_gap: MaxGap
) -> int:
    """
    Custom compare function
    """
    # Assert if the two events are the same group first
    # if not then just compare the scene ids
    equal = True
    ignore_fields = []

    # Check the time gap
    if max_gap.time_gap is not None and max_gap.time_gap.unit == "none":
        time_gap = max_gap.time_gap
        ignore_fields.append(time_gap.unit)
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

    for field in fields:
        if field in ignore_fields:
            continue
        cmp_field = field
        if field not in ISEQUAL:
            cmp_field = "*"
        if not ISEQUAL[cmp_field](getattr(event1, field), getattr(event2, field)):
            equal = False
            break

    if equal:
        return 0

    # If the two events are not the same group then compare the sceneid
    return 1 if event1.scene > event2.scene else -1


@validate_call
def merge_events(
    results: EventResults, relevant_fields: RelevantFields = RelevantFields()
) -> EventResults:
    """
    Merge the events
    """
    # Check if any of the groupby should be calculated
    available_fields = set(type(results.events[0]).model_fields)
    derive_fields = []
    to_remove = set()
    for criteria in relevant_fields.merge_by:
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
    groupby = set(relevant_fields.merge_by)
    groupby = groupby.difference(to_remove)

    # Derive the fields
    if derive_fields:
        results.events = deriving_fields(results.events, derive_fields)

    if len(results.events) == 1:
        return results

    # If groupby is empty then group by "group"
    if not groupby:
        groupby = set(["group"])

    cmp = lambda x, y: custom_compare_function(x, y, groupby, relevant_fields.max_gap)

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

    if relevant_fields.sort_by:
        for sort in relevant_fields.sort_by[::-1]:
            new_results, new_scores = zip(
                *sorted(
                    zip(new_results, new_scores),
                    key=lambda x: getattr(x[0], sort.field),
                    reverse=sort.order == "desc",
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


def merge_scenes_and_images(scenes: EventResults, images: EventResults) -> EventResults:
    all_scenes: Dict[str, Event] = {}
    for scene, score in zip(scenes.events, scenes.scores):
        scene.image_scores = [score] * len(scene.images)
        all_scenes[scene.scene] = scene

    for image, score in zip(images.events, images.scores):
        image.image_scores = [score]
        if image.scene not in all_scenes:
            all_scenes[image.scene] = image
        else:
            scene_images = [img.src for img in all_scenes[image.scene].images]
            if image.images[0].src not in scene_images:
                all_scenes[image.scene].images.extend(image.images)
                all_scenes[image.scene].image_scores.extend(image.image_scores)

    all_scene_scores = []
    for scene in all_scenes.values():
        # Reorder the images by their scores
        scene_images = scene.images
        scene_scores = scene.image_scores
        if scene_images:
            scene.images, scene.image_scores = zip(
                *sorted(
                    zip(scene_images, scene_scores),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
            scene.image_scores = list(scene.image_scores)
            scene.images = list(scene.images)
        all_scene_scores.append(max(scene_scores))

    # Sort the scenes by their scores
    sorted_scenes, sorted_scores = zip(
        *sorted(
            zip(all_scenes.values(), all_scene_scores),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    return EventResults(
        events=list(sorted_scenes),
        scores=list(sorted_scores),
        min_score=min(scenes.min_score, images.min_score),
        max_score=max(scenes.max_score, images.max_score),
    )

@timer("limit_images_per_event")
def limit_images_per_event(
    results: GenericEventResults, text_query: str, max_images: int
) -> GenericEventResults:
    """
    Limit the number of images per event
    This is achieved by selecting the images with the highest score
    and highest relevance to the query
    """
    encoded_query = encode_query(text_query)
    for generic_event in results.events:
        for event in generic_event.custom_iter():  # might be a doublet or triplet

            images: List[Image] = event.images
            scores = event.image_scores

            if len(images) <= max_images:
                continue

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

        # We have to go from specific to general
        # If the specific field is available then ignore the general fields
        filtered_fields = {"place": [], "time": [], "location": []}

        # 1st line: specific place:
        for field in ["location", "location_info"]:
            if field in all_fields:
                filtered_fields["place"].append(field)
        if not filtered_fields:
            for field in ["place", "place_info"]:
                if field in all_fields:
                    filtered_fields["place"].append(field)

        # 2nd line: time fields
        if "time" in all_fields:
            filtered_fields["time"].append("time")
            all_fields.discard("start_time")
            all_fields.discard("end_time")
            all_fields.discard("hour")
            all_fields.discard("minute")

        for field in ["start_time", "end_time"]:
            if field in all_fields:
                filtered_fields["time"].append(field)
                all_fields.discard("hour")
                all_fields.discard("minute")

        for field in ["weekday", "date", "month", "year"]:
            if field in all_fields:
                filtered_fields["time"].append(field)

        # 3rd line: location fields
        # Location fields
        if "city" in all_fields and "country" in all_fields:
            all_fields.discard("region")

        if "region" in all_fields:
            all_fields.discard("city")
            all_fields.discard("country")

        for field in ["city", "region", "country"]:
            if field in all_fields:
                filtered_fields["location"].append(field)

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
                    city = ", ".join(event.city).title()
                    label["city"] = f"{city}, {event.country.title()}"
                    done.update(["city", "country"])
                elif "region" in all_fields:
                    label["city"] = ", ".join(event.region).title()
                    done.update(["region", "country"])

                # The rest of the fields
                for field in all_fields:
                    if field in EXCLUDE_FIELDS or field in done:
                        continue
                    value = getattr(event, field, None)
                    if isinstance(value, list):
                        value = ", ".join(value)
                    if value:
                        value = str(value).strip()
                        if value:
                            label[field] = value.title()

                # # Create the label
                # label = [
                #     f"<strong>{format_key(k)}</strong>: {v}" for k, v in label.items()
                # ]

                # Create the label
                lines = []
                line = []
                for key in filtered_fields["place"]:
                    if key in label and label[key]:
                        if key == "location" or key == "place":
                            line.append(f"<strong>{label[key]}</strong>")
                        else:
                            line.append(label[key])
                if line:
                    lines.append(", ".join(line))

                line = []
                for key in filtered_fields["time"]:
                    if key in label and label[key]:
                        line.append(label[key])
                if line:
                    lines.append("@" + ", ".join(line))

                line = []
                for key in filtered_fields["location"]:
                    if key in label and label[key]:
                        line.append(label[key])
                if line:
                    lines.append(", ".join(line))

                event.name = "\n".join(lines)
                break
    return results


def format_key(key: str) -> str:
    """
    Format the key
    """
    return key.replace("_", " ").title()
