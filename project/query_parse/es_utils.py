from datetime import timedelta
from typing import List, Sequence, Tuple
from rich import print

from results.models import EventResults

from query_parse.constants import GPS_NORMAL_CASE, MAP_VISUALISATION
from query_parse.time import MONTHS
from query_parse.types import (
    GPS,
    ESAndFilters,
    ESCombineFilters,
    ESEmbedding,
    ESFilter,
    ESFuzzyMatch,
    ESGeoBoundingBox,
    ESGeoDistance,
    ESMatch,
    ESOrFilters,
    ESRangeFilter,
    LocationInfo,
    TimeInfo,
    VisualInfo,
)
from query_parse.types.lifelog import Mode, TimeCondition
from query_parse.visual import encode_text


def range_filter(
    start: int | str | float, end: int | str | float, field: str, boost: float = 1.0
) -> ESRangeFilter:
    return ESRangeFilter(field=field, start=start, end=end, boost=boost)


def get_time_filters(
    timeinfo: TimeInfo, mode: Mode = Mode.event
) -> Tuple[ESOrFilters, dict]:
    """
    We are using bool query to get the minimum_should_match parameter.
    Because we want an OR operation between the time filters, we use the should clause.
    """

    s, e = timeinfo.time
    has_child = False
    time_filters = []
    if s != 0 or e != 24 * 3600:
        if mode == "image":
            # for images, we only have the time
            if s <= e:
                time_filters = [
                    range_filter(s, e, "seconds_from_midnight"),
                ]
            else:
                time_filters = [
                    range_filter(0, e, "seconds_from_midnight"),
                    range_filter(s, 24 * 3600, "seconds_from_midnight"),
                ]
        elif mode == "event":
            if s <= e:
                # OR (should queries)
                time_filters = [
                    range_filter(s, e, "start_seconds_from_midnight"),
                    range_filter(s, e, "end_seconds_from_midnight"),
                    ESAndFilters(
                        queries=[
                            range_filter(0, s, "start_seconds_from_midnight"),
                            range_filter(e, 24 * 3600, "end_seconds_from_midnight"),
                        ],
                    ),
                ]
                has_child = True
            else:  # either from midnight to end or from start to midnight
                time_filters = [
                    range_filter(0, e, "start_seconds_from_midnight"),
                    range_filter(0, e, "end_seconds_from_midnight"),
                    range_filter(s, 24 * 3600, "start_seconds_from_midnight"),
                    range_filter(s, 24 * 3600, "end_seconds_from_midnight"),
                ]

        # combine the date filters using a bool should clause
    bool_time_filters = ESOrFilters(
        name="TIME", queries=time_filters, has_child=has_child
    )
    query_visualisation = {}
    if s != 0 or e != 24 * 3600:
        # format the time for visualisation
        start_time_text = f"{s // 3600:02d}:{(s % 3600) // 60:02d}"
        end_time_text = f"{e // 3600:02d}:{(e % 3600) // 60:02d}"
        query_visualisation = {"TIME": [f"{start_time_text} - {end_time_text}"]}
    return bool_time_filters, query_visualisation


def get_duration_filters(timeinfo: TimeInfo) -> ESOrFilters:
    """
    Get the duration filters
    """
    duration_filters = []
    if timeinfo.duration:
        duration_filters = [
            range_filter(
                timeinfo.duration // 2, round(timeinfo.duration * 1.5), "duration", 0.1
            ),
            range_filter(
                timeinfo.duration // 2,
                round(timeinfo.duration * 1.5),
                "group_duration",
                0.05,
            ),
        ]
    return ESOrFilters(name="DURATION", queries=duration_filters)


def get_weekday_filters(timeinfo: TimeInfo) -> ESOrFilters:
    """
    Get the weekday filters
    """
    weekday_filters = []
    if timeinfo.weekdays:
        weekday_filters = [
            ESFilter(field="weekday", value=weekday) for weekday in timeinfo.weekdays
        ]
    return ESOrFilters(name="WEEKDAY", queries=weekday_filters)


def get_timestamp_filters(timeinfo: TimeInfo, mode: Mode = Mode.event) -> ESOrFilters:
    """
    Get the timestamp filters
    (usually before/after a certain date)
    """
    timestamp_filters = []
    if timeinfo.timestamps:
        for timestamp, condition in timeinfo.timestamps:
            if condition == "before":
                if mode == Mode.event:
                    timestamp_filters.append(
                        range_filter(0, timestamp, "end_timestamp", 0.1)
                    )
                elif mode == Mode.image:
                    timestamp_filters.append(
                        range_filter(0, timestamp, "timestamp", 0.1)
                    )
            elif condition == "after":
                if mode == Mode.event:
                    timestamp_filters.append(
                        range_filter(timestamp, 24 * 3600, "start_timestamp", 0.1)
                    )
                elif mode == Mode.image:
                    timestamp_filters.append(
                        range_filter(timestamp, 24 * 3600, "timestamp", 0.1)
                    )
            elif condition == "around":
                if mode == "image":
                    timestamp_filters.append(
                        range_filter(
                            timestamp - 2 * 3600,
                            timestamp + 2 * 3600,
                            "end_timestamp",
                            0.1,
                        )
                    )
                elif mode == "event":
                    timestamp_filters.append(
                        range_filter(
                            timestamp - 2 * 3600,
                            timestamp + 2 * 3600,
                            "start_timestamp",
                            0.1,
                        )
                    )
    return ESOrFilters(name="TIMESTAMP", queries=timestamp_filters)


def get_date_filters(timeinfo: TimeInfo) -> Tuple[ESOrFilters, dict]:
    """
    Get the date filters
    """
    # Preprocess the dates:
    date_filters = []
    years = set()
    common_year = None
    # if there is only one year available, add that year to all dates
    for date in timeinfo.dates:
        if date.year:
            years.add(date.year)

    if len(years) == 1:
        common_year = years.pop()
        # remove the date with the year-only
        if len(timeinfo.dates) > 1:
            new_dates = []
            for date in timeinfo.dates:
                y, m, d = date.year, date.month, date.day
                if not m and not d and y:
                    continue
                new_dates.append(date)
            timeinfo.dates = new_dates
    query_visualisation = {}
    # create a list to store the date filters
    if timeinfo.dates:
        query_visualisation = {"DATE": []}
        for date in timeinfo.dates:
            y, m, d = date.year, date.month, date.day
            if not y and common_year:
                y = common_year

            field = ""
            date_string = ""
            visualised_date = ""

            # date format in database is yyyy/MM/dd HH:mm:00Z
            if y and m and d:
                date_string = f"{y}/{m:02d}/{d:02d}"
                field = "date"
                visualised_date = f"{d:02d}/{m:02d}/{y}"
            elif y and m:
                date_string = f"{m:02d}/{y}"
                field = "month_year"
                visualised_date = f"{MONTHS[m - 1].capitalize()}/{y}"
            elif y and d:
                date_string = f"{d:02d}/{y}"
                field = "day_year"
                visualised_date = f"Day {d:02d} from {y}"
            elif m and d:
                date_string = f"{d:02d}/{m:02d}"
                field = "day_month"
                visualised_date = f"{d:02d} {MONTHS[m - 1].capitalize()}"
            elif y:
                date_string = f"{y}"
                field = "year"
                visualised_date = f"Year {y}"
            elif m:
                date_string = f"{m:02d}"
                field = "month"
                visualised_date = MONTHS[m - 1].capitalize()
            elif d:
                date_string = f"{d:02d}"
                field = "day"
                visualised_date = f"Day {d:02d}"

            date_filters.append(ESFilter(field=field, value=date_string, boost=0.1))
            query_visualisation["DATE"].append(visualised_date)

        # combine the date filters using a bool should clause
    return ESOrFilters(name="DATE", queries=date_filters), query_visualisation


def get_temporal_filters(
    timeinfo: TimeInfo, mode: Mode = Mode.event
) -> Tuple[Sequence[ESCombineFilters], dict]:
    """
    Get the time filters
    """
    time_filters, time_visualisation = get_time_filters(timeinfo, mode)
    date_filters, date_visualisation = get_date_filters(timeinfo)
    timestamp_filters = get_timestamp_filters(timeinfo, mode)
    duration_filters = get_duration_filters(timeinfo)
    weekday_filters = get_weekday_filters(timeinfo)

    # For visualisation
    query_visualisation = {}
    query_visualisation.update(time_visualisation)
    query_visualisation.update(date_visualisation)
    if timeinfo.weekdays:
        query_visualisation["WEEKDAY"] = timeinfo.weekdays

    # combine the time filters using a bool should clause
    temporal_filters = [
        time_filters,
        date_filters,
        timestamp_filters,
        duration_filters,
        weekday_filters,
    ]

    return temporal_filters, query_visualisation


def get_location_search_parameters(location: str) -> Tuple[str, str]:
    dist = "0.5km"  # How far from the location should we search
    pivot = "5m"  # How far from the location should we pivot
    if "airport" in location or "home" in location:
        dist = "2km"
        pivot = "200m"
    elif "dcu" in location:
        dist = "1km"
        pivot = "100m"
    return dist, pivot


def get_location_filters(locationinfo: LocationInfo) -> Sequence[ESCombineFilters]:
    """
    Get the location filters
    """
    locations = locationinfo.locations
    regions = locationinfo.regions
    location_types = locationinfo.location_types
    gps_bounds = locationinfo.gps_bounds

    queries = []
    for loc in locations:
        place = GPS_NORMAL_CASE[loc]
        dist, pivot = get_location_search_parameters(loc)

        # GeoDistance query
        for place_iter, (lat, lon) in MAP_VISUALISATION:
            if place == place_iter:
                queries.append(
                    ESGeoDistance(lat=lat, lon=lon, distance=dist, pivot=pivot)
                )

        # Match query
        queries.append(ESMatch(field="location", query=loc, boost=0.0005))

    place_filters = ESOrFilters(name="PLACE", queries=queries)
    place_type_filters = ESMatch(
        name="LOCATION_TYPE",
        field="location_type",
        query=" ".join(location_types),
        boost=0.001,
    )

    queries = []
    for region in regions:
        queries.append(ESFilter(field="region", value=region))
    region_filters = ESOrFilters(name="REGION", queries=queries)

    if gps_bounds:
        lon1, lat2, lon2, lat1 = gps_bounds
        top_left = GPS(lat=lat1, lon=lon1)
        bottom_right = GPS(lat=lat2, lon=lon2)
        region_filters.queries.append(
            ESGeoBoundingBox(top_left=top_left, bottom_right=bottom_right)
        )

    return [place_filters, place_type_filters, region_filters]


def get_visual_filters(visual_info: VisualInfo) -> Sequence[ESCombineFilters]:
    embedding = ESEmbedding(field="google_vector")
    ocr = ESFuzzyMatch(field="ocr", boost=0.001)
    concepts = ESMatch(field="descriptions", boost=0.01)
    if visual_info.text:
        print("Visual info text:", visual_info.text)
        encoded_query = encode_text(visual_info.text).tolist()
        embedding.embedding = encoded_query
        ocr.query = visual_info.text
        concepts.query = " ".join(visual_info.concepts)
    return [embedding, ocr, concepts]


# =================================================================== #
# TEMPORAL QUERIES
# =================================================================== #
def get_conditional_time_filters(
    results: EventResults, condition: TimeCondition
) -> List[ESRangeFilter]:
    """
    Get the time filters based on the events
    condition: TimeCondition (before, after)
    time_limit: float (in hours)
    """
    filters: List[ESRangeFilter] = []
    time_limit = condition.time_limit_float
    leeway = timedelta(hours=min(time_limit - 1, time_limit / 2))
    upper_limit = timedelta(hours=max(time_limit + 0.5, time_limit * 1.5))
    for event in results.events:
        match condition.condition:
            case "before":
                filters.append(
                    range_filter(
                        (event.start_time - upper_limit).timestamp(),
                        (event.start_time - leeway).timestamp(),
                        "end_timestamp",
                    )
                )
            case "after":
                filters.append(
                    range_filter(
                        (event.end_time + leeway).timestamp(),
                        (event.end_time + upper_limit).timestamp(),
                        "start_timestamp",
                    )
                )
            case _:
                raise ValueError("Invalid time condition")
    return filters
