"""
Location module
"""

import re
from typing import List, Tuple

from results.models import Visualisation

from .constants import (
    COUNTRIES,
    GPS_NORMAL_CASE,
    LOCATION_INFOS,
    LOCATIONS,
    LOWERCASE_COUNTRIES,
    REGIONS,
)
from .types import LocationInfo
from .utils import is_enabled, search_keywords


def postprocess_countries(countries: List[str]) -> List[str]:
    """
    Postprocess the countries
    so that they can be used for filtering and visualisation
    """
    maps = {
        "korea": "south korea",  # I assume you want South Korea
        "england": "united kingdom",  # England doesn't exist in the geojson
    }

    for i, country in enumerate(countries):
        if country in maps:
            countries[i] = maps[country]

    return countries


def choose_countries_for_map(countries: List[str]) -> List[dict]:
    """
    Postprocess the countries
    so that they can be used for filtering and visualisation
    """
    geojsons = []
    for country in countries:
        if country in LOWERCASE_COUNTRIES:
            country = LOWERCASE_COUNTRIES[country]  # or just country.title()
            geojsons.append({"country": country, "geojson": COUNTRIES[country]})
    return geojsons


def search_possible_location(text: str, disabled: List[str] = []) -> List[str]:
    """
    Search for possible locations in the text
    Based on partial matches of the location names
    """
    results = []
    for location in LOCATIONS:
        if is_enabled(location, disabled):
            for extra in LOCATIONS[location]:
                if is_enabled(extra, disabled):
                    if re.search(r"\b" + re.escape(extra) + r"\b", text, re.IGNORECASE):
                        if location not in results:
                            results.append(location)
                        break
    return results


def search_for_locations(
    text: str, parsed: dict
) -> Tuple[str, LocationInfo, Visualisation]:
    """
    Search for locations in the text
    """
    clean_query = text

    def search_words(wordset, disabled=[]):
        return search_keywords(wordset, text, disabled)

    locations = search_words(LOCATIONS, parsed["disabled_locations"])
    # clean_query = remove_keywords(text, locations)
    location_types = search_words(LOCATION_INFOS, parsed["disabled_locations"])

    regions = search_words(REGIONS, parsed["disabled_regions"])
    # clean_query = remove_keywords(clean_query, regions)
    regions = postprocess_countries(regions)

    info = LocationInfo(
        locations=locations, regions=regions, location_types=location_types
    )

    query_visualisation = Visualisation()
    if locations:
        query_visualisation.locations = locations
        query_visualisation.map_locations = [
            GPS_NORMAL_CASE[location] for location in locations
        ]
    else:
        query_visualisation.suggested_locations = search_possible_location(
            text, parsed["disabled_locations"]
        )

    # Regions
    if regions:
        query_visualisation.regions = regions
        query_visualisation.map_countries = choose_countries_for_map(regions)

    return clean_query, info, query_visualisation
