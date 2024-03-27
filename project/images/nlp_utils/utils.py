from project.images.nlp_utils.common import COUNTRIES, LOWERCASE_COUNTRIES, STOP_WORDS
from project.images.nlp_utils.time import Tags


def rreplace(s: str, old: str, new: str, occurrence: int) -> str:
    """
    Replace the last occurrence of a substring in a string
    """
    li = s.rsplit(old, occurrence)
    return new.join(li)


def remove_keywords(query: str, keywords: list[str]) -> str:
    """
    Remove keywords from the query
    """
    for keyword in keywords:
        query = rreplace(query, keyword, "", 1)
    return query


def parse_tags(query: str) -> tuple[str, dict]:
    """
    Parse tags in the query
    """
    # replace --
    query = query.replace("—", "--")

    words = query.split()

    # possible tags
    tags = {
        "--disable-region": [],
        "--disable-location": [],
        "--disable-time": [],
        "--negative": [],
    }

    # Get all indexes of tags
    all_indexes = [i for i, word in enumerate(words) if word in tags]
    if all_indexes:
        for i, begin_index in enumerate(all_indexes):
            # Find the index of the next tag
            end_index = all_indexes[i + 1] if i + 1 < len(all_indexes) else len(words)

            # Add the arguments of the tag to the list of disabled information
            tag = words[begin_index]
            tags[tag].extend(
                [
                    word.strip()
                    for word in " ".join(words[begin_index + 1 : end_index]).split(",")
                ]
            )

        words = words[: all_indexes[0]]

    # Join the remaining words back into a modified query string
    modified_query = " ".join(words)

    # Example output for demonstration purposes
    result = {
        "disabled_locations": tags["--disable-location"],
        "disabled_regions": tags["--disable-region"],
        "disabled_times": tags["--disable-time"],
        "negative": tags["--negative"],
    }
    print(result)
    return modified_query, result


def postprocess_countries(countries: list[str]) -> list[str]:
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


def choose_countries_for_map(countries: list[str]) -> list[dict]:
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


def strip_stopwords(sentence):
    """
    Remove stopwords from the end of a sentence
    """
    if sentence:
        words = sentence.split()
        for i in reversed(range(len(words))):
            if words[i].lower() in STOP_WORDS:
                words.pop(i)
            else:
                break
        return " ".join(words)
    return ""


def get_visual_text(text: str, unprocessed_words: list[Tags]):
    """
    Get the visual text
    """
    # Get the visual text
    last_non_prep = 0
    visual_text = ""
    for i in range(len(unprocessed_words) + 1):
        if (
            unprocessed_words[-i][1] not in ["DT", "IN"]
            and unprocessed_words[-i][0] not in STOP_WORDS
        ):
            last_non_prep = -i
            break
    if last_non_prep > 1:
        visual_text = " ".join([word[0] for word in unprocessed_words[:last_non_prep]])
    else:
        visual_text = " ".join([word[0] for word in unprocessed_words])

    # Remove stopwords from the end of the sentence
    visual_text = visual_text.strip(", ?")
    visual_text = strip_stopwords(visual_text)
    visual_text = visual_text.strip(", ?")

    return visual_text
